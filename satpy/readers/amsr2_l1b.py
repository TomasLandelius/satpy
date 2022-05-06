#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2018 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Reader for AMSR2 L1B files in HDF5 format."""

# test file /nobackup/smhid15/sm_tland/amsr2/201705/GW1AM2_201705011030_043A_L1SGBTBR_2220220.h5

from satpy.readers.hdf5_utils import HDF5FileHandler
import re, numpy as np

def split_xGval(s):
    """Parse CoefficientAxx, CoRegistrationParameterAx
    or CalibrationCurveCoefficient#x into dictionary.
    """
    xGyy_z = re.compile('(\d+G[AB]?[HV]?)\-([-]?\d+\.\d+)')

    d = {}
    for ghz, val in xGyy_z.findall(s):
        d[ghz] = float(val)
    return d

def coreg_rad(lon_89a, lat_89a, a1, a2, ECCSQ=6.69437999014e-3):
    """Use the coregistration parameters A1 and A2 to calculate
    accurate longitude, latitude information for 'lower' channels.
    ECCSQ: eccentricity squared.
    Code originates from Lothar Meyer-Lerbs thesis Gridding of AMSR-E Satellite Data; 
    http://www.iup.uni-bremen.de/PEP_master_thesis/thesis_2005/Thesis_LotharMeyer-Lerbs.pdf
    The method is also described in the AMSR2 Level 1 Product Format Specification; 
    https://gportal.jaxa.jp/gpr/assets/mng_upload/GCOM-W/AMSR2_Level1_Product_Format_EN.pdf
    """
    # to radians
    lon_89a = lon_89a*np.pi/180
    lat_89a = lat_89a*np.pi/180
    # odd for array base 1    
    odd = (np.arange(lon_89a.shape[1]/2)*2).astype('int')
    even = odd + 1
    lon1 = np.take(lon_89a, odd, axis=1)
    lon2 = np.take(lon_89a, even, axis=1)
    lat1 = np.take(lat_89a, odd, axis=1)
    lat2 = np.take(lat_89a, even, axis=1)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)
    exx, exy, exz = clat1*np.cos(lon1), clat1*np.sin(lon1), np.sin(lat1)
    p2x, p2y, p2z = clat2*np.cos(lon2), clat2*np.sin(lon2), np.sin(lat2)
    # ez = ex x p2/
    ezx = exy*p2z - exz*p2y
    ezy = exz*p2x - exx*p2z
    ezz = exx*p2y - exy*p2x
    normez = np.sqrt(ezx*ezx + ezy*ezy + ezz*ezz)
    # ez /= Norm(ez)/
    ezx, ezy, ezz = ezx/normez, ezy/normez, ezz/normez
    # ey = ez x ex/
    eyx = ezy*exz - ezz*exy
    eyy = ezz*exx - ezx*exz
    eyz = ezx*exy - ezy*exx
    theta = np.arccos(exx*p2x + exy*p2y + exz*p2z)
    a1t, a2t = a1*theta, a2*theta
    c1t, s1t = np.cos(a1t), np.sin(a1t)
    c2t, s2t = np.cos(a2t), np.sin(a2t)
    ptx = c2t*(c1t*exx + s1t*eyx) + s2t*ezx
    pty = c2t*(c1t*exy + s1t*eyy) + s2t*ezy
    ptz = c2t*(c1t*exz + s1t*eyz) + s2t*ezz
    lon = np.arctan2(pty, ptx)
    lat = np.arctan(ptz/(np.hypot(ptx, pty)*(1.0 - ECCSQ)))
    # return degrees
    return lon*180/np.pi, lat*180/np.pi


class AMSR2L1BFileHandler(HDF5FileHandler):
    """File handler for AMSR2 l1b."""

    def get_metadata(self, ds_id, ds_info):
        """Get the metadata."""
        var_path = ds_info['file_key']
        info = getattr(self[var_path], 'attrs', {})
        info.update(ds_info)
        info.update({
            "shape": self.get_shape(ds_id, ds_info),
            "units": self[var_path + "/attr/UNIT"],
            "platform_name": self["/attr/PlatformShortName"],
            "sensor": self["/attr/SensorShortName"].lower(),
            "start_orbit": int(self["/attr/StartOrbitNumber"]),
            "end_orbit": int(self["/attr/StopOrbitNumber"]),
        })
        info.update(ds_id.to_dict())
        return info

    def get_shape(self, ds_id, ds_info):
        """Get output shape of specified dataset."""
        var_path = ds_info['file_key']
        shape = self[var_path + '/shape']
        if ((ds_info.get('standard_name') == "longitude" or ds_info.get('standard_name') == "latitude") and
            (ds_id['name'].split('_')[-1][:2] != '89')):
            return shape[0], int(shape[1] / 2)
        return shape

    def get_dataset(self, ds_id, ds_info):
        """Get output data and metadata of specified dataset."""
        var_path = ds_info['file_key']
        fill_value = ds_info.get('fill_value', 65535)
        metadata = self.get_metadata(ds_id, ds_info)
        data = self[var_path]
        data = data * self[var_path + "/attr/SCALE FACTOR"]
        if ((ds_info.get('standard_name') == "longitude" or
             ds_info.get('standard_name') == "latitude") and
            (ds_id['name'].split('_')[-1][:2] != '89')):
            # map yaml name freq to coreg freq name
            # yaml: 6.9 7.3 10.7 18.7 23.8 36.6
            # coreg: 6G 7G 10G 18G 23G 36G
            ch = ds_info.get('name').split('_')[-1].split('.')[0] + 'G'
            a1= split_xGval(self["/attr/CoRegistrationParameterA1"])
            a2= split_xGval(self["/attr/CoRegistrationParameterA2"])
            # we need both lon and lat for 89a to find lon or lat
            # for other channels.
            lo89a = self["Longitude of Observation Point for 89A"]
            la89a = self["Latitude of Observation Point for 89A"]
            lo89a = lo89a * self["Longitude of Observation Point for 89A/attr/SCALE FACTOR"]
            la89a = la89a * self["Latitude of Observation Point for 89A/attr/SCALE FACTOR"]
            # insert nan to catch missing data in output
            lo89a = np.where(lo89a == fill_value, np.nan, lo89a)
            la89a = np.where(la89a == fill_value, np.nan, la89a)
            lon, lat = coreg_rad(lo89a, la89a, a1[ch], a2[ch])
            if (ds_info.get('standard_name') == "longitude"):
                data = data[:,::2]
                lon = np.where(np.isnan(lon), fill_value, lon)
                data[:,:] = lon
            else:
                data = data[:,::2]
                lat = np.where(np.isnan(lat), fill_value, lat)
                data[:,:] = lat
        data = data.where(data != fill_value)
        data.attrs.update(metadata)
        return data






