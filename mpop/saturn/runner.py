#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Batch run stuff.
"""
import datetime
import getopt
import logging
import sys

from mpop.channel import NotLoadedError
from mpop.satellites import get_satellite_class
from mpop.saturn.tasklist import TaskList


LOG = logging.getLogger("runner")

def usage(scriptname):
    """Print usefull information for running the script.
    """
    print("""
usage: %s [options]

Available options:
    -r
    --rgb
        Output only the pure rgb images from the product list
    -p
    --pge
        Output only the cloudproduct derived images from the product list
    -d <date_str>
    --date <date_str>
        Run on the specified date (eg. 200910081430)
    -a <area_name>
    --area <area_name>
        Run on the specified area (eg. eurol)
    -c <name>
    --composite <name>
        Output the specified composite. Available composites are:
        overview
        natural
        fog
        nightfog
        convection
        airmass
        ir9
        wv_low
        wv_high
        greensnow
        redsnow
        cloudtop
        hr_overview
        PGE02
        PGE02b
        PGE02bj
        PGE02c
        PGE02cj
        PGE02d
        PGE02e
        PGE03
        CtypeHDF
        NordRad
        CtthHDF
    -v
        Be verbose
    --vv
        Be very verbose
    -h
    --help
        Print this message
          """%scriptname)

def parse_options():
    """Parse command line options.
    """

    time_slots = []
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(levelname)s: %(asctime)s :"
                                           " %(name)s] %(message)s",
                                           '%Y-%m-%d %H:%M:%S'))
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "hd:va:c:pr",
                                   ["vv", "help", "date=", "pge", "rgb",
                                    "area=", "composite="])
        del args
    except getopt.GetoptError, err:
        print err
        usage(sys.argv[0])
        sys.exit(2)

    mode = set([])
    areas = set([])
    composites = set([])

    for opt, arg in opts:
        if opt == "-v":
            console.setLevel(logging.INFO)
            logging.getLogger('').addHandler(console)
        elif opt == "--vv":
            console.setLevel(logging.DEBUG)
            logging.getLogger('').addHandler(console)
        elif opt in ("-h", "--help"):
            usage(sys.argv[0])
            sys.exit()
        elif opt in ("-d", "--date"):
            time_slots.append(datetime.datetime.strptime(arg, "%Y%m%d%H%M"))
        elif opt in ("-p", "--pge"):
            mode |= set(["pge"])
        elif opt in ("-r", "--rgb"):
            mode |= set(["rgb"])
        elif opt in ("-a", "--area"):
            areas |= set([arg])
        elif opt in ("-c", "--composites"):
            composites |= set([arg])
        else:
            raise ValueError("Option %s not recognized.")

    return time_slots, mode, areas, composites


class SequentialRunner(object):
    """Runs scenes in a sequential order, as opposed to parallelized running.
    """

    def __init__(self, satellite, tasklist, precompute=False):
        if isinstance(tasklist, str):
            tasklist = TaskList(tasklist)
        self.tasklist = tasklist
        self.data = None
        self.satellite = satellite[0]
        self.number = satellite[1]
        self.variant = satellite[2]
        self.klass = get_satellite_class(self.satellite,
                                         self.number,
                                         self.variant)
        self.precompute = precompute
            
    def run_from_cmd(self):
        """Batch run mpop.
        """
        time_slots, mode, areas, composites = parse_options()
                
        tasklist = self.tasklist.shape(self.klass, mode, areas, composites)
        
        for time_slot in time_slots:
            self.data = self.klass(time_slot=time_slot)
            prerequisites = tasklist.get_prerequisites(self.klass)
            self.data.load(prerequisites)
            self.run_from_data(tasklist)
                                

    def run_from_data(self, tasklist=None):
        """Run on given data.
        """
        if tasklist is None:
            tasklist = self.tasklist
        for area, productlist in tasklist.items():
            prerequisites = tasklist.get_prerequisites(self.klass, area)
            local_data = self.data.project(area, prerequisites, self.precompute,
                                           mode="nearest")
            for product, flist in productlist.items():
                fun = getattr(local_data, product)
                flist = flist.put_date(local_data.time_slot)
                if local_data.orbit is not None:
                    flist = flist.put_metadata({"orbit": int(local_data.orbit)})
                try:
                    LOG.debug("Doing "+product+".")
                    img = fun()
                    flist.save_object(img)
                    del img
                except (NotLoadedError, KeyError), err:
                    LOG.warning("Error in "+product+": "+str(err))
                    LOG.info("Skipping "+product)
            del local_data

    def run_from_local_data(self, tasklist=None):
        """Run on given local data (already projected).
        """
        if tasklist is None:
            tasklist = self.tasklist
        area_name = self.data.area_id or self.data.area_def.area_id
        tasks, dummy = tasklist.split(area_name)
        for product, flist in tasks[area_name].items():
            fun = getattr(self.data, product)
            flist = flist.put_date(self.data.time_slot)
            if self.data.orbit is not None:
                flist = flist.put_metadata({"orbit": int(self.data.orbit)})
            try:
                LOG.debug("Doing "+product+".")
                img = fun()
                flist.save_object(img)
                del img
            except (NotLoadedError, KeyError), err:
                LOG.warning("Error in "+product+": "+str(err))
                LOG.info("Skipping "+product) 
        
if __name__ == "__main__":
    SR = SequentialRunner(["metop", "02", "global"],
                          "/local_disk/usr/src/mpop/etc/meteosat09_products.py")
    SR.run_from_cmd()
    
