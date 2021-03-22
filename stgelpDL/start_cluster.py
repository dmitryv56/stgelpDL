#!/usr/bin/python3
""" Stand-alone script to load Clusterizing  Green Energy Load Deep Learning predictor.
    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled predictor>\stgelpDL  (Windows 10)
    or
        cd <Unix path to installed predictor>/stgelpDL      (Ubuntu 18.*),

    and put following command:
           python start_cluster.py -v
    To read  a version of predictor
            python start_cluster.py --version
    To read a help
            python start_cluster.py --help
    or
            python start_cluster.py -h

"""

import sys

from tensorflow import random

from predictor.cfg import MAGIC_SEED
from clustgelDL.main import main

if __name__ == "__main__":
    random.set_seed(MAGIC_SEED)
    main(len(sys.argv), sys.argv)

    exit(0)