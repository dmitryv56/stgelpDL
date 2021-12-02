#!/usr/bin/python3
""" Stand-alone script to load Short-Term Green Energy Load Deep Learning predictor.
    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled predictor>\stgelpDL  (Windows 10)
    or
        cd <Unix path to installed predictor>/stgelpDL      (Ubuntu 18.*),

    and put following command:
           python start_predictor.py -v
    To read  a version of predictor
            python start_predictor.py --version
    To read a help
            python start_predictor.py --help
    or
            python start_predictor.py -h

"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import random

from predictor.cfg import MAGIC_SEED
from predictor.stgelpDL import main

if __name__ == "__main__":
    random.set_seed(MAGIC_SEED)
    main(len(sys.argv), sys.argv)

    exit(0)
