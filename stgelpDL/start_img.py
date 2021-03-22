#!/usr/bin/python3
""" Stand-alone script to load Image Classificator (for 'Past without Things' blog).
    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled package>\PastWoThings  (Windows 10)
    or
        cd <Unix path to installed package>/PastWoThings      (Ubuntu 18.*),

    and put following command:
           python3 start_img.py --src_folder /home/dmitryv/PastWithoutToughts --wv_type db38 --num_clust 6
    To read  a version of predictor
            python3 start_img.py --version
    To read a help
            python3 start_img.py --help
    or
            python3 start_img.py -h

"""

import sys
from PastWoThings.img import main

if __name__ == "__main__":

    main(len(sys.argv), sys.argv)