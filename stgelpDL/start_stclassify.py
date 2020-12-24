#!/usr/bin/python3

""" Stand-alone script to load State Classification for Green Electricity  Load processes by using Deep Learning .
    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled predictor>\stgelpDL  (Windows 10)
    or
        cd <Unix path to installed predictor>/stgelpDL      (Ubuntu 18.*),

    and put following command:
           python start_stclassify.py --csv_dataset "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
           --endogen 'Imbalance' --exogen 'Diesel_Power,WindTurbine_Power,HydrawlicTurbine_Power,Pump_Power,CO2'
           --timestamp 'Date Time' --n_step 32 --num_clusters 10 --discret 10 --num_predicts 4 --verbose
           
    To read  a version of predictor
             python start_stclassify.py --version
    or
            python start_stclassify.py -v
    To read a help
             python start_stclassify.py --help
    or
             python start_stclassify.py -h

"""

import sys
from tensorflow import random
from predictor.cfg import MAGIC_SEED
from stcgelDL.main import main

if __name__ == "__main__":
    random.set_seed(MAGIC_SEED)
    main(len(sys.argv), sys.argv)

    exit(0)