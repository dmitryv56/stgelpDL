#!/usr/bin/python3

""" Stand-alone script to load Short Term (green) Electricity Load Processes (Time Series -TS)  Offline Predictor
                          by using Deep Learning .
    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled predictor>\stgelpDL  (Windows 10)
    or
        cd <Unix path to installed predictor>/stgelpDL      (Ubuntu 18.*),

    and put following command to run predictor:
           python start_offlpredictor.py --mode offlp
           --csv_dataset "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020_CommonAnalyze.csv"
           --endogen "Real_demand,Imbalance,WindTurbine_Power,Forecasting,ForecastImbalance,Programmed_demand"
           --timestamp "Date Time"
           --n_step 32
          --num_predicts 4
          --discret 10
          --test 64
          --eval 1024
          --title "ElHiero"
           -vvv



    To read  a version of predictor
             python start_offlpredictor.py --version

    To read a help
             python start_offlpredictor.py --help
    or
             python start_offlpredictor.py -h

"""

import sys

from tensorflow import random
from predictor.cfg import MAGIC_SEED
from offlinepred.stgelopDL import main

if __name__ == "__main__":

    random.set_seed(MAGIC_SEED)
    main(len(sys.argv), sys.argv)

    sys.exit(0)