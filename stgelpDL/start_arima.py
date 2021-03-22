#!/usr/bin/python3

""" Stand-alone script to load ARIMA (green) Electricity Load Processes (Time Series -TS)  Offline Predictor.

    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled predictor>\stgelpDL  (Windows 10)
    or
        cd <Unix path to installed predictor>/stgelpDL      (Ubuntu 18.*),

    and put following command to run predictor:
           python start_arima.py
           --csv_dataset "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/aggSolarPlantPowerGen_21012020.csv"
           --endogen "max_power" --exogenuos "min_power,mean_power"  --timestamp "Date Time"
           --chunk 32 --out_sample 4 --in_sample 32 --discret 10
           --p_max 4 --d_max 2 --q_max 4 --P_max 4 --D_max 2 --Q_max 4 --season 6
           --title "SolarPlant"


    or in the short form
          python start_arima.py
           -c "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/aggSolarPlantPowerGen_21012020.csv"
           -o "max_power" -x "min_power,mean_power"  -t "Date Time"
           -u 32 -f 4 -s 32 -e 10 -p 4 -d 2 -q 4 -P 4 -D 2 -Q 4 -S 6 -i "SolarPlant"



    To read  a version of predictor
             python start_arima.py --version

    To read a help
             python start_arima.py --help
    or
             python start_arima.py -h

"""

import sys
from predictor.cfg import MAGIC_SEED
from arima.stgelpARIMA import main

if __name__ == "__main__":

    main(len(sys.argv), sys.argv)

    exit(0)
