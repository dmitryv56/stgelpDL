#!/usr/bin/python3

""" Stand-alone script to load simple Linear Regression for Green Electricity  Load processes .
    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled predictor>\stgelpDL  (Windows 10)
    or
        cd <Unix path to installed predictor>/stgelpDL      (Ubuntu 18.*),

    and put following command to estimate coeffcients and quality of the CO2=A0 +A1*Diesel_Power + A2*Pump_Power
    regression approximation
           python start_linreg.py
           --csv_dataset "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
           --endogen "CO2" --exogen "Diesel_Power,Pump_Power" --timestamp "Date Time" --width 256



    To read  a version of prohram
             python start_linreg.py --version

    To read a help
             python start_linreg.py --help
    or
             python start_linreg.py -h

"""

import sys
from simpleRegr.main import main

if __name__ == "__main__":

    main(len(sys.argv), sys.argv)

    exit(0)