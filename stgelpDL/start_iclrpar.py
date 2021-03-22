#!/usr/bin/python3

""" Stand-alone script to load Information Criterions for linear regression parameters .
    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled predictor>\stgelpDL  (Windows 10)
    or
        cd <Unix path to installed predictor>/stgelpDL      (Ubuntu 18.*),

    and put following command to estimate coeffcients and statistical hypothesises testing by usin Kulback-Liebler
    Information Criterion

           python start_iclrpar.py
           --mode kcc --repository "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna" --datasets "low_co2, mid_co2, high_co2"
           --endogen "CO2" --exogen "Diesel_Power,Pump_Power" --exclude "Pump_Power --timestamp "Date Time" --width 256
           --alfa 0.05


    To read  a version of prohram
             python start_iclrpar.py --version

    To read a help
             python start_iclrpar.py --help
    or
             python start_iclrpar.py -h

"""

import sys
from simpleRegr.iclrpar import main

if __name__ == "__main__":

    main(len(sys.argv), sys.argv)

    exit(0)