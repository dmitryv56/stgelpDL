#!/usr/bin/python3

""" Stand-alone script to transform  time series to matrix . The time series (feature, column in the source dataset)
should beginning at midnt., 00:00:00. Its length should be multiply of segement(time period like as day, week).

    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled predictor>\stgelpDL  (Windows 10)
    or
        cd <Unix path to installed predictor>/stgelpDL      (Ubuntu 18.*),

    and put following command to create matrix view for time series:
           ./start_ts2matrix.py --csv_dataset /home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv
           --ts "Real_demand" --timestamp "Date Time" --direction X --discret 10 --period 144
           --scale absmax --outrowind "rowNames" --title "ElHierro"

    or
            ./start_ts2matrix.py -c /home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv
           -t "Real_demand" --timestamp "Date Time" -r X -d 10 -p 144 -s absmax -o "rowNames" -i "ElHierro"



    To read  a version of predictor
             ./start_ts2matrix.py --version

    To read a help
             ./start_ts2matrix.py --help
    or
             ./start_ts2matrix.py -h

"""

import sys
from ts2matrix.ts2mat import main

if __name__ == "__main__":

    nret = main(len(sys.argv), sys.argv)
    sys.exit(nret)