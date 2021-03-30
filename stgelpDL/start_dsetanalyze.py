#!/usr/bin/python3
""" Stand-alone script to run Data Analysis for Green Energy Load Deep Learning predictor.
    In order to run this stand-alone script open  command-line, select folder
        cd <Windows path to unstalled predictor>\stgelpDL  (Windows 10)
    or
        cd <Unix path to installed predictor>/stgelpDL      (Ubuntu 18.*),

    and put following command:
           python start_dsetanalyze.py -v {arguments}
    To read  a version of predictor
            python start_cluster.py --version
    To read a help
            python start_cluster.py --help
    or
            python start_cluster.py -h
    Below example how to analyze El-Yierro dataset.

    python start_dsetanalyze.py --csv_dataset="~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
    --tsname "Imbalance" --timestamp "Date Time" --extra_ts "CO2,Hydrawlic"
    --one_side_upper "Diesel_Power,HydrawlicTurbine_Power,Pump_Power"
    --two_sides "Imbalance,Real_demand,WindTurbine_Power" --discret 10
    --abrupt_level 0.9 --log "Logs_Analysis_ElHierro"

    where csv_dataset - absolute or relative path to El-Hierro dataset
          tsname  - the name of target time series into dataset. This name is used  to form the titles in logs.
          timestamp - the column with time stamps in the dataset.
          extra_ts - extra time series in the dataset which should not be analysed.
          one_side_upper - the time series ( columns in the dataset) for which the upper sharp abrupt should be
                     detected.
          one_side_lower - the time series  for which the lower  sharp abrupt should be detected.
          two_sides  - the time series for which the upper and lower sharp abrupt should be detected.
          discret - the time series discretization in minutes.
          abrupt_type - one of {'upper','lower','all'} values, used if no one_side_upper, one_side_lower set.
          abrupt_level - the significance level for one side abrupt detection, splits by half for two-sides abrupt
                      detection.
          log - Log folder is a parent for all logs excluding 'execution_time.log' and 'cmdl_args.log'

"""

import sys

from tsstan.dsetStatAn import  main

if __name__ == "__main__":

    main(len(sys.argv), sys.argv)

    exit(0)