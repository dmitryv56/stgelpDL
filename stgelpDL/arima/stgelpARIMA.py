#!/usr/bin/python3

import argparse
import os
import sys
from datetime import datetime,timedelta
from pathlib import Path

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX,SARIMAXResults,SARIMAXParams
from statsmodels.tsa.statespace.mlemodel import MLEResults
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

from predictor.utility import msg2log
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from arima.api import readDataset,checkStationarity,arima_order,arima_run


__version__="0.0.1"

url = 'https://raw.githubusercontent.com/selva86/datasets/master/a10.csv'
url1 = 'https://github.com/dmitryv56/stgelpDL/tree/dataset/dataLaLaguna/SolarPowerGen_21012020.csv'
url2="https://cloud.mail.ru/public/P8A3/ScG9CTy9Y"

def parser()->(argparse.ArgumentParser, str):

 # command-line parser
    sDescriptor         = "Short Term (green) Electricity Load Processes (Time Series -TS)  Offline Predictor " +  \
                          "by using ARIMA (p,d,q) x (P,D,Q,S) model."

    sCSVhelp            = "Absolute path to a source dataset (csv-file)."
    sEndogenTSnameHelp  = "The TS, the name of column into dataset."
    sExogenousTSnameHelp="The exogenous features list. A string contains the comma-separated column names in the " + \
                         "dataset."
    sTimeStampHelp      = "Time Stamp (column) name in the dataset."
    schunkHelp          = "The chunk size for long TS. ARIMA parameters are estimated over this tail chunk. "+ \
                          "The statistics like as mean,std,autocorrelation are estimated over all TS."
    sdiscretHelp        = "TS sampling discretization(minutes)"
    snum_predictsHelp   = "Predict period for forecasting or out-of-sample predicting."
    spmaxHelp           = "Max  order (number of time lags) of the autoregressive model (AR)."
    sdmaxHelp           = "Max degree of differencing (the number of times the data have had past values subtracted)."
    sqmaxHelp           = "Max  order  of the moving-average model (MA)."
    sSHelp              = "Number of time lags in each season."
    sPmaxHelp           = "Max  order (number of time lags) of the season autoregressive model (AR)."
    sDmaxHelp           = "Max degree of season differencing."
    sQmaxHelp           = "Max  order  of the season moving-average model (MA)."
    sinsampleHelp       = "Offset from end of TS for in-sample predicting."
    stitleHelp          = "Title, one word using as log folder name."


    parser = argparse.ArgumentParser(description=sDescriptor)

    parser.add_argument('-c', '--csv_dataset',dest='cl_dset',    action='store',            help=sCSVhelp)
    parser.add_argument('-o', '--endogen',   dest='cl_endots',   action='store', default='',
                        help=sEndogenTSnameHelp)
    parser.add_argument('-x', '--exogenuos', dest='cl_exogen',    action='store', default='',
                    help=sExogenousTSnameHelp)
    parser.add_argument('-t', '--timestamp', dest='cl_timestamp', action='store', default='Date Time',
                        help=sTimeStampHelp)
    parser.add_argument('-u', '--chunk',     dest='cl_chunk',     action='store', default=32,  help=schunkHelp)
    parser.add_argument('-e', '--discret',   dest='cl_discret',   action='store', default=10,  help=sdiscretHelp)
    parser.add_argument('-f', '--out_sample',dest='cl_out_sample',action='store', default=4,
                        help=snum_predictsHelp)
    parser.add_argument('-s', '--in_sample', dest='cl_in_sample', action='store', default=32,  help=sinsampleHelp)
    parser.add_argument('-p', '--p_max',     dest='cl_p_max',     action='store', default=4,   help=spmaxHelp)
    parser.add_argument('-d', '--d_max',     dest='cl_d_max',     action='store', default=2,   help=sdmaxHelp)
    parser.add_argument('-q', '--q_max',     dest='cl_q_max',     action='store', default=2,   help=sqmaxHelp)
    parser.add_argument('-P', '--P_max',     dest='cl_P_max',     action='store', default=2,   help=sPmaxHelp)
    parser.add_argument('-D', '--D_max',     dest='cl_D_max',     action='store', default=0,   help=sDmaxHelp)
    parser.add_argument('-Q', '--Q_max',     dest='cl_Q_max',     action='store', default=1,   help=sQmaxHelp)
    parser.add_argument('-S', '--season',    dest='cl_S',         action='store', default=0,   help=sSHelp)
    parser.add_argument('-i', '--title',     dest='cl_title',     action='store', default='ElHiero', help=stitleHelp)
    parser.add_argument('--verbose', '-v',   dest='cl_verbose',   action='count', default=0)
    parser.add_argument('--version',         action='version', version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()
    # command-line parser
    arglist=sys.argv[0:]
    argstr=""
    for arg in arglist:
        argstr+=" " + arg
    message0=f"""  Command-line arguments

{argstr}
Title:                             : {args.cl_title}
Dataset path                       : {args.cl_dset}
Time Series in dataset             : {args.cl_endots}
Exogenous features in dataset      : {args.cl_exogen}
Timestamp in dataset               : {args.cl_timestamp}
Chunk for ARIMA estimation size    : {args.cl_chunk}  
Discretization (min)               : {args.cl_discret}
Out sample forecast periods        : {args.cl_out_sample}
In sample predict periods          : {args.cl_in_sample}
ARIMA                              : ({args.cl_p_max},{args.cl_d_max},{args.cl_q_max})
ARIMAS                             : ({args.cl_P_max},{args.cl_D_max},{args.cl_Q_max}) {args.cl_S}

"""

    return args, message0

def paramLog(param:tuple=None)->str:
    (filecsv , title, data_col_name ,     dt_col_name, exogen, n, dt_start,dt_finish, chunk_size, \
    discret , chunk_start, forecast_period , in_sample_size, in_sample_start, max_order ,max_seasonal_order , \
    log_folder ) = param

    TS_lasts=n*discret
    days=int(TS_lasts/1440)
    hours= int((TS_lasts % 1440)/60)
    minutes = (TS_lasts % 1440)%60
    message0 =f""" 
Dataset                   : {filecsv}
Title                     : {title}
TS                        : {data_col_name}
Timestamps                : {dt_col_name}
Exogenous                 : {exogen}
TS size                   : {n}
TS starts at              : {dt_start}  ends at {dt_finish}
TS discret                : {discret} min
TS duration               : D{days}:H{hours}:M{minutes}
Chunk size                : {chunk_size}
Chunk starts at           : {chunk_start}
Forecast periods          : {forecast_period}
In sample predict         : {in_sample_size}
In sample predict starts  : {in_sample_start}
Max ARIMA x ARIMAS orders : {max_order} x {max_seasonal_order}
Log folder                : {log_folder}

"""
    return message0

def main(argc,argv):
    # filecsv="/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/SolarPowerGen_21012020.csv"
    # filecsv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
    # filecsv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/aggSolarPlantPowerGen_21012020.csv"

    args,message0 = parser()
    filecsv            = args.cl_dset
    title              = args.cl_title
    data_col_name      = args.cl_endots
    dt_col_name        = args.cl_timestamp
    exogen             = args.cl_exogen
    chunk_size         = int(args.cl_chunk)
    discret            = int(args.cl_discret)
    forecast_period    = int(args.cl_out_sample)
    in_sample_size     = int(args.cl_in_sample)
    max_order          = (int(args.cl_p_max), int(args.cl_d_max), int(args.cl_q_max))
    max_seasonal_order = (int(args.cl_P_max), int(args.cl_D_max), int(args.cl_Q_max), int(args.cl_S))

    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(title, date_time)
    listLogSet(str(folder_for_logging))
    msg2log(None,message0,D_LOGS['clargs'])
    msg2log(None, message1, D_LOGS['timeexec'])

    df = readDataset(csv_file=filecsv, endogen=[data_col_name], title=title, f = D_LOGS['control'])
    n=len(df[data_col_name])
    chunk_offset=n-chunk_size
    in_sample_start = n-in_sample_size

    param = ( filecsv, title, data_col_name, dt_col_name, exogen, n, df[dt_col_name][0], df[dt_col_name][len(df)-1], \
              chunk_size, discret, df[dt_col_name][chunk_offset], forecast_period, in_sample_size, \
              df[dt_col_name][in_sample_start], max_order, max_seasonal_order,  str(folder_for_logging))
    msg2log(None, paramLog(param),D_LOGS['main'])
    log2All()

    checkStationarity(df, data_col_name=data_col_name, title=title, f = D_LOGS['train'])
    log2All()

    (p,d,q),(pS,dS,qS,S) = arima_order(df, data_col_name=data_col_name, training_size= chunk_size, title=title,
                                       max_order= max_order, max_seasonal_order=max_seasonal_order,f=D_LOGS['train'])
    log2All()
    arima_run(df = df, data_col_name= data_col_name, dt_col_name = dt_col_name, chunk_offset = chunk_offset,
              chunk_size= chunk_size, in_sample_start=in_sample_start, in_sample_size= in_sample_size,
              forecast_period=forecast_period, title= title,  order=(p,d,q), seasonal_order = (pS,dS,qS,S),
              f=D_LOGS['predict'])


    message1 = "Time execution logging finished at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    msg2log(None, message1, D_LOGS['timeexec'])
    closeLogs()
    return


pass

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)