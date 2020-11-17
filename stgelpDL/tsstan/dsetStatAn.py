#!/usr/bin/python3

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import scipy.stats as st


from hmm.api_acppgeg import  plotStates
from predictor.api import initCP,endCP
from predictor.cfg import PATH_DATASET_REPOSITORY,PATH_REPOSITORY, CONTROL_PATH,TRAIN_PATH,PREDICT_PATH,LOG_FILE_NAME,\
    DISCRET
from predictor.control import ControlPlane
from predictor.utility import msg2log,cSFMT,PlotPrintManager
from tsstan.pltStan import setPlot, plotAll
from tsstan.apiStatAn import getAbruptChanges

__version__='0.0.1'

PROGRAM_TITLE ="Data Analysis for  the Processes of (green) Electricity Grid Predictor"
CSV_PATH ="~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
DT_COL_NAME = "Date Time"
DATA_COL_NAME="Imbalance"


def cmdlineParser(program_title,vers:str='0.0.0'):
    # command-line parser
    sDescriptor = program_title
    sCSVhelp          = "Absolute path to a source dataset (csv-file)."
    sTSnameHelp       = "Time Series (column) name in the dataset."
    sTimeStampHelp    = "Time Stamp (column) name in the dataset."
    sdiscretHelp      = "Time series sampling discretization(minutes)."
    sextraTSHelp      = "Extra Time Series into dataset. The quoted string  contains the substrings separated by comma."
    sabrupttypeHelp   = "'upper','lower','all'. Ignored for time series from extra,one-side and  two-sides lists."
    sabruptlevelHelp  = "Significance level for one side abrupt detection, splits by half for two-sides abrupt detection."
    sonesideupperHelp = "Time Series are checked for one-side upper abrupt detection."
    sonesidelowerHelp = "Time Series are checked for one-side lower abrupt detection."
    stwosidesHelp     = "Time Series are checked for two-sides abrupts detection."
    slogHelp          = "Log folder is a parent for all logs excluding 'execution_time.log' and 'cmdl_args.log'"
    parser = argparse.ArgumentParser(description=sDescriptor)

    parser.add_argument('-c', '--csv_dataset',    dest= 'cl_dset',      action='store', default='', help=sCSVhelp)
    parser.add_argument('-t', '--tsname',         dest= 'cl_tsname',    action='store', default='Imbalance',
                        help=sTSnameHelp)
    parser.add_argument(      '--timestamp',      dest= 'cl_timestamp', action='store', default='Date Time',
                        help=sTimeStampHelp)
    parser.add_argument('-x', '--extra_ts',       dest= 'cl_extraTS',    action='store', default='',
                        help=sextraTSHelp)
    parser.add_argument('-u', '--one_side_upper', dest= 'cl_onesideU',   action='store', default='',
                        help=sonesideupperHelp)
    parser.add_argument('-l', '--one_side_lower', dest= 'cl_onesideL',   action='store', default='',
                        help=sonesidelowerHelp)
    parser.add_argument('-w', '--two_sides',      dest= 'cl_twosides',   action='store', default='',
                        help=stwosidesHelp)
    parser.add_argument('-b', '--abrupt_type',    dest= 'cl_abrupttype', action='store', default='upper',
                        help=sabrupttypeHelp, choices=['upper','lower','all'])
    parser.add_argument('-r', '--abrupt_level',   dest='cl_abruptlevel', action='store', default=0.9,
                        help=sabruptlevelHelp)
    parser.add_argument('-d', '--discret',        dest='cl_discret',     action='store', default=10, help=sdiscretHelp)
    parser.add_argument('-g', '--log',            dest='cl_log',         action='store', default='Logs_DataAnalysis',
                        help=slogHelp)
    parser.add_argument('-v', '--verbose',        dest='cl_verbose',     action='count', default=0)
    parser.add_argument(      '--version',                               action='version',
                        version='%(prog)s {}'.format(vers))

    args = parser.parse_args()
    return parser, args

def cmdLineLog(parser:argparse.ArgumentParser):
    # command-line parser

    args = parser.parse_args()
    arglist = sys.argv[0:]
    argstr = ""
    for arg in arglist:
        argstr += " " + arg
    message = f"""  Command-line arguments

{argstr}

Dataset path                   : {args.cl_dset}
Time Series in dataset         : {args.cl_tsname}
Discretization (min)           : {args.cl_discret}
Timestamp in dataset           : {args.cl_timestamp}
Extra Time Series in dataset   : {args.cl_extraTS}
Time Series are checked for 
one-side upper abrupt detection: {args.cl_onesideU}
Time Series are checked for 
one-side lower abrupt detection: {args.cl_onesideL}            
Time Series are checked for 
two-sides abrupts detection    : {args.cl_twosides}
Significance level for one side
abrupt detection,              : {float(args.cl_abruptlevel) * 100} %  {1.0 - float(args.cl_abruptlevel) } of range
splits by half for two-sides   : {(1.0 - float(args.cl_abruptlevel) )/2.0 } of range  
abrupt detection.              : {(1.0 - float(args.cl_abruptlevel) )/2.0 } of range
Type of abrupt detection       : {args.cl_abrupttype} 
Log folder                     : {args.cl_log}

     """

    with open("cmdl_args.log", "w+") as fcl:
        msg2log(None, message, fcl)
    fcl.close()
    return

def readDataset(csv_file: str, dt_col_name:str, cp: object)->(pd.DataFrame,list):
    ds =pd.read_csv(csv_file)
    ltimeseries=ds.columns.tolist()
    ltimeseries.remove(dt_col_name)
    return ds,ltimeseries

def firstDiffTS(ds: pd.DataFrame,dt_col_name:str,listTS:list,cp:ControlPlane)->(pd.DataFrame, list,list):
    listTSext=copy.copy(listTS)
    for item in listTS:
        difflist=[ round(ds[item].values[i+1]-ds[item].values[i], 3) for i in range(len(ds)-1)]
        c=round((difflist[-1]+difflist[-2])/2.0, 3)
        difflist.append(c)
        new_name="diff_{}".format(item)
        ds[new_name]=difflist
        listTSext.append(new_name)

    ltimeseries = ds.columns.tolist()
    ltimeseries.remove(dt_col_name)
    return ds, ltimeseries,listTSext


def getScalarProperties(ds:pd.DataFrame,item:str,f:object=None)->(float,float,float,float,float,float):
    minValue    = ds[item].min(skipna=True)
    maxValue    = ds[item].max(skipna=True)
    meanValue   = ds[item].mean(skipna=True)
    medianValue = ds[item].median(skipna=True)
    modeValue   = ds[item].squeeze().mode().tolist()[0]
    stdValue    = ds[item].std(skipna=True)

    return minValue,  maxValue , meanValue, medianValue,modeValue, stdValue

def getVectorProperties(ds:pd.DataFrame,item:str,cp:ControlPlane)->(np.array,np.array,np.array,np.array,np.array):
    hist, bins = np.histogram(ds[item].values)
    f, Pxx = signal.periodogram(ds[item].values, 1.0 / (60.0 * cp.discret))
    Pxx=np.round(Pxx,4)
    autocorr = np.round(np.array([ds[item].squeeze().autocorr(lag=i) for i in range(512)]),4)

    return hist,bins, f,Pxx, autocorr


def getLambda(states, T)->(dict):
    sorted_array, count_array = np.unique(states, return_counts=True)
    lambdaAbrupt={}

    for i in range(len(sorted_array)):
       lambdaAbrupt[sorted_array[i]] = (float(count_array[i])/float(T))

    return lambdaAbrupt

def drive_tsAnalysis(ds,dt_col_name, listTS,  cp, abrupt_type:str='all', abrupt_level: float = 0.9 )->(pd.DataFrame, list):

    listTSest=[]
    LOWER_ABRUPT = 0
    NO_ABRUPT = 1
    UPPER_ABRUPT = 2
    enum = (LOWER_ABRUPT, NO_ABRUPT, UPPER_ABRUPT)



    for item in listTS:
        minValue,  maxValue , meanValue, medianValue,modeValue, stdValue = getScalarProperties(ds, item, cp.fc)
        hist, bins , f, Pxx,autocorr =  getVectorProperties(ds, item, cp)

        state_list, abrupt_dict, abrupt_period_dict,tuple_lower,tuple_no,tuple_upper = \
            getAbruptChanges(ds, item, dt_col_name, minValue, maxValue, enum, abrupt_type=abrupt_type,
                             abrupt_level = abrupt_level,f = cp.fc)
        (ind_lower,lower_longest) =tuple_lower
        (ind_no_abrupt, no_abrupt_longest) = tuple_no
        (ind_upper, upper_longest) = tuple_upper
        ds["states_{}".format(item)] = state_list
        N = len(ds[dt_col_name])
        T = N * cp.discret
        lambdaAbrupt=getLambda(state_list,T)
        #correct logged values
        if abrupt_type=='lower':
            lambdaLower = round(lambdaAbrupt[LOWER_ABRUPT],4)
            tauLower    = round(1./lambdaLower,4) if lambdaLower!=0.0 else '--'
            lowerStart  = ds[dt_col_name].values[ind_lower]
            lower_len   = len(abrupt_dict["LOWER_ABRUPT"])
            lambdaNo    = round(lambdaAbrupt[NO_ABRUPT],4)
            tauNo       = round(1. / lambdaNo, 4) if lambdaNo!=0.0 else "--"
            lambdaUpper ='--'
            tauUpper    ='--'
            upperStart  = '--'
            upper_len   = '--'
        elif abrupt_type=='upper':
            lambdaLower ='--'
            tauLower    ='--'
            lowerStart  ='--'
            lower_len   ='--'
            lambdaNo    = round(lambdaAbrupt[NO_ABRUPT], 4)
            tauNo       = round(1. / lambdaNo, 4) if lambdaNo!=0.0 else "--"
            lambdaUpper = round(lambdaAbrupt[UPPER_ABRUPT], 4)
            tauUpper    = round(1. / lambdaUpper, 4) if lambdaUpper!=0.0 else "--"
            upperStart  = ds[dt_col_name].values[ind_upper]
            upper_len   = len(abrupt_dict["UPPER_ABRUPT"])
        elif abrupt_type=='all':
            lambdaLower = round(lambdaAbrupt[LOWER_ABRUPT], 4)
            tauLower    = round(1. / lambdaLower, 4) if lambdaLower!=0.0 else '--'
            lowerStart  = ds[dt_col_name].values[ind_lower]
            lower_len    = len(abrupt_dict["LOWER_ABRUPT"])
            lambdaNo    = round(lambdaAbrupt[NO_ABRUPT], 4)
            tauNo       = round(1. / lambdaNo, 4) if lambdaNo!=0.0 else "--"
            lambdaUpper = round(lambdaAbrupt[UPPER_ABRUPT], 4)
            tauUpper    = round(1. / lambdaUpper, 4) if lambdaUpper!=0.0 else "--"
            upperStart  = ds[dt_col_name].values[ind_upper]
            upper_len   = len(abrupt_dict["UPPER_ABRUPT"])

        message=f"""
************************************************************************************************************************
                Time Series:  {item}

Started at: {ds[dt_col_name].min()} Ended at:   {ds[dt_col_name].max()}  Discretization :{cp.discret}  minutes 
TS size :   {N}  Time Series duration: {T } minutes   

Min:    {minValue}
Max:    {maxValue}
Mean:   {meanValue}
Median: {medianValue}
Mode:   {modeValue}
s.t.d.: {stdValue}
                
Autocorr: {autocorr[:32]} ...
                Histogram
Bins:        {bins}
Hist.values: {hist}
                Periodogram
Frequence:   {f[:32]}   ...
Pxx:         {Pxx[:32]} ...

***********************************************************************************************************************
                Abrupt detection
      States 0-Lower Process Limit  Abrupt, 1 -no abrupt, 2 -Upper Process Limit  Abrupt
Detected abrupt type is : {abrupt_type}
Abrupt level is { round(abrupt_level,2) if abrupt_type=='all' else round(abrupt_level + (1.0-abrupt_level)/2.0 ,3)}
First states:  {state_list[:32]} ...
Last states: ... {state_list[len(state_list)-32:]}
Number of Lower Process Limit  Abrupts: {lower_len }
Longest Lower Process  Limit Abrupt Sequence: {lower_longest}   at: {lowerStart}
Number of Upper Process Limit  Abrupts: {upper_len }
Longest Upper Process  Limit Abrupt Sequence: {upper_longest}   at  {upperStart}
Lambda (Lower Abrupt Posson Process): {lambdaLower} 1/minutes. Average time for first lower abrupt : {tauLower} minutes
Lambda (No Abrupt Posson Process):    {lambdaNo} 1/minutes. Average time for first :              {tauNo} minutes
Lambda (Upper Abrupt Posson Process): {lambdaUpper} 1/minutes. Average time for first :              {tauUpper} minutes.


"""

        msg2log(None,message,cp.fc)
        msg2log(None,"\n\nAbrupts dictionary\n{}".format(abrupt_dict),cp.fc)
        if abrupt_type!='upper':
            msg2log(None, "\n\nLower Process Limit Abrupts\n{}".format(abrupt_period_dict["LOWER_ABRUPT"]), cp.fc)
        if abrupt_type != 'lower':
            msg2log(None, "\n\nUpper Process Limit Abrupts\n{}".format(abrupt_period_dict["UPPER_ABRUPT"]), cp.fc)

        pngLogs(ds, item, state_list, cp)
        dict_estimations={
            item:
                {
                    "Min": minValue,
                    "Max": maxValue,
                    "Mean":meanValue,
                    "Median": medianValue,
                    "Mode": modeValue,
                    "std":  stdValue,
                    "autocorr": autocorr.tolist(),
                    "hist": hist.tolist(),
                    "bins": bins.tolist(),
                    "f_array":f.tolist(),
                    "Pxx":Pxx.tolist()
                }
        }
        listTSest.append(dict_estimations)
    return ds, listTSest

def pngLogs(ds, data_col_name, state_sequence, cp ):

    folder_log = Path(cp.folder_control_log/data_col_name)
    Path(folder_log).mkdir(parents=True, exist_ok=True)
    step_index = 512
    for start_index in range(0, len(state_sequence), step_index):
        title = "{}_from_{}_till_{}".format(data_col_name,start_index,start_index+ step_index)
        plotStates(ds, data_col_name, state_sequence, title, str(folder_log), start_index=start_index,
                   end_index=start_index + step_index, f=cp.fc)
    name=""
    plotAll(name, ds[data_col_name], data_col_name, folder_log, f=cp.fc)
    printHist(ds, data_col_name, folder_log, f=cp.fc)
    printHistStates(state_sequence, data_col_name, folder_log, f=cp.fc)
    plot_ts(ds, "Date Time", data_col_name, folder_log, f=cp.fc)
    return

def printHist( ds,data_col_name,path_to_folder, f:object=None):

    name=data_col_name.replace(' ','_')
    title = "Histogram ({})".format(name)
    png_file_ = "Histogram_{}.png".format(name)
    msg = ""
    try:
        png_file=Path(path_to_folder / png_file_)
        plt.hist(ds[data_col_name],density=True,bins=int(len(ds[data_col_name])/10), label ="Data",color='red')
        mn,mx =plt.xlim()
        kde_xs =np.linspace(mn,mx, 301)
        kde = st.gaussian_kde(ds[data_col_name])
        plt.plot(kde_xs,kde.pdf(kde_xs), label='PDF', color='green')
        plt.legend(loc='upper left')
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title(title)

        plt.savefig(png_file)

    except np.linalg.LinAlgError:
        msg="Oops! raise LinAlgError(singular matrix)"
    except:
        msg ="Unexpected error: {}".format( sys.exc_info()[0])
    finally:

        msg2log(None,msg,f)
        plt.close("all")
    return

def printHistStates( states,data_col_name,path_to_folder, f:object=None):

    name=data_col_name.replace(' ','_')
    title = "State Sequence Histogram ({})".format(name)
    png_file_ = "State_Sequence_Histogram_{}.png".format(name)
    msg = ""
    try:
        png_file=Path(path_to_folder / png_file_)
        plt.hist(states,density=True,bins=int(len(states)/10), label ="States")
        mn,mx =plt.xlim()
        kde_xs =np.linspace(mn,mx, 301)
        kde = st.gaussian_kde(states)
        plt.plot(kde_xs,kde.pdf(kde_xs), label='PDF')
        plt.legend(loc='upper left')
        plt.ylabel('Probability')
        plt.xlabel('States')
        plt.title(title)

        plt.savefig(png_file)

    except np.linalg.LinAlgError:
        msg="Oops! raise LinAlgError(singular matrix)"
    except:
        msg ="Unexpected error: {}".format( sys.exc_info()[0])
    finally:

        msg2log(None,msg,f)
        plt.close("all")
    return

def plot_ts(df, dt_col_name, data_col_name, logfolder, f:object=None):

    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    fig, ax = plt.subplots()
    num = 0

    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    year_fmt = mdates.DateFormatter('%Y')
    # for column in df1.drop(['Date Time'], axis=1):
    #     num += 1
    #     ax.plot(df1['Date Time'], df1[column], marker='', color=palette(num), label=column)

    ax.plot(df[dt_col_name], df[data_col_name], marker='', color=palette(num), label=data_col_name)
    plt.legend(loc=2, ncol=2)
    ax.set_title('{}'.format(data_col_name))
    fig.autofmt_xdate()

    ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(year_fmt)
    ax.xaxis.set_minor_locator(months)
    datemin = df[dt_col_name].min()
    datemax = df[dt_col_name].max()
    ax.set_xlim(datemin, datemax)
    fig.autofmt_xdate()

    ax.set_xlabel("Time")
    ax.set_ylabel(data_col_name)

    sfile = "{}.png".format(data_col_name.replace(' ', '_'))
    # sFolder = PlotPrintManager.get_ControlLoggingFolder()
    # filePng = Path(sFolder) / (sfile)

    filePng = Path(logfolder / sfile )
    plt.savefig(filePng)
    plt.close("all")


    return


def main(argc,argv):

    parser, args = cmdlineParser(PROGRAM_TITLE, vers=__version__)
    cmdLineLog(parser)

    setPlot()
    program_title = PROGRAM_TITLE
    csv_file      = args.cl_dset if args.cl_dset is not None and args.cl_dset else CSV_PATH
    data_col_name = args.cl_tsname if args.cl_tsname is not None and args.cl_tsname else DATA_COL_NAME
    dt_col_name   = args.cl_timestamp if args.cl_timestamp is not None and args.cl_timestamp else DT_COL_NAME
    discret       = int(args.cl_discret)
    sExtraTS      = args.cl_extraTS if args.cl_extraTS is not None and args.cl_extraTS else ""
    sOneSideU     = args.cl_onesideU if args.cl_onesideU is not None and args.cl_onesideU else ""
    sOneSideL     = args.cl_onesideL if args.cl_onesideL is not None and args.cl_onesideL else ""
    sTwoSides     = args.cl_twosides if args.cl_twosides is not None and args.cl_twosides else ""
    abrupt_level  = float(args.cl_abruptlevel)
    abrupt_type   = args.cl_abrupttype
    sLogFolder    = args.cl_log

    actual_mode="control_plan"  # for compatibility with initCP()

    tuple_cfg=(PATH_DATASET_REPOSITORY,PATH_REPOSITORY, CONTROL_PATH,TRAIN_PATH,PREDICT_PATH,LOG_FILE_NAME)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    cp = initCP(program_title, tuple_cfg, csv_file=csv_file, actual_mode=actual_mode, dt_col_name=dt_col_name,
                data_col_name=data_col_name, logFolderName=sLogFolder,
                dir_path = os.path.dirname(os.path.realpath(__file__)),discret =discret)

    lExtraTS  = list(sExtraTS.split(','))
    lOneSideU = list(sOneSideU.split(','))
    lOneSideL = list(sOneSideU.split(','))
    lTwoSides = list(sTwoSides.split(','))

    ds,listTS = readDataset(csv_file,dt_col_name, cp)
    if not lOneSideU and not lOneSideL and not lTwoSides:
        # del extra TS from lstTS
        for item in lExtraTS:
            listTS=listTS.remove(item)
        ds, listTS,listTSext = firstDiffTS(ds, dt_col_name, listTS, cp)
        ds, listTSestimations = drive_tsAnalysis(ds, dt_col_name, listTSext, cp, abrupt_type=abrupt_type, abrupt_level=abrupt_level)
    else:
        if lOneSideU:
            ds, listAll, lOneSideU = firstDiffTS(ds, dt_col_name, lOneSideU, cp)
            ds, listTSUestimations = drive_tsAnalysis(ds, dt_col_name, lOneSideU, cp, abrupt_type='upper',abrupt_level=abrupt_level)
        if lOneSideL:
            ds, listAll,lOneSideL = firstDiffTS(ds, dt_col_name, lOneSideL, cp)
            ds, listTSLestimations = drive_tsAnalysis(ds, dt_col_name, lOneSideU, cp, abrupt_type='lower', abrupt_level=abrupt_level)
        if lTwoSides:
            ds, listAll, lTwoSides = firstDiffTS(ds, dt_col_name, lTwoSides, cp)
            ds, listTSULestimations = drive_tsAnalysis(ds, dt_col_name, lTwoSides, cp, abrupt_type='all',abrupt_level=abrupt_level)

    ds,listTS,listTSext = firstDiffTS(ds, dt_col_name, listTS, cp)

    ds, listTSestimations = drive_tsAnalysis(ds, dt_col_name, listTS,  cp,abrupt_type='all',abrupt_level=abrupt_level)
    ds.to_csv("backup_extnendedElHiero.csv")
    msg ="List of Time Series properties\n\n{}".format(listTSestimations)
    # msg2log(None,msg,cp.ft)


    endCP(cp, program_title =program_title)

    return 0
if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
