#!/usr/bin/python3

import numpy as np
from time import time, perf_counter
import dateutil.parser
from datetime import datetime
from datetime import timedelta
from pathlib import Path
"""
print to log
"""
# constants
cSFMT = "%Y-%m-%d %H:%M:%S"
DEBUG_PRINT_ = None


"""


tsBoundaries2log prints and logs time series boundaries characterisics (min,max,len)
"""


def tsBoundaries2log(title, df, dt_dset, rcpower_dset, f=None):
    """

    :param title: - title for print
    :param df:    - pandas DataFrame object
    :param dt_dset: - Data/Time column name
    :param rcpower_dset: - time series column name
    :param f:
    :return:
    """
    # For example,title ='Number of rows and columns after removing missing values'
    print('\n{}: {}'.format(title, df.shape))
    print('The time series length: {}\n'.format(len(df[dt_dset])))
    print('The time series starts from: {}\n'.format(df[dt_dset].min()))
    print('The time series ends on: {}\n\n'.format(df[dt_dset].max()))
    print('The minimal value of the time series: {}\n\n'.format(df[rcpower_dset].min()))
    print('The maximum value of the time series: {}\n\n'.format(df[rcpower_dset].max()))
    if f is not None:
        f.write('\n{}: {}\n'.format(title, df.shape))
        f.write('The time series length: {}\n'.format(len(df[dt_dset])))
        f.write('The time series starts from: {}\n'.format(df[dt_dset].min()))
        f.write('The time series ends on: {}\n'.format(df[dt_dset].max()))
        f.write('The minimal value of the time series: {}\n'.format(df[rcpower_dset].min()))
        f.write('The maximum value of the time series: {}\n\n'.format(df[rcpower_dset].max()))

    return


def tsSubset2log(dt_dset, rcpower_dset, df_train, df_val=None, df_test=None, f=None):
    pass
    print('Train dates: {} to {}'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
    if f is not None:
        f.write("\nTrain dataset\n")
        f.write('Train dates: {} to {}\n\n'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))

        if PlotPrintManager.isNeedPrintDataset():
            for i in range(len(df_train)):
                f.write('{} {}\n'.format(df_train[dt_dset][i], df_train[rcpower_dset][i]))

    if df_val is not None:

        print('Validation dates: {} to {}'.format(df_val[dt_dset].min(), df_val[dt_dset].max()))
        if f is not None:
            f.write("\nValidation dataset\n")
            f.write(
                'Validation  dates: {} to {}\n\n'.format(df_val[dt_dset].min(), df_val[dt_dset].max()))
            if PlotPrintManager.isNeedPrintDataset():
                for i in range(len(df_train), len(df_train) + len(df_val)):
                    f.write('{} {}\n'.format(df_val[dt_dset][i], df_val[rcpower_dset][i]))

    if df_test is not None:

        print('Test dates: {} to {}'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
        f.write("\nTest dataset\n")
        f.write('Test  dates: {} to {}\n\n'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
        start = len(df_train) if df_val is None else len(df_train) + len(df_val)
        stop = len(df_train) + len(df_test) if df_val is None else len(df_train) + len(df_val) + len(df_test)
        for i in range(start, stop):
            f.write('{} {}\n'.format(df_test[dt_dset][i], df_test[rcpower_dset][i]))
    return


def chunkarray2log(title, nparray, width=8, f=None):
    # scaled train data

    if not PlotPrintManager.isNeedPrintDataset():
        return
    if f is not None:

        f.write("\n{}\n".format(title))

        for i in range(len(nparray)):
            if i % width == 0:
                f.write('\n{} : {}'.format(i, nparray[i]))
            else:
                f.write(' {} '.format(nparray[i]))
    return


"""
Supervised learning data to log
"""


def svld2log(X, y, print_weight, f=None):
    """

    :param X:
    :param y:
    :param print_weight:
    :param f:
    :return:
    """
    if not PlotPrintManager.isNeedPrintDataset():
        return

    if (f is None):
        return

    for i in range(X.shape[0]):
        k = 0
        line = 0
        f.write("\nRow {}: ".format(i))
        for j in range(X.shape[1]):
            f.write(" {}".format(X[i][j]))
            k = k + 1
            k = k % print_weight
            if k == 0 or k == X.shape[1]:

                if line == 0:
                    f.write(" |  {} \n      ".format(y[i]))
                    line = 1
                else:
                    f.write("\n      ")
    return


def dataset_properties2log(csv_path, dt_dset, rcpower_dset, discret, test_cut_off, val_cut_off, n_steps, n_features, \
                           n_epochs, f=None):
    pass
    if f is not None:
        f.write(
            "====================================================================================================")
        f.write("\nDataset Properties\ncsv_path: {}\ndt_dset: {}\nrcpower_dset: {}\ndiscret: {}\n".format(csv_path,
                                                                                                          dt_dset,
                                                                                                          rcpower_dset,
                                                                                                          discret))
        f.write(
            "\n\nDataset Cut off Properties\ncut of for test sequence: {} minutes\ncut off for validation sequence: {} minutes\n".format(
                test_cut_off, val_cut_off))

        f.write("\n\nTraining Properties\n time steps: {},\nfeatures: {}\n,epochs: {}\n".format(n_steps, n_features,
                                                                                                n_epochs))
        f.write(
            "====================================================================================================\n\n")
    return


def msg2log(funcname, msg, f=None):
    print("\n{}: {}".format(funcname, msg))
    if f is not None:
        f.write("\n{}: {}\n".format(funcname, msg))

def vector_logging(title, seq, print_weigth, f=None):
    if f is None:
        return
    f.write("{}\n".format(title))
    k=0
    line = 0
    f.write("{}: ".format(line))
    for i in range(len(seq)):
        f.write(" {}".format(seq[i]))
        k = k + 1
        k = k % print_weigth
        if k == 0:
            line=line+1
            f.write("\n{}: ".format(line))

    return

def psd_logging(title,freq,psd):

    sfile = "{}.log".format(title.replace(' ', '_'))
    sFolder = PlotPrintManager.get_ControlLoggingFolder()
    filePrint = Path(sFolder) / (sfile)
    stemplate = "{:>5d}  {:>7.7f} {:>15.2f}\n"
    with open(filePrint, 'w+') as fpsd:
        fpsd.write('\n Index  Frequency(Hz) Psd \n')
        for i in range (len(psd)):
            fpsd.write(stemplate.format(i,freq[i],psd[i]))
        fpsd.write('\n')
    return


def incDateStr(inDateTime :str, days:int=0,seconds:int=0,minutes:int=0,hours:int=0,weeks:int=0)->str:
    tDateTime = dateutil.parser.parse(inDateTime)
    return (tDateTime+timedelta(days=days,seconds=seconds,minutes=minutes,hours=hours,weeks=weeks)).strftime(cSFMT)

def decDateStr(inDateTime :str, days:int=0,seconds:int=0,minutes:int=0,hours:int=0,weeks:int=0)->str:
    tDateTime = dateutil.parser.parse(inDateTime)
    return (tDateTime-timedelta(days=days,seconds=seconds,minutes=minutes,hours=hours,weeks=weeks)).strftime(cSFMT)

# ##############################################charting################################################################

# preallocate empty array and assign slice (https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array)
def shift(arr, num, fill_value=np.nan):
    shift_arr = np.empty_like(arr)
    if num > 0:
        shift_arr[:num] = fill_value
        shift_arr[num:] = arr[:-num]
    elif num < 0:
        shift_arr[num:] = fill_value
        shift_arr[:num] = arr[-num:]
    else:
        shift_arr[:] = arr
    return shift_arr


"""
Decorator exec_time
"""
def exec_time(function):
    def timed(*args,**kw):
        time_start = perf_counter()
        ret_value=function(*args,**kw)
        time_end = perf_counter()

        execution_time = time_end - time_start

        arguments =", ".join([str(arg) for arg in args] + ["{}={}".format(k, kw[k]) for k in kw])

        smsg ="  {:.2f} sec  for {}({})\n".format(  execution_time , function.__name__, arguments)
        print(smsg)

        with open("execution_time.log",'a') as fel:
            fel.write(smsg)

        return ret_value
    return timed




class  PlotPrintManager():
    _number_of_plots =0
    _max_number_of_plots=1
    _ds_printed = 0
    _folder_for_control_logging =None
    _folder_for_predict_logging = None

    @staticmethod
    def set_Logfolders(control_logging, predict_logging):
        PlotPrintManager._folder_for_control_logging = control_logging
        PlotPrintManager._folder_for_predict_logging = predict_logging

    @staticmethod
    def get_ControlLoggingFolder():
        return PlotPrintManager._folder_for_control_logging

    @staticmethod
    def get_PredictLoggingFolder():
        return PlotPrintManager._folder_for_predict_logging


    @staticmethod
    def get_numberPlots():
        return PlotPrintManager._number_of_plots

    @staticmethod
    def get_maxnumberPlots():
        return PlotPrintManager._max_number_of_plots
    """
    In auto mode  the opened plot windows seriously affect the consumed resources. to avoid proram damage, the plots are
    closed and the charts are saved to png-files.
    In order to print all and show plots, is need to set DEBUP_PRINT_ = 1
       
    """
    @staticmethod
    def isNeedDestroyOpenPlots():

        # bDestroy=False

        # PlotPrintManager._number_of_plots += 1
        # PlotPrintManager._number_of_plots = PlotPrintManager._number_of_plots % PlotPrintManager._max_number_of_plots
        # if PlotPrintManager._number_of_plots == 0:
        #     bDestroy=True

        bDestroy= True
        if DEBUG_PRINT_ :
            bDestroy = False
        return bDestroy

    @staticmethod
    def isNeedPrintDataset():
        bPrint = True
        if DEBUG_PRINT_ :
            return bPrint
        PlotPrintManager._ds_printed +=1

        if PlotPrintManager._ds_printed>0:
            pPrint = False
        return bPrint






