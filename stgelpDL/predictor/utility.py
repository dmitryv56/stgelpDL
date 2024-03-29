#!/usr/bin/python3
""" Utiliy.py


"""

import sys
from datetime import timedelta
from pathlib import Path
from time import perf_counter, sleep

import dateutil.parser
import numpy as np
import logging

logger=logging.getLogger(__name__)

""" constants """
cSFMT = "%Y-%m-%d %H:%M:%S"

""" Model training after N dataset updates"""
PERIOD_MODEL_RETRAIN = 4


def isCLcsvExists(cl_csv):
    ret = False
    csvPath = Path(cl_csv)
    if csvPath.exists() and csvPath.is_file():
        ret = True
    else:
        msg = f"""\n
             File does not exist : {cl_csv}
             The program should be terminated.\n   
        """
        print(msg)
        logger.error(msg)
    return ret


"""
tsBoundaries2log prints and logs time series boundaries characterisics (min,max,len)
"""


class OutVerbose():
    _lvl_verbose = 0

    @staticmethod
    def set_verbose_level(val):
        OutVerbose._lvl_verbose = val

    @staticmethod
    def get_verbose_level():
        return OutVerbose._lvl_verbose


def tsBoundaries2log(title, df, dt_dset, rcpower_dset, f=None):
    """

    :param title: - title for print
    :param df:    - pandas DataFrame object
    :param dt_dset: - Data/Time column name
    :param rcpower_dset: - time series column name
    :param f:
    :return:
    """
    message = f"""
            Time series (TS) title : {title}
            TS shape : {df.shape}

            TS length : {len(df[dt_dset])}
            TS starts at : {df[dt_dset].min()}
            TS ends at   : {df[dt_dset].max()}
            TS minimal value : {df[rcpower_dset].min()}
            TS maximal value : {df[rcpower_dset].max()}

    """
    msg2log(tsBoundaries2log.__name__, message, f)
    logging.info(message)

    return


def tsSubset2log(dt_dset, rcpower_dset, df_train, df_val=None, df_test=None, f=None):
    pass
    msg = '    Train dataset\nTrain dates: {} to {}'.format(df_train[dt_dset].min(), df_train[dt_dset].max())
    msg2log(tsSubset2log.__name__, msg, f)
    logging.info(msg)

    if PlotPrintManager.isNeedPrintDataset():
        for i in range(len(df_train)):
            msg = '{} {}'.format(df_train[dt_dset][i], df_train[rcpower_dset][i])
            msg2log(" ", msg, f)
            logging.debug(msg)

    if df_val is not None:

        msg = '\n    Validation dataset\nValidation dates: {} to {}'.format(df_val[dt_dset].min(),
                                                                            df_val[dt_dset].max())
        msg2log(tsSubset2log.__name__, msg, f)
        logging.debug(msg)

        if PlotPrintManager.isNeedPrintDataset():
            for i in range(len(df_train), len(df_train) + len(df_val)):
                msg = '{} {}'.format(df_val[dt_dset][i], df_val[rcpower_dset][i])
                msg2log(" ", msg, f)
                logging.debug(msg)

    if df_test is not None:

        msg = '\n     Test dataset\nTest dates: {} to {}'.format(df_test[dt_dset].min(), df_test[dt_dset].max())
        msg2log(tsSubset2log.__name__, msg, f)
        logging.debug(msg)

        start = len(df_train) if df_val is None else len(df_train) + len(df_val)
        stop = len(df_train) + len(df_test) if df_val is None else len(df_train) + len(df_val) + len(df_test)
        for i in range(start, stop):
            msg = '{} {}\n'.format(df_test[dt_dset][i], df_test[rcpower_dset][i])
            msg2log(" ", msg, f)
            logging.debug(msg)
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
    message = f"""
                =======================================================================================================
                Dataset Properties
                csv_path                              : {csv_path}
                dt_dset                               : {dt_dset}
                rcpower_dset                          : {rcpower_dset}
                discreet (minutes)                    : {discret}
                
                Dataset cut-off properties
                cut-off test sequence (minutes)       : {test_cut_off}
                cut-off validation sequence (minutes) : {val_cut_off}

                Training properties
                time steps for learning sequence      : {n_steps}
                features                              : {n_features}
                epochs                                : {n_epochs}
                =======================================================================================================

    """
    msg2log(dataset_properties2log.__name__, message, f)
    logging.info(message)

    return


def msg2log(funcname, msg, f=None):
    if funcname is not None:
        print("\n{}: {}".format(funcname, msg))
        if f is not None:
            f.write("\n{}: {}\n".format(funcname, msg))
    else:
        print("\n{}".format(msg))
        if f is not None:
            f.write("{}\n".format(msg))


def vector_logging(title, seq, print_weigth, f=None):
    if f is None:
        return
    f.write("{}\n".format(title))
    k = 0
    line = 0
    f.write("{}: ".format(line))
    for i in range(len(seq)):
        f.write(" {}".format(seq[i]))
        k = k + 1
        k = k % print_weigth
        if k == 0:
            line = line + 1
            f.write("\n{}: ".format(line))

    return


""" This function logs the Spectral Density or two-side AutoCorrelation"""


def psd_logging(title: str, freqORlag: np.array, psdORacorr: np.array, functype: str = 'psd'):
    """
    :param title:  title
    :param freqORlag:   numpy array of frequencies or lags.  The lag values starts from -lag_max, -lag_max+1, ...,-1,0,
     1,..., lag_max-1,lag_max
    :param psdORacorr:    numpy array of psd or acorr values
    :param functype: {'psd', 'acorr'}, 'psd' for spectral density (default),'acorr' for autocorrelation/
    :return:
    """
    sfile = "{}.log".format(title.replace(' ', '_'))
    sFolder = PlotPrintManager.get_ControlLoggingFolder()
    filePrint = Path(sFolder) / (sfile)
    stemplate = "{:>5d}  {:>7.7f} {:>15.2f}\n"
    sHeaderLine = '\n Index  Frequency(Hz) Psd \n'
    if functype == 'acorr':
        sHeaderLine = '\n Index  Lag  Autocorrelation \n'
    with open(filePrint, 'w+') as fpsd:
        fpsd.write(sHeaderLine)
        for i in range(len(psdORacorr)):
            fpsd.write(stemplate.format(i, freqORlag[i], psdORacorr[i]))
        fpsd.write('\n')
    return


def logDictArima(dct, indent=0, f=None):
    """ This function recursive  puts theARIMA-model dictionary.

    :param dct:   object <ARIMA-model>.to_dict()
    :param indent: indent for prints
    :param f:
    :return:
    """

    deltaindent = 4
    try:
        if isinstance(dct, (int, float, bool, str)):
            s = ' {}'.format(str(dct).rjust(indent))
            msg2logDictArima(indent, s, f)
            return
        elif isinstance(dct, (list, np.ndarray)):
            for i in range(len(dct)):
                if i % 8 == 0:
                    msg2logDictArima(indent, '\n', f)
                logDictArima(dct[i], indent + int(deltaindent / 2), f)

            msg2logDictArima(indent, '\n', f)
            return
        elif isinstance(dct, tuple):
            logDictArima(list(dct), indent, f)
        elif isinstance(dct, dict):
            for k, v in dct.items():
                s = '\n{}:'.format(str(k).rjust(indent))
                msg2logDictArima(indent, s, f)
                logDictArima(v, indent + deltaindent, f)
            msg2logDictArima(indent, '\n', f)
            return
        else:
            msg2logDictArima(indent, type(dct), f)
    except:
        message = f"""
                Oops! Unexpected error!
                Error : {sys.exc_info()[0]}
                (continue) : {sys.exc_info()[1]}
        """
        msg2log(logDictArima.__name__, message, f)
    return


def msg2logDictArima(indent, msg, f=None):
    try:
        st = '{' + ':>{}s'.format(indent) + '} {}'
        if f is not None:
            f.write(st.format(" ", msg))
    except:
        message = f"""
                        Oops! Unexpected error!
                        Error : {sys.exc_info()[0]}
                        (continue) : {sys.exc_info()[1]}
                """
        msg2log(msg2logDictArima.__name__, message, f)


def incDateStr(inDateTime: str, days: int = 0, seconds: int = 0, minutes: int = 0, hours: int = 0,
               weeks: int = 0) -> str:
    tDateTime = dateutil.parser.parse(inDateTime)
    return (tDateTime + timedelta(days=days, seconds=seconds, minutes=minutes, hours=hours, weeks=weeks)).strftime(
        cSFMT)


def decDateStr(inDateTime: str, days: int = 0, seconds: int = 0, minutes: int = 0, hours: int = 0,
               weeks: int = 0) -> str:
    tDateTime = dateutil.parser.parse(inDateTime)
    return (tDateTime - timedelta(days=days, seconds=seconds, minutes=minutes, hours=hours, weeks=weeks)).strftime(
        cSFMT)


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
    def timed(*args, **kw):
        time_start = perf_counter()
        ret_value = function(*args, **kw)
        time_end = perf_counter()

        execution_time = time_end - time_start

        arguments = ", ".join([str(arg) for arg in args] + ["{}={}".format(k, kw[k]) for k in kw])

        smsg = "  {:.2f} sec  for {}({})\n".format(execution_time, function.__name__, arguments)
        print(smsg)

        with open("execution_time.log", 'a') as fel:
            fel.write(smsg)

        return ret_value

    return timed


class PlotPrintManager():
    _number_of_plots = 0
    _max_number_of_plots = 1
    _ds_printed = 0
    _folder_for_control_logging = None
    _folder_for_predict_logging = None
    _folder_for_train_logging = None
    _list_bak_png = []

    @staticmethod
    def set_Logfolders(control_logging, predict_logging):
        PlotPrintManager._folder_for_control_logging = control_logging
        PlotPrintManager._folder_for_predict_logging = predict_logging

    @staticmethod
    def set_LogfoldersExt(control_logging, predict_logging,train_logging):
        PlotPrintManager._folder_for_control_logging = control_logging
        PlotPrintManager._folder_for_predict_logging = predict_logging
        PlotPrintManager._folder_for_train_logging   = train_logging

    @staticmethod
    def get_ControlLoggingFolder():
        return PlotPrintManager._folder_for_control_logging

    @staticmethod
    def get_PredictLoggingFolder():
        return PlotPrintManager._folder_for_predict_logging

    @staticmethod
    def get_TrainLoggingFolder():
        return PlotPrintManager._folder_for_train_logging

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
        sleep(2)
        bDestroy = True
        if OutVerbose.get_verbose_level() > 3:
            bDestroy = False
        return bDestroy

    @staticmethod
    def isNeedPrintDataset():
        bPrint = True
        if OutVerbose.get_verbose_level() < 2:
            bPrint = False

        return bPrint

    @staticmethod
    def addPng2Baklist(strBak):
        PlotPrintManager._list_bak_png.append(strBak)
        PlotPrintManager.remPngBak()

    @staticmethod
    def remPngBak():
        if len(PlotPrintManager._list_bak_png) < 2:
            return
        try:
            strBak = PlotPrintManager._list_bak_png.pop(0)
            p = Path(strBak)
            p.unlink(missing_ok=True)
            print('{} removed'.format(strBak))

        except Exception as e:
            print('{} exception:\n {}'.format(PlotPrintManager.remPngBak.__name__, e))
        return


def logMatrix(X:np.array,title:str=None,f:object = None):
    if title is not None:
        msg2log(None,title,f)
    (n,m)=X.shape
    z=np.array([i for i in range(n)])
    z=z.reshape((n,1))
    a=np.append(z,X,axis=1)
    s = '\n'.join([''.join(['{:10.4f}'.format(item) for item in row]) for row in a])
    if f is not None:
        f.write(s)

    return



