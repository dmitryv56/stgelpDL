#!/usr/bin/python3
# Short Term GreenEnergy Load Predictor using Deep Learning - stgelpDP
#

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

from predictor.api import prepareDataset
from predictor.cfg import AUTO_PATH, TRAIN_PATH, PREDICT_PATH, CONTROL_PATH, MODES, CSV_PATH, DT_DSET, \
    RCPOWER_DSET_AUTO, DISCRET, \
    TEST_CUT_OFF, VAL_CUT_OFF, LOG_FILE_NAME, STOP_ON_CHART_SHOW, PATH_REPOSITORY, \
    PATH_DATASET_REPOSITORY, PATH_DESCRIPTOR_REPOSITORY, PATH_MAIN_LOG, MAX_LOG_SIZE_BYTES, BACKUP_COUNT, ALL_MODELS, \
    EPOCHS, N_STEPS, N_FEATURES, UNITS, FILTERS, KERNEL_SIZE, POOL_SIZE, HIDDEN_NEYRONS, DROPOUT, \
    SEASONALY_PERIOD, PREDICT_LAG, MAX_P, MAX_Q, MAX_D, \
    FILE_DESCRIPTOR, MODE_IMBALANCE, IMBALANCE_NAME, PROGRAMMED_NAME, DEMAND_NAME, \
    TS_DURATION_DAYS, SEGMENT_SIZE
from predictor.control import ControlPlane
from predictor.dataset import Dataset
from predictor.utility import msg2log, exec_time, PlotPrintManager, OutVerbose, isCLcsvExists

__version__ = '2.0.3'

main_log = Path(PATH_MAIN_LOG, Path(__file__).stem + "_main").with_suffix(".log")

size_handler=RotatingFileHandler(main_log, mode='a', maxBytes=MAX_LOG_SIZE_BYTES, backupCount=BACKUP_COUNT )

logger=logging.getLogger()

logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)



"""
This Control Plane function creates a dataset and runs specified plane functions (Train plane or Predict Plane)
"""


@exec_time
def drive_STGELPDL(cp):
    ds = Dataset(cp.csv_path, cp.dt_dset, cp.rcpower_dset, cp.discret, cp.fc)  # create dataset

    if cp.actual_mode != cp.auto_path:  # the dataset is not created for AUTO_PATH mode
        prepareDataset(cp, ds, cp.fc)
        cp.ts_analysis(ds)

    cp.modes[cp.actual_mode](cp, ds)

    return


def main(argc, argv):
    """

    :param argc:
    :param argv:
    :return:
    """

    # command-line parser
    sDescriptor = 'Short-Term (Green) Energy Load Predictor using Deep Learning and Statistical Time Series methods'
    sCSVhelp = "Absolute path to a source dataset (csv-file). The dataset header and content must meet the requriments README"
    sTSnameHelp = "Time Series (column) name in the dataset, its relevanted for 'train' and 'predict' mode"
    parser = argparse.ArgumentParser(description=sDescriptor)

    parser.add_argument('-m', '--mode', dest='cl_mode', action='store', default='auto',
                        choices=['auto', 'train', 'predict', 'control'],
                        help='Possible modes of operations of the short-term predictor')
    parser.add_argument('-c', '--csv_dataset', dest='cl_dset', action='store', help=sCSVhelp)
    parser.add_argument('-t', '--tsname', dest='cl_tsname', action='store', help=sTSnameHelp)
    parser.add_argument('--verbose', '-v', action='count', dest='cl_verbose', default=0)
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))
    args = parser.parse_args()
    msg="args: mode {}".format(args.cl_mode)
    print("\n\n\n{}".format(msg))
    logger.info(msg)
    OutVerbose.set_verbose_level(args.cl_verbose)

    # command-line parser

    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")

    with open("execution_time.log", 'w') as fel:
        msg="Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
        fel.write(msg)
        logger.info(msg)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    folder_for_train_logging = Path(dir_path) / "Logs" / TRAIN_PATH / date_time
    folder_for_predict_logging = Path(dir_path) / "Logs" / PREDICT_PATH / date_time
    folder_for_control_logging = Path(dir_path) / "Logs" / CONTROL_PATH / date_time
    folder_for_auto_logging = Path(dir_path) / "Logs" / AUTO_PATH / date_time
    folder_for_forecast = Path(dir_path) / "Logs" / "Forecast" / date_time

    folder_for_rt_datasets = Path(PATH_DATASET_REPOSITORY)
    folder_for_descriptor_repostory = Path(PATH_DESCRIPTOR_REPOSITORY)

    Path(folder_for_train_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_predict_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_control_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_auto_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_forecast).mkdir(parents=True, exist_ok=True)

    PlotPrintManager.set_Logfolders(folder_for_control_logging, folder_for_predict_logging)

    suffics = ".log"
    if args.cl_tsname is not None:
        RCPOWER_DSET = args.cl_tsname

    sRCPOWER_DSET = RCPOWER_DSET
    if args.cl_mode == AUTO_PATH:  # if ACTUAL_MODE == AUTO_PATH:
        sRCPOWER_DSET = RCPOWER_DSET_AUTO.replace(' ', '_')
    # if args.cl_tsname is not None:
    #     # RCPOWER_DSET =args.cl_tsname
    #     sRCPOWER_DSET=args.cl_tsname

    file_for_train_logging = Path(folder_for_train_logging, sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_predict_logging = Path(folder_for_predict_logging, sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_control_logging = Path(folder_for_control_logging, sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_auto_logging = Path(folder_for_auto_logging, sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_forecast = Path(folder_for_forecast, sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)

    ft = open(file_for_train_logging, 'w+')
    fp = open(file_for_predict_logging, 'w+')
    fc = open(file_for_control_logging, 'w+')
    fa = open(file_for_auto_logging, 'w+')
    ff = open(file_for_forecast, 'w+')

    cp = ControlPlane()

    cp.modes = MODES
    cp.actual_mode = args.cl_mode  # ACTUAL_MODE
    cp.auto_path = AUTO_PATH
    cp.train_path = TRAIN_PATH
    cp.predict_path = PREDICT_PATH
    cp.control_path = CONTROL_PATH

    if cp.actual_mode == cp.auto_path:
        cp.csv_path = ""
        cp.rcpower_dset = sRCPOWER_DSET
        cp.log_file_name = sRCPOWER_DSET
    elif cp.actual_mode == cp.train_path or cp.actual_mode == cp.predict_path:
        if args.cl_dset is None:
            cp.csv_path = CSV_PATH
        else:

            cp.csv_path = args.cl_dset
            if False == isCLcsvExists(cp.csv_path):
                fc.close()
                fp.close()
                ft.close()
                ff.close()
                sys.exit(2)

        cp.rcpower_dset = RCPOWER_DSET
        cp.log_file_name = LOG_FILE_NAME
    elif cp.actual_mode == cp.control_path:
        cp.csv_path = CSV_PATH
        cp.rcpower_dset = RCPOWER_DSET
        cp.log_file_name = LOG_FILE_NAME
    else:
        msg ="Undefined mode. Exit 1"
        print("Undefined mode. Exit 1")
        logger.error(msg)
        fc.close()
        fp.close()
        ft.close()
        fa.close()
        ff.close()
        exit(1)
    # print('\n\n\n\n\n')
    # print('\x1b[3;31;43m' + sDescriptor + ' runs in ' + cp.actual_mode + ' mode.' + '\x1b[0m')
    # print('\n\n\n\n\n')

    cp.rcpower_dset_auto = RCPOWER_DSET_AUTO
    cp.dt_dset = DT_DSET
    cp.discret = DISCRET
    cp.test_cut_off = TEST_CUT_OFF
    cp.val_cut_off = VAL_CUT_OFF
    cp.path_repository = str(PATH_REPOSITORY)
    cp.path_descriptor = str(PATH_DESCRIPTOR_REPOSITORY)
    cp.all_models = ALL_MODELS
    cp.epochs = EPOCHS
    cp.n_steps = N_STEPS
    cp.n_features = N_FEATURES
    cp.units = UNITS
    cp.filters = FILTERS
    cp.kernel_size = KERNEL_SIZE
    cp.pool_size = POOL_SIZE
    cp.hidden_neyrons = HIDDEN_NEYRONS
    cp.dropout = DROPOUT

    cp.folder_control_log = folder_for_control_logging
    cp.folder_train_log = folder_for_train_logging
    cp.folder_predict_log = folder_for_predict_logging
    cp.folder_auto_log = folder_for_auto_logging
    cp.folder_forecast = folder_for_forecast

    cp.folder_rt_datasets = folder_for_rt_datasets
    cp.folder_descriptor = folder_for_descriptor_repostory
    cp.file_descriptor = FILE_DESCRIPTOR
    cp.stop_on_chart_show = STOP_ON_CHART_SHOW

    cp.fc = fc
    cp.fp = fp
    cp.ft = ft
    cp.fa = fa
    cp.ff = ff

    cp.seasonaly_period = SEASONALY_PERIOD
    cp.predict_lag = PREDICT_LAG
    cp.max_p = MAX_P
    cp.max_q = MAX_Q
    cp.max_d = MAX_D

    ControlPlane.set_modeImbalance(MODE_IMBALANCE)
    ControlPlane.set_modeImbalanceNames((IMBALANCE_NAME, PROGRAMMED_NAME, DEMAND_NAME))
    ControlPlane.set_ts_duration_days(TS_DURATION_DAYS)
    ControlPlane.set_psd_segment_size(SEGMENT_SIZE)
    # for debug
    # cp.actual_mode = cp.train_path
    # cp.actual_mode = cp.predict_path

    title1 = "Short Term Green Energy Load Predictor {} ".format(cp.actual_mode)
    title2 = "started at"
    msg = "{}\n".format(date_time)
    msg2log("{} (Control Plane) {} ".format(title1, title2), msg, fc)
    logger.info("{} (Control Plane) {} : {}".format(title1, title2, msg))
    msg2log("{} (Train Plane) {} ".format(title1, title2), msg, ft)
    logger.info("{} (Train Plane) {} : {}".format(title1, title2, msg))
    msg2log("{} (Predict Plane) {} ".format(title1, title2), msg, fp)
    logger.info("{} (Predict Plane) {} : {}".format(title1, title2, msg))
    msg2log("{} (Auto Management Plane) {} ".format(title1, title2), msg, fa)
    logger.info("{} (Auto Management Plane) {} : {}".format(title1, title2, msg))

    drive_STGELPDL(cp)

    title1 = "Short Term Green Energy Load Predictor "
    title2 = " finished at "
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    msg = "{}\n".format(date_time)
    msg2log("{} Control Plane {}".format(title1, title2), msg, fc)
    logger.info("{} (Control Plane) {} : {}".format(title1, title2, msg))
    msg2log("{} Train Plane {}".format(title1, title2), msg, ft)
    logger.info("{} (Train Plane) {} : {}".format(title1, title2, msg))
    msg2log("{} Predict Plane {}".format(title1, title2), msg, fp)
    logger.info("{} (Predict Plane) {} : {}".format(title1, title2, msg))
    msg2log("{} Auto Management Plane {}".format(title1, title2), msg, fa)
    logger.info("{} (Auto Management Plane) {} : {}".format(title1, title2, msg))

    fc.close()
    fp.close()
    ft.close()
    fa.close()
    ff.close()

    with open("execution_time.log", 'a') as fel:
        msg="Time execution logging finished at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
        fel.write(msg)
        logger.info(msg)
    return


if __name__ == "__main__":
    pass
