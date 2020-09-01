#!/usr/bin/python3
# Short Term GreenEnergy Load Predictor using Deep Learning - stgelpDP
#

import os
import sys
from datetime import datetime
from pathlib import Path

from tensorflow import random

from predictor.dataset import Dataset

from predictor.cfg import AUTO_PATH, TRAIN_PATH, PREDICT_PATH, CONTROL_PATH, MODES, ACTUAL_MODE
from predictor.cfg import MAGIC_SEED, CSV_PATH, DT_DSET, RCPOWER_DSET, RCPOWER_DSET_AUTO, DISCRET
from predictor.cfg import TEST_CUT_OFF, VAL_CUT_OFF, LOG_FILE_NAME, STOP_ON_CHART_SHOW, PATH_REPOSITORY, ALL_MODELS
from predictor.cfg import EPOCHS, N_STEPS, N_FEATURES, UNITS, FILTERS, KERNEL_SIZE, POOL_SIZE, HIDDEN_NEYRONS, DROPOUT
from predictor.cfg import SEASONALY_PERIOD, PREDICT_LAG, MAX_P, MAX_Q, MAX_D
from predictor.cfg import SCALED_DATA_FOR_AUTO,START_DATE_FOR_AUTO,END_DATE_FOR_AUTO ,TIME_TRUNC_FOR_AUTO
from predictor.cfg import GEO_LIMIT_FOR_AUTO , GEO_IDS_FOR_AUTO
from predictor.control import ControlPlane
from predictor.api import prepareDataset

from predictor.utility import msg2log, exec_time




"""
This Control Plane function creates a dataset and runs specified plane functions (Train plane or Predict Plane)
"""
@exec_time
def drive_STGELPDL(cp):

    ds = Dataset(cp.csv_path, cp.dt_dset, cp.rcpower_dset, cp.discret, cp.fc)  # create dataset
    prepareDataset(cp, ds, cp.fc)

    cp.ts_analysis(ds)

    cp.modes[cp.actual_mode](cp, ds)

    return


def main(argc, argv):


    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")

    with open("execution_time.log", 'w') as fel:
        fel.write("Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S")))

    dir_path = os.path.dirname(os.path.realpath(__file__))

    folder_for_train_logging = Path(dir_path) / "Logs" / TRAIN_PATH / date_time
    folder_for_predict_logging = Path(dir_path) / "Logs" / PREDICT_PATH / date_time
    folder_for_control_logging = Path(dir_path) / "Logs" / CONTROL_PATH / date_time
    folder_for_auto_logging    = Path(dir_path) / "Logs" / AUTO_PATH / date_time

    Path(folder_for_train_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_predict_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_control_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_auto_logging).mkdir(parents=True, exist_ok=True)

    suffics = ".log"
    sRCPOWER_DSET= RCPOWER_DSET
    if ACTUAL_MODE == AUTO_PATH:
        sRCPOWER_DSET= RCPOWER_DSET_AUTO.replace(' ','_')

    file_for_train_logging   = Path(folder_for_train_logging,   sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_predict_logging = Path(folder_for_predict_logging, sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_control_logging = Path(folder_for_control_logging, sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_auto_logging    = Path(folder_for_auto_logging,    sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)

    ft = open(file_for_train_logging,   'w+')
    fp = open(file_for_predict_logging, 'w+')
    fc = open(file_for_control_logging, 'w+')
    fa = open(file_for_auto_logging,    'w+')

    cp = ControlPlane()

    cp.modes = MODES
    cp.actual_mode = ACTUAL_MODE
    cp.auto_path = AUTO_PATH
    cp.train_path = TRAIN_PATH
    cp.predict_path = PREDICT_PATH
    cp.control_path = CONTROL_PATH
    cp.csv_path = CSV_PATH
    cp.rcpower_dset = RCPOWER_DSET
    cp.rcpower_dset_auto = RCPOWER_DSET_AUTO
    cp.dt_dset = DT_DSET
    cp.discret = DISCRET
    cp.test_cut_off = TEST_CUT_OFF
    cp.val_cut_off = VAL_CUT_OFF
    cp.path_repository = PATH_REPOSITORY
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
    cp.folder_train_log   = folder_for_train_logging
    cp.folder_predict_log = folder_for_predict_logging
    cp.folder_auto_log    = folder_for_auto_logging

    cp.log_file_name = LOG_FILE_NAME
    cp.stop_on_chart_show = STOP_ON_CHART_SHOW

    cp.fc = fc
    cp.fp = fp
    cp.ft = ft
    cp.fa = fa

    cp.seasonaly_period = SEASONALY_PERIOD
    cp.predict_lag = PREDICT_LAG
    cp.max_p = MAX_P
    cp.max_q = MAX_Q
    cp.max_d = MAX_D

     # for debug
    # cp.actual_mode = cp.train_path
    # cp.actual_mode = cp.predict_path


    title1 = "Short Term Green Energy Load Predictor {} ".format(cp.actual_mode)
    title2 = "started at"
    msg = "{}\n".format(date_time)
    msg2log("{} (Control Plane) {} ".format(title1, title2), msg, fc)
    msg2log("{} (Train Plane) {} ".format(title1, title2), msg, ft)
    msg2log("{} (Predict Plane) {} ".format(title1, title2), msg, fp)
    msg2log("{} (Auto Management Plane) {} ".format(title1, title2), msg, fa)

    drive_STGELPDL(cp)

    title1 = "Short Term Green Energy Load Predictor "
    title2 = " finished at "
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    msg = "{}\n".format(date_time)
    msg2log("{} Control Plane {}".format(title1,title2), msg, fc)
    msg2log("{} Train Plane {}".format(title1, title2), msg, ft)
    msg2log("{} Predict Plane {}".format(title1,title2), msg, fp)
    msg2log("{} Auto Management Plane {}".format(title1, title2), msg, fa)


    fc.close()
    fp.close()
    ft.close()
    fa.close()


    with open("execution_time.log", 'a') as fel:
        fel.write("Time execution logging finished at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S")))

    return



if __name__ == "__main__":


    random.set_seed(MAGIC_SEED)
    main(len(sys.argv), sys.argv)


