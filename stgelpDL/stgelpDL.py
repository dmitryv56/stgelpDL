#!/usr/bin/python3
# Short Term GreenEnergy Load Predictor using Deep Learning - stgelpDP
#

import os
import sys
from datetime import datetime
from pathlib import Path

from tensorflow import random

from predictor.dataset import dataset

from predictor.cfg import AUTO_PATH, TRAIN_PATH, PREDICT_PATH, CONTROL_PATH, MODES, ACTUAL_MODE
from predictor.cfg import MAGIC_SEED, CSV_PATH, DT_DSET, RCPOWER_DSET, DISCRET
from predictor.cfg import TEST_CUT_OFF, VAL_CUT_OFF, LOG_FILE_NAME, STOP_ON_CHART_SHOW, PATH_REPOSITORY, ALL_MODELS
from predictor.cfg import EPOCHS, N_STEPS, N_FEATURES, UNITS, FILTERS, KERNEL_SIZE, POOL_SIZE, HIDDEN_NEYRONS, DROPOUT

from predictor.control import controlPlane
from predictor.api import prepareDataset

from predictor.utility import msg2log


"""
This Control Plane function creates a dataset and runs specified plane functions (Train plane or Predict Plane)
"""
def drive_STGELPDL(cp):

    ds = dataset(cp.csv_path, cp.dt_dset, cp.rcpower_dset, cp.discret, cp.fc)  # create dataset
    prepareDataset(cp, ds, cp.fc)

    cp.modes[cp.actual_mode](cp, ds)

    pass


def main(argc, argv):


    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_for_train_logging = Path(dir_path) / "Logs" / TRAIN_PATH / date_time
    folder_for_predict_logging = Path(dir_path) / "Logs" / PREDICT_PATH / date_time
    folder_for_control_logging = Path(dir_path) / "Logs" / CONTROL_PATH / date_time
    Path(folder_for_train_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_predict_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_control_logging).mkdir(parents=True, exist_ok=True)
    suffics = ".log"
    file_for_train_logging = Path(folder_for_train_logging, RCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_predict_logging = Path(folder_for_predict_logging, RCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_control_logging = Path(folder_for_control_logging, RCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    ft = open(file_for_train_logging, 'w+')
    fp = open(file_for_predict_logging, 'w+')
    fc = open(file_for_control_logging, 'w+')

    cp = controlPlane()

    cp.modes = MODES
    cp.actual_mode = ACTUAL_MODE
    cp.auto_path = AUTO_PATH
    cp.train_path = TRAIN_PATH
    cp.predict_path = PREDICT_PATH
    cp.control_path = CONTROL_PATH
    cp.csv_path = CSV_PATH
    cp.rcpower_dset = RCPOWER_DSET
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
    cp.folder_train_log = folder_for_train_logging
    cp.folder_predict_log = folder_for_predict_logging
    cp.log_file_name = LOG_FILE_NAME
    cp.stop_on_chart_show = STOP_ON_CHART_SHOW
    cp.fc = fc
    cp.fp = fp
    cp.ft = ft

     # for debug
    # cp.actual_mode = cp.predict_path


    title = "Short Term Green Energy Load Predictor {} started at ".format(cp.actual_mode)
    msg = "{}\n".format(date_time)
    msg2log(title, msg, fc)
    msg2log(title, msg, fp)
    msg2log(title, msg, ft)

    drive_STGELPDL(cp)

    title = "Short Term Green Energy Load Predictor finised at "
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    msg = "{}\n".format(date_time)
    msg2log(title, msg, fc)
    msg2log(title, msg, fp)
    msg2log(title, msg, ft)

    fc.close()
    fp.close()
    ft.close()

    return



if __name__ == "__main__":

    random.set_seed(MAGIC_SEED)
    main(len(sys.argv), sys.argv)


