#!/usr/bin/python3
# Short Term GreenEnergy Load Predictor using Deep Learning - stgelpDP
#

from tensorflow import random
from predictor.NNmodel import NNmodel, MLP, LSTM, CNN

from predictor.predictor import Predictor
from predictor.dataset import dataset

from predictor.cfg import AUTO_PATH, TRAIN_PATH, PREDICT_PATH, CONTROL_PATH, MODES, ACTUAL_MODE
from predictor.cfg import MAGIC_SEED, CSV_PATH, DT_DSET, RCPOWER_DSET, DISCRET
from predictor.cfg import TEST_CUT_OFF, VAL_CUT_OFF, LOG_FILE_NAME, STOP_ON_CHART_SHOW, PATH_REPOSITORY, ALL_MODELS
from predictor.cfg import EPOCHS, N_STEPS, N_FEATURES, UNITS, FILTERS, KERNEL_SIZE, POOL_SIZE, HIDDEN_NEYRONS, DROPOUT

from predictor.control import controlPlane
from predictor.drive import prepareDataset

import copy
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from pickle import dump, load

from predictor.api import autocorr, autocorr_firstDiff, autocorr_secondDiff, show_autocorr
from predictor.utility import msg2log, shift
import matplotlib.pyplot as plt
import numpy as np


def drive_STGELPDP(cp):

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

    cp.fc = fc
    cp.fp = fp
    cp.ft = ft

     # for debug
    cp.actual_mode = cp.predict_path
    # fullp='F:\\model_Repository\\Imbalance\\MLP\\mlp_2'
    # p=Path(fullp)
    # ppar = p.parts
    # ll=len(ppar)
    # name1=ppar[ll-1]
    # name2=ppar[ll-2]

    # import json
    # dict_srz={}
    # dict_srz[cp.rcpower_dset] = ['a','b']
    # filepath_to_serialize = Path(Path(cp.path_repository) / cp.rcpower_dset / cp.rcpower_dset).with_suffix('.json')
    # if sys.platform == 'win32':
    #     filep = str(filepath_to_serialize).replace('/', '\\')
    # else:
    #     filep = filepath_to_serialize
    # with open(filep, 'w') as fw:
    #     x = json.dump(dict_srz, fw)
    # x=None
    # filepath_to_serialize1 = Path(Path(cp.path_repository) / cp.rcpower_dset / cp.rcpower_dset).with_suffix('.json')
    # if sys.platform == 'win32':
    #     filep1 = str(filepath_to_serialize1).replace('/', '\\')
    # else:
    #     filep1 = filepath_to_serialize1
    # with open(filep1, 'r') as fr:
    #     x = json.load( fr)
    # print(x)


    title = "Short Term Green Energy Load Predictor started at "
    msg = "{}\n".format(date_time)
    msg2log(title, msg, fc)
    msg2log(title, msg, fp)
    msg2log(title, msg, ft)

    drive_STGELPDP(cp)

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
    # y = [28, 28, 26, 19, 16, 24, 26, 24, 24, 29, 29, 27, 31, 26, 38, 23, 13, 14, 28, 19, 19, \
    #      17, 22, 2, 4, 5, 7, 8, 14, 14, 23]
    # with open("Autocorrelation.log", "w") as f:
    #     show_autocorr(y, 144, "Imbalance", "Logs", False, f)
    # f.close()
    # d ={"Imbalace":["F:\\model_Repository\\Imbalance\\CNN\\univar_cnn", "F:\\model_Repository\\Imbalance\\LSTM\\bidir_lstm"]}
    # import json
    # with open('F:\\model_Repository\\Imbalance\\json.json','w') as fw:
    #
    #     x=json.dump(d,fw)
    # print(x)
    #
    # with open('F:\\model_Repository\\Imbalance\\json.json', 'r') as fr:
    #     x=json.load(fr)
    # print(x)
    # from tensorflow import keras
    #
    #
    # model = keras.models.load_model('F:\\model_Repository\\Imbalance\\LSTM\\bidir_lstm')
    # model.summary()
    # # with open('F:\\model_Repository\\Imbalance\\LSTM\\bidir_lstm\\scaler.pkl') as ffp:
    # #     scaler =load(ffp)
    # forcast=[]
    # x_scaled = np.random.randn(32)
    # for i in range(16):
    #
    #     xx_scaled = x_scaled.reshape((1, x_scaled.shape[0], 1))
    #     y = model.predict(xx_scaled)
    #     x_scaled = shift(x_scaled,-1)
    #     x_scaled[31]=y[0][0]
    #     forcast.append(y[0][0])
    # print(forcast)
    # y=np.array([1,2,3,4,5])
    # y1 =shift(y, -1, 6)


    random.set_seed(MAGIC_SEED)
    main(len(sys.argv), sys.argv)


