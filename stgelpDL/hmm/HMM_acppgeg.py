#!/usr/bin/python3

""" Hidden Markov Model for the analysis of the abrupt changing properties of the process of the green electrisity grid.
    HMM_a(brupt)c(hanging)p(roperties of the)p(rocess of )g(reen)e(lectrisity)g(rid).py

The given time series is an imbalance of consumers and electricity generation.
A- priori, we assume that the process can be in one of five states:
 - balance of consumer power and generated power (0)
 - lack of electricity capacity,  (1)
 - sharp increase in consumer power(2)
 - dominance of generation power(3)
 - sharp increase in the generation power(4)
The symbols  of states {0,1,2,3,4} or indexes of the states are compatible with tensorflow_probability.hmm.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hmm.HMM import hmm_gmix
from hmm.HMM_drive import imprLogDist
from hmm.api_acppgeg import STATES_DICT,CSV_PATH,DATA_COL_NAME,DT_COL_NAME, readDataset, plotStates, trainHMMprob, \
    trainPath,predictPath,printListState

from predictor.cfg import PATH_DATASET_REPOSITORY, DT_DSET, DISCRET, LOG_FILE_NAME, \
    PROGRAMMED_NAME, DEMAND_NAME, CONTROL_PATH, TRAIN_PATH,PREDICT_PATH, PATH_REPOSITORY, MODE_IMBALANCE, \
    IMBALANCE_NAME, TS_DURATION_DAYS, SEGMENT_SIZE, RCPOWER_DSET_AUTO
from predictor.control import ControlPlane
from predictor.utility import msg2log, PlotPrintManager, cSFMT, logDictArima

LEARN_HMM = 0
PREDICT_HMM = 1
PROGRAM_TITLE ="HMM Abrupt Changing Properties of the Process of (green) Electricity Grid Predictor"





def main(argc,argv):

    workmode = LEARN_HMM
    # workmode = PREDICT_HMM

    csv_file      = CSV_PATH
    data_col_name = DATA_COL_NAME
    dt_col_name   = DT_COL_NAME


    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    with open("execution_time.log", 'w') as fel:
        fel.write("Time execution logging started at {}\n\n".format(datetime.now().strftime(cSFMT)))
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # (mypath, LISTCSV) = listCSV(PATH_DATASET_REPOSITORY, "Demanda_el_huerro", "Archieve")
    folder_for_control_logging = Path(dir_path) / "Logs" / CONTROL_PATH / date_time
    folder_for_predict_logging = Path(dir_path) / "Logs" / PREDICT_PATH / date_time
    folder_for_train_logging = Path(dir_path) / "Logs" / TRAIN_PATH / date_time


    # CSV_PATH = Path(PATH_DATASET_REPOSITORY / "Demanda_el_huerro" / "Imbalance.csv")
    RCPOWER_DSET = data_col_name   #RCPOWER_DSET_AUTO.replace(' ', '_')


    folder_for_rt_datasets = Path(PATH_DATASET_REPOSITORY)
    Path(folder_for_control_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_predict_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_train_logging).mkdir(parents=True, exist_ok=True)
    PlotPrintManager.set_LogfoldersExt(folder_for_control_logging, folder_for_predict_logging, folder_for_train_logging)

    suffics = ".log"
    sRCPOWER_DSET = RCPOWER_DSET
    file_for_predict_logging = Path(folder_for_predict_logging, data_col_name + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_control_logging = Path(folder_for_control_logging, data_col_name + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_train_logging = Path(folder_for_train_logging, data_col_name + "_" + Path(__file__).stem).with_suffix(
        suffics)
    fp = open(file_for_predict_logging, 'w+')
    fc = open(file_for_control_logging, 'w+')
    ft = open(file_for_train_logging, 'w+')

    cp = ControlPlane()
    cp.actual_mode = "Hidden Markov Model"
    cp.csv_path = CSV_PATH
    cp.rcpower_dset = data_col_name
    cp.log_file_name = LOG_FILE_NAME
    cp.rcpower_dset_auto = data_col_name
    cp.dt_dset = dt_col_name
    cp.discret = DISCRET
    cp.path_repository = str(PATH_REPOSITORY)
    cp.folder_control_log = folder_for_control_logging
    cp.folder_predict_log = folder_for_predict_logging
    cp.fc = fc
    cp.fp = fp
    cp.ft = ft

    # TODO Are we  need this code?
    ControlPlane.set_modeImbalance(MODE_IMBALANCE)
    ControlPlane.set_modeImbalanceNames((IMBALANCE_NAME, PROGRAMMED_NAME, DEMAND_NAME))
    ControlPlane.set_ts_duration_days(TS_DURATION_DAYS)
    ControlPlane.set_psd_segment_size(SEGMENT_SIZE)

    msg2log(PROGRAM_TITLE, " (Control Plane) started at {} ".format(date_time), fc)
    msg2log(PROGRAM_TITLE, " (Predict Plane) started at {} ".format(date_time), fp)
    msg2log(PROGRAM_TITLE, " (Train Plane) started at {} ".format(date_time), ft)

    # printListState(STATES_DICT, fc)
    # printListState(STATES_DICT, fp)
    # printListState(STATES_DICT, ft)

    if workmode == LEARN_HMM:
        trainPath(cp)
    elif workmode == PREDICT_HMM:
        predictPath(cp, cp.csv_path)
    else:
        pass

    # ds, states = readDataset(csv_file, data_col_name, dt_col_name,cp.fc)
    # title="{} and States".format(data_col_name)
    #
    # plotStates(ds, data_col_name, states, title, cp.folder_control_log, start_index = 0,end_index = 512, f=cp.fc)
    # pai, transDist, emisDist,list_states = trainHMMprob(ds, data_col_name, states, cp)
    #
    # imprLogDist(pai,       list_states, [0],            "Initial Distribution",    f=cp.fc)
    # imprLogDist(transDist, list_states, list_states,    "Transition Distribution", f=cp.fc)
    # imprLogDist(emisDist,  list_states, ['mean','std'], "Emission Distribution",   f=cp.fc)

    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")

    msg2log(PROGRAM_TITLE, " (Control Plane) finished at {} ".format(date_time), fc)
    msg2log(PROGRAM_TITLE, " (Predict Plane) finished at {} ".format(date_time), fp)
    msg2log(PROGRAM_TITLE, " (Train Plane) finished at {} ".format(date_time), ft)

    fc.close()
    fp.close()
    ft.close()
    with open("execution_time.log", 'a') as fel:
        fel.write("Time execution logging finished at {}\n\n".format(datetime.now().strftime(cSFMT)))

    return 0





if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

