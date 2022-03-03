#!/usr/bin/python3

""" Hidden Markov Model for the analysis of the abrupt changing properties of the process of the green electricity grid.
    HMM_a(brupt)c(hanging)p(roperties of the)p(rocess of )g(reen)e(lectrisity)g(rid).py

The given time series is an imbalance of consumers and electricity generation.

The parameters of H(idden)M(arkov)M(odel) are estimated.
First, the state set is defined according by rules described below. The dataset contained a timeserires (TS),
timestamp labels(labels) and exogenious is read. The 'pattern' of the dataset is following:

Date Time, Imbalance =[Programmed_demand - Real_demand],  Diesel_Power, WindTurbine_Power, Hydrawlic,
HydrawlicTurbine_Power, Pump_Power

All powers are measured in MWatt.
The TS is 'Imbalance', exogenous are 'Diesel_Power, WindTurbine_Power, Hydrawlic,HydrawlicTurbine_Power, Pump_Power'.
Three modes are provided in the program: 5 , 8 or 36 states. The specific mode is configured before program launching.
The rules for assigning i-th observation of TS, i=0,len(TS), to a certain state j, j belongs to set of states, are as
follows:

Mode 5 states:
A- priori, we assume that the process can be in one of five states:
 - balance of consumer power and generated power (0)
 - lack of electricity capacity,  (1)
 - sharp increase in consumer power(2)
 - dominance of generation power(3)
 - sharp increase in the generation power(4)
The symbols  of states {0,1,2,3,4} or indexes of the states are compatible with tensorflow_probability.hmm.

    State 0 :          'BLNC': 'Balance of consumer power and generated power',
    State 1 :          'LACK': 'Lack electrical power',
                       'SIPC': 'Sharp increase in power consumption',
                       'EXCS': 'Excess electrical power',
    State 4 :          'SIGP': 'Sharp increase in generation power'




Mode 7 states:
A- priori, we assume that the process can be in one of 8 states:
 - only disels generate the power (0)
 - only wind turbine generates the power  (1)
 - disels and wind turbine generate the power(2)
 - disels and hydrawlic turbine generate the power(3)
 - wind turbine and hydrawlic turbine generate the power(4)
 - diesel, wind turbine and hydrawlic turbine generate the power(5)
 - wind turbine  generates the power and hydrawlic pump accumulates(6)

The symbols  of states {0,1,2,3,4,5,6} or indexes of the states are compatible with tensorflow_probability.hmm.

    State 0 :          'D__': 'Diesels, WindTurbine Off, Hydrawlic Off',
    State 1 :          'W__': 'Diesels Off,WindTurbine, Hydrawlic Off',
                       'DW_': 'Diesels, WindTurbine, Hydrawlic Off'},
                       'DT_': 'Diesels, WindTurbines Off, Hydrawlic Turbine',
                       'WT_': 'Diesels Off, WindTurbines, Hydrawlic Turbine',
                       'DWT': 'Diesels , WindTurbines, Hydrawlic Turbine',
    State 6            'WP_': 'Diesels Off, WindTurbines, Hydrawlic Pump'




Mode 36 states:
The symbols  of states {0,1,2,...,34,35} or indexes of the states are compatible with tensorflow_probability.hmm.

    State 0 :          '_Dxxx': 'Diesels, WindTurbine Off, Hydrawlic Off'
    State 1 :          'iDxxx': 'IncPower Diesels, WindTurbine Off, Hydrawlic Off'},
    State 2 :          'xxxW_': 'Diesels Off, WindTurbine, Hydrawlic Off',
    State 3:           'xxiW_': 'Diesels Off, IncPower WindTurbine, Hydrawlic Off',
    State 4 :          'xxdW_': 'Diesels Off, DecPower WindTurbine, Hydrawlic Off',
                       '_D_W_': 'Diesels, WindTurbine, Hydrawlic Off',
                       '_DiW_': 'Diesels, IncPower WindTurbine, Hydrawlic Off',
                       '_DdW_': 'Diesels, DecPower WindTurbine, Hydrawlic Off',
                       'iD_W_': 'IncPower Diesels, WindTurbine, Hydrawlic Off',
                       'iDdW_': 'IncPower Diesels, DecPower WindTurbine, Hydrawlic Off',
                       'dD_W_': 'DecPower Diesels, WindTurbine, Hydrawlic Off',
                       'iDiW_': 'DecPower Diesels, IncPower WindTurbine, Hydrawlic Off',
                       '_DxxT': 'Diesels, WindTurbine Off, Hydrawlic Turbine',
                       'iDxxT': 'IncPower Diesels, WindTurbine Off,Hydrawlic Turbine',
                       'dDxxT': 'DecPower Diesels, WindTurbine Off, Hydrawlic Turbine',
                       'xxxWT': 'Diesels Off, WindTurbine, Hydrawlic Turbine',
                       'xxiWT': 'Diesels Off, IncPower WindTurbine, Hydrawlic Turbine',
                       'xxdWT': 'Diesels Off, DecPower WindTurbine, Hydrawlic Turbine',
                       '_D_WT': 'Diesels, WindTurbine, Hydrawlic Turbine',
                       '_DiWT': 'Diesels, IncPower WindTurbine, Hydrawlic Turbine',
                       '_DdWT': 'Diesels, DecPower WindTurbine, Hydrawlic Turbine',
                       'iD_WT': 'IncPower Diesels, WindTurbine, Hydrawlic Turbine',
                       'iDdWT': 'IncPower Diesels, DecPower WindTurbine, Hydrawlic Turbine',
                       'dD_WT': 'DecPower Diesels, WindTurbine, Hydrawlic Turbine',
                       'dDiWT': 'DecPower Diesels, IncPower WindTurbine, Hydrawlic Turbine',

                       'xxxWP': 'Diesels Off, WindTurbine, Hydrawlic Pump',
                       'xxiWP': 'Diesels Off, IncPower WindTurbine, Hydrawlic Pump',
                       'dWxxP': 'Diesels Off, DecPower WindTurbine, Hydrawlic Pump',
                       '_D_WP': 'Diesels, WindTurbine, Hydrawlic Pump',
                       '_DiWP': 'Diesels, IncPower WindTurbine, Hydrawlic Pump',
                       '_DdWP': 'Diesels, DecPower WindTurbine, Hydrawlic Pump',
                       'iD_WP': 'IncPower Diesels, WindTurbine, Hydrawlic Pump',
                       'iDiWP': 'IncPower Diesels, IncPower WindTurbine, Hydrawlic Pump',
                       'iDdWP': 'IncPower Diesels, DecPower WindTurbine, Hydrawlic Pump',
                       'dD_WP': 'DecPower Diesels, WindTurbine, Hydrawlic Pump',
    State 35 :         'dDiWP': 'DecPower Diesels, IncPower WindTurbine, Hydrawlic Pump',

    where Disel and Wind Turbine  can be in following modes: (Off, work, increment power, decrement power),
         Hydrawlic has one of (Off, Hydrawlic Turbine, Hydrawlic Pump).

         'Increment power ' if Y[i]-y[i-1]>a, 'Decrement power' if y[i]-y[i-1]<-a, else 'work'

Second, parameters of model initialization.  After assignung state for each observation the initial probabilites for
each state, transition from state to state probabilities and parameters of emission (mean and std) are estimated over
TS.

Thirdly, the Viterby path is found out for this TS.



"""

import sys
import os
from datetime import datetime
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

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
D_MODES={LEARN_HMM: "Train Model Mode", PREDICT_HMM: "Predict Model Mode"}
PROGRAM_TITLE ="HMM Abrupt Changing Properties of the Process of (green) Electricity Grid Predictor"
MODE_STATES = 5

""" Set main system log for all modules exclude cfg.py.
SERVER_LOG is used in server.service module"""
MAX_LOG_SIZE_BYTES = 4*1024 *1024
BACKUP_COUNT = 2

MAIN_SYSTEM_LOG="main_hmm_{}_states".format(MODE_STATES)
LOG_FOLDER_NAME="Logs"
REPOSITORY_FOLDER_NAME = "hmm_Repository"
PATH_ROOT_FOLDER = Path(Path(__file__).parent.absolute())
PATH_LOG_FOLDER = Path(PATH_ROOT_FOLDER/LOG_FOLDER_NAME)
PATH_LOG_FOLDER.mkdir(parents=True, exist_ok=True)
PATH_MAIN_LOG = Path(PATH_LOG_FOLDER/MAIN_SYSTEM_LOG).with_suffix(".log")
PATH_REPOSITORY = Path(PATH_ROOT_FOLDER/REPOSITORY_FOLDER_NAME)
PATH_REPOSITORY.mkdir(parents=True, exist_ok=True)

size_handler = RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes=int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)


def main(argc,argv):
    logger.info("\n\n{}\n".format(PROGRAM_TITLE))

    workmode = LEARN_HMM
    # workmode = PREDICT_HMM
    csv_file      = CSV_PATH
    data_col_name = DATA_COL_NAME
    dt_col_name   = DT_COL_NAME

    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    msg = "Time execution logging started at {}\n\n".format(datetime.now().strftime(cSFMT))
    logger.info(msg)
    with open("execution_time.log", 'w') as fel:
        fel.write(msg)
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
    msg="Aux.logs:\n{}\n{}\n{}".format(file_for_control_logging, file_for_train_logging, file_for_predict_logging)
    logger.info(msg)
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
    msg = f"""Model       :  {cp.actual_mode}
              Dataset     :  {cp.csv_path}
              Timestamp   :  {cp.dt_dset}
              Time Series :  {cp.rcpower_dset}
              Discret, sec:  {cp.discret}
              Mode        :  {D_MODES[workmode]} 
              
            """
    logger.info(msg)
    if workmode == LEARN_HMM:
        trainPath(cp, MODE_STATES)
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

    msg  = "Time execution logging finished at {}\n\n".format(datetime.now().strftime(cSFMT))
    logger.info(msg)

    with open("execution_time.log", 'a') as fel:
        fel.write(msg)

    return 0

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

