#!/usr/bin/python3
"""
Configuration settings that are used in the control, train, predict and management planes.

"""

import sys
from os import getcwd
from pathlib import Path
import configparser

# from predictor.drive import drive_auto, drive_train, drive_predict, drive_control

current_dir = Path(__file__).parent


def getparserobj()->configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(Path(current_dir / "{}.cfg".format(current_dir.stem)))
    return parser

def getItems(section:str, dict_items:dict= None)->(tuple):
    parser = getparserobj()
    res=[]
    for item, typ in dict_items.items():
        if typ=='str':
            res.append(parser[section][item])
        elif typ=='int':
            res.append(int(parser[section][item]))
        elif typ=='float':
            res.append(float(parser[section][item]))
        elif typ=='bool':
            res.append(bool(parser[section][item]))
        elif typ=='eval':
            res.append(eval(parser[section][item]))
        elif typ=='None':
            res.append(None)
        else:
            res.append(parser[section][item])

    ret=tuple(res)
    return ret

""" parse config and get configuration items 
"""

section = 'DEFAULT'
(MAGIC_SEED,MAGIC_TRAIN_CLIENT,MAGIC_PREDICT_CLIENT,GRPC_PORT,GRPC_IP) = getItems(section,dict_items={
    'MAGIC_SEED':'int','MAGIC_TRAIN_CLIENT':'int','MAGIC_PREDICT_CLIENT':'int','GRPC_PORT':'int','GRPC_IP':'str' })

section = 'MODES'
(AUTO_PATH,TRAIN_PATH,PREDICT_PATH,CONTROL_PATH,ACTUAL_MODE) = getItems(section,dict_items={'AUTO_PATH':'str',
    'TRAIN_PATH':'str', 'PREDICT_PATH':'str','CONTROL_PATH':'str','ACTUAL_MODE':'str'})

section = 'IMBALANCE'
(DT_DSET, RCPOWER_DSET, RCPOWER_DSET_AUTO, MODE_IMBALANCE, IMBALANCE_NAME, PROGRAMMED_NAME, DEMAND_NAME) = getItems(
    section,dict_items={'DT_DSET':'str', 'RCPOWER_DSET':'str', 'RCPOWER_DSET_AUTO':'str','MODE_IMBALANCE':'int',
                        'IMBALANCE_NAME':'str', 'PROGRAMMED_NAME':'str', 'DEMAND_NAME':str})

section = 'LOG'
(LOG_FOLDER_NAME, MAIN_SYSTEM_LOG, SERVER_LOG, MAX_LOG_SIZE_BYTES, BACKUP_COUNT ) =getItems(section,
    dict_items={'LOG_FOLDER_NAME':str, 'MAIN_SYSTEM_LOG':str, 'SERVER_LOG':'str', 'MAX_LOG_SIZE_BYTES':'eval',
                'BACKUP_COUNT':'int' })

section = 'ANN'
(EPOCHS, N_STEPS, N_FEATURES, UNITS, FILTERS, KERNEL_SIZE, POOL_SIZE, HIDDEN_NEYRONS, DROPOUT ) = getItems( section,
    dict_items={'EPOCHS':'int', 'N_STEPS':'int', 'N_FEATURES':'int', 'UNITS':'int', 'FILTERS':'int',
                'KERNEL_SIZE':'int', 'POOL_SIZE':'int','HIDDEN_NEYRONS':'int', 'DROPOUT':'float'})

section ='ARIMA'
(SEASONALY_PERIOD, PREDICT_LAG, MAX_P, MAX_Q, MAX_D ) = getItems( section,dict_items={'SEASONALY_PERIOD':'int',
    'PREDICT_LAG':'int', 'MAX_P':'int', 'MAX_Q':'int','MAX_D':'int'} )

section='TIMESERIES'
(DISCRET, TEST_CUT_OFF, VAL_CUT_OFF, TS_DURATION_DAYS, SEGMENT_SIZE ) = getItems( section,dict_items={'DISCRET':'int',
    'TEST_CUT_OFF':'eval', 'VAL_CUT_OFF':'eval', 'TS_DURATION_DAYS':'int', 'SEGMENT_SIZE':'int'})

section='REQUEST'
(FILE_DESCRIPTOR, SCALED_DATA_FOR_AUTO, START_DATE_FOR_AUTO, END_DATE_FOR_AUTO, TIME_TRUNC_FOR_AUTO, GEO_LIMIT_FOR_AUTO,
    GEO_IDS_FOR_AUTO) = getItems( section,dict_items={ 'FILE_DESCRIPTOR':'str', 'SCALED_DATA_FOR_AUTO':'bool',
    'START_DATE_FOR_AUTO':'str', 'END_DATE_FOR_AUTO':'str', 'TIME_TRUNC_FOR_AUTO':'str', 'GEO_LIMIT_FOR_AUTO':'None',
                                                       'GEO_IDS_FOR_AUTO':'None'})

section= 'MIX'
(STOP_ON_CHART_SHOW) = getItems( section,dict_items={'STOP_ON_CHART_SHOW':'bool'})

"""
# Dataset properties. Can be replaced by command-line parameters.
A csv file is used as dataset in the current version.
The dataset comprises a few univariate time series.
For processing we set two columns of the dataset time data column and time series.
The csv file contains header line. Below example comma-separated file
Date Time,       Imbalance,Imbalance + 0.5 MW
08.11.2019 20:00,1.1,      1.6
08.11.2019 20:10,0.3,      0.8
08.11.2019 20:20,0.1,      0.6 
"""

CSV_PATH = Path(Path.home() / ".keras" / "datasets")


# Logging & Charting
PATH_ROOT_FOLDER = Path(Path(__file__).parent.absolute())
PATH_LOG_FOLDER=Path(PATH_ROOT_FOLDER/LOG_FOLDER_NAME)
PATH_LOG_FOLDER.mkdir(parents=True,exist_ok=True)
PATH_MAIN_LOG=Path(PATH_LOG_FOLDER/MAIN_SYSTEM_LOG).with_suffix(".log")
PATH_SERVER_LOG=Path(PATH_LOG_FOLDER/SERVER_LOG).with_suffix(".log")

PATH_REPOSITORY = Path(PATH_ROOT_FOLDER/ "model_Repository")
PATH_REPOSITORY.mkdir(parents=True,exist_ok=True)
PATH_DATASET_REPOSITORY = Path(PATH_ROOT_FOLDER/ "dataset_Repository")
PATH_DATASET_REPOSITORY.mkdir(parents=True,exist_ok=True)
PATH_DESCRIPTOR_REPOSITORY = Path(PATH_ROOT_FOLDER/ "descriptor_Repository")
PATH_DESCRIPTOR_REPOSITORY.mkdir(parents=True,exist_ok=True)

LOG_FILE_NAME = RCPOWER_DSET
#
# if sys.platform == 'win32':
#     PATH_REPOSITORY = Path(Path(Path(getcwd()).drive) / '/' / "model_Repository")
#     PATH_DATASET_REPOSITORY = Path(Path(Path(getcwd()).drive) / '/' / "dataset_Repository")
#     PATH_DESCRIPTOR_REPOSITORY = Path(Path(Path(getcwd()).drive) / '/' / "descriptor_Repository")
#     PATH_MAIN_LOG = Path(Path(Path(getcwd()).drive) / '/' / "stgelpDL_logs")
# elif sys.platform == 'linux':
#     PATH_REPOSITORY = Path(Path.home() / "model_Repository")
#     PATH_DATASET_REPOSITORY = Path(Path.home() / "dataset_Repository")
#     PATH_DESCRIPTOR_REPOSITORY = Path(Path.home() / "descriptor_Repository")
#     PATH_MAIN_LOG = Path(Path.home() /  "stgelpDL_logs")
# else:
#     PATH_REPOSITORY = Path(Path(Path(getcwd()).drive) / '/' / "model_Repository")
#     PATH_DATASET_REPOSITORY = Path(Path(Path(getcwd()).drive) / '/' / "dataset_Repository")
#     PATH_DESCRIPTOR_REPOSITORY = Path(Path(Path(getcwd()).drive) / '/' / "descriptor_Repository")
#     PATH_MAIN_LOG = Path(Path(Path(getcwd()).drive) / '/' / "stgelpDL_logs")
#
# print('{} \n{} \n{}'.format(PATH_REPOSITORY, PATH_DATASET_REPOSITORY, PATH_DESCRIPTOR_REPOSITORY))
# Path.mkdir(PATH_DATASET_REPOSITORY, parents=True, exist_ok=True)
# Path.mkdir(PATH_DESCRIPTOR_REPOSITORY, parents=True, exist_ok=True)
# Path.mkdir(PATH_MAIN_LOG, parents=True, exist_ok=True)

ALL_MODELS = {'MLP': [(0, "mlp_1"), (1, "mlp_2")], 'CNN': [(2, 'univar_cnn')], \
              'LSTM': [(3, 'vanilla_lstm'), (4, 'stacked_lstm'), (5, 'bidir_lstm')], \
              'tsARIMA': [(6, 'seasonal_arima'), (7, 'best_arima')]}





