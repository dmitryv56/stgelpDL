#!/usr/bin/python3

import sys
from predictor.drive import drive_auto, drive_train, drive_predict, drive_control
"""
Configuration settings that are used in the control, train, predict and management planes.

"""

MAGIC_SEED = 1956


#Title for planes
AUTO_PATH    = 'auto'
TRAIN_PATH   = 'train'
PREDICT_PATH = 'predict'
CONTROL_PATH = 'control'

MODES ={AUTO_PATH: drive_auto, TRAIN_PATH: drive_train,PREDICT_PATH:drive_predict,CONTROL_PATH:drive_control}
ACTUAL_MODE = TRAIN_PATH
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
if sys.platform == "win32":
    CSV_PATH = "C:\\Users\\dmitr_000\\.keras\\datasets\\Imbalance_data.csv"
elif sys.platform == "linux":
    CSV_PATH = "/home/dmitryv/.keras/datasets/Imbalance_data.csv"
else:
    CSV_PATH = "C:\\Users\\dmitr_000\\.keras\\datasets\\Imbalance_data.csv"
# Header names and discrtization in minutes
DT_DSET      = "Date Time"
RCPOWER_DSET = "Imbalance"
DISCRET      = 10

# The time cutoffs for the formation of the validation  and test sequence in the format of the parameter passed
# to the timedelta() like as 'days=<value>' or 'hours=<value>' or 'minutes=<value>'
# We use 'minute' resolution
TEST_CUT_OFF = 48 * DISCRET
VAL_CUT_OFF  = 1000 * DISCRET


# Logging & Charting

LOG_FILE_NAME = RCPOWER_DSET
# Matplotlib.pyplot is used for charting
STOP_ON_CHART_SHOW=False

#Model repository
if sys.platform == 'win32':
    PATH_REPOSITORY = "f:\\model_Repository"
elif sys.platform == 'linux':
    PATH_REPOSITORY = "/home/dmitryv/model_Repository"
else:
    PATH_REPOSITORY = "f:\\model_Repository"

ALL_MODELS = {'MLP':[(0, "mlp_1"), (1,"mlp_2")], 'CNN':[(2,'univar_cnn')],\
              'LSTM':[(3,'vanilla_lstm'),(4,'stacked_lstm'), (5,'bidir_lstm')]}

#training model
EPOCHS     = 100
N_STEPS    = 32
N_FEATURES = 1

# specifical set for for differ types on Neural Nets
#LSTM
UNITS          = 32
#CNN models
FILTERS        = 64
KERNEL_SIZE    = 2
POOL_SIZE      = 2
#MLP model
HIDDEN_NEYRONS = 16
DROPOUT        = 0.2

