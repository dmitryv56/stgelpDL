#!/usr/bin/python3
"""
Configuration settings that are used in the control, train, predict and management planes.

"""

from os import getcwd
import sys
from pathlib import Path

from predictor.drive import drive_auto, drive_train, drive_predict, drive_control

MAGIC_SEED = 1956
#Title for planes
AUTO_PATH    = 'auto'
TRAIN_PATH   = 'train'
PREDICT_PATH = 'predict'
CONTROL_PATH = 'control'

MODES ={AUTO_PATH: drive_auto, TRAIN_PATH: drive_train,PREDICT_PATH:drive_predict,CONTROL_PATH:drive_control}
ACTUAL_MODE = AUTO_PATH
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

CSV_PATH = Path( Path.home() / ".keras" / "datasets" )
# if sys.platform == "win32":
#     CSV_PATH = "C:\\Users\\dmitr_000\\.keras\\datasets\\Imbalance_data.csv"
#     CSV_PATH = os.path.join(os.environ['USERPROFILE'])
# elif sys.platform == "linux":
#     CSV_PATH = "/home/dmitryv/.keras/datasets/Imbalance_data.csv"
# else:
#     CSV_PATH = "C:\\Users\\dmitr_000\\.keras\\datasets\\Imbalance_data.csv"
# Header names and discrtization in minutes
DT_DSET      = "Date Time"
RCPOWER_DSET = "Real_demand" #"Imbalance"

RCPOWER_DSET_AUTO = 'Real demand'

DISCRET      = 10

# The time cutoffs for the formation of the validation  and test sequence in the format of the parameter passed
# to the timedelta() like as 'days=<value>' or 'hours=<value>' or 'minutes=<value>'
# We use 'minute' resolution
TEST_CUT_OFF = 6 * DISCRET
VAL_CUT_OFF  = 500 * DISCRET   # 8 hours





# Logging & Charting

LOG_FILE_NAME = RCPOWER_DSET
# Matplotlib.pyplot is used for charting
STOP_ON_CHART_SHOW=False
#Models, descriptor, datasets( for auto_plane)  repository
if sys.platform == 'win32':
    PATH_REPOSITORY            = Path( Path( Path(getcwd()).drive) / '/' / "model_Repository")
    PATH_DATASET_REPOSITORY    = Path(Path( Path(getcwd()).drive) / '/' / "dataset_Repository")
    PATH_DESCRIPTOR_REPOSITORY = Path(Path( Path(getcwd()).drive) / '/' / "descriptor_Repository")
elif sys.platform == 'linux':
    PATH_REPOSITORY            = Path( Path.home() / "model_Repository" )
    PATH_DATASET_REPOSITORY    = Path( Path.home() / "dataset_Repository" )
    PATH_DESCRIPTOR_REPOSITORY = Path( Path.home() / "descriptor_Repository" )
else:
    PATH_REPOSITORY            = Path( Path( Path(getcwd()).drive) / '/' / "model_Repository")
    PATH_DATASET_REPOSITORY    = Path(Path( Path(getcwd()).drive) / '/' / "dataset_Repository")
    PATH_DESCRIPTOR_REPOSITORY = Path(Path( Path(getcwd()).drive) / '/' / "descriptor_Repository")

    print('{} \n{} \n{}'.format(PATH_REPOSITORY,PATH_DATASET_REPOSITORY,PATH_DESCRIPTOR_REPOSITORY))
    Path.mkdir(PATH_DATASET_REPOSITORY,    parents=True, exist_ok=True)
    Path.mkdir(PATH_DESCRIPTOR_REPOSITORY, parents=True, exist_ok=True)

ALL_MODELS = {'MLP':[(0, "mlp_1"), (1,"mlp_2")], 'CNN':[(2,'univar_cnn')],\
              'LSTM':[(3,'vanilla_lstm'),(4,'stacked_lstm'), (5,'bidir_lstm')],\
              'tsARIMA':[(6,'seasonal_arima'),(7,'best_arima')]}





# ALL_MODELS = {'tsARIMA':[(6,'seasonal_arima'),(7,'best_arima')]}

#training model
EPOCHS     = 100
N_STEPS    = 64
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

#ARIMA
SEASONALY_PERIOD = 6  # hour season , 144 for daily season
PREDICT_LAG      = 20
MAX_P            = 3
MAX_Q            = 3
MAX_D            = 2

""" Samples/ Time Series """
TS_DURATION_DAYS = 7 # days
SEGMENT_SIZE     = 1024

"""AUTO_PATH """
FILE_DESCRIPTOR = "peninsular.pickle"
SCALED_DATA_FOR_AUTO = False
START_DATE_FOR_AUTO  = "2020-08-01 00:00:00"
END_DATE_FOR_AUTO    = "2020-08-31 00:00:00"
TIME_TRUNC_FOR_AUTO  = 'hour'
GEO_LIMIT_FOR_AUTO   = None
GEO_IDS_FOR_AUTO     = None

""" set imbalance mode"""
MODE_IMBALANCE   = 1
IMBALANCE_NAME  = "ProgrammedMinusReal"
PROGRAMMED_NAME = "Programmed_demand"
DEMAND_NAME     = "Real_demand"





