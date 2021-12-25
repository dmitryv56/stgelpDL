#!/usr/bin/env python3

""" Short-time predictor for loading green energy on base Deep Learning (service).

 """

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import time
from pathlib import Path
import argparse

from msrvcpred.app.clients.client_Predictor import PredictorClient
from msrvcpred.cfg import MAX_LOG_SIZE_BYTES, BACKUP_COUNT , \
    PATH_LOG_FOLDER, PATH_MAIN_LOG, PATH_REPOSITORY, PATH_DATASET_REPOSITORY, \
    PATH_DESCRIPTOR_REPOSITORY

from msrvcpred.cfg import ALL_MODELS #TODO redesign MODES should in drive , not in cfg
from predictor.stgelpDL import drive_STGELPDL
from predictor.api import prepareDataset
from predictor.control import ControlPlane
from predictor.dataset import Dataset
from predictor.utility import msg2log, exec_time, PlotPrintManager, OutVerbose, isCLcsvExists
from ms_util import cli_parser, createControlPlane, destroyControlPlane



""" Set main system log for all modules exclude cfg.py.
SERVER_LOG is used in server.service module"""

# LOG_FOLDER=Path(Path(__file__).parent.parent.absolute()/LOG_FOLDER_NAME)
# LOG_FOLDER.mkdir(parents=True,exist_ok=True)
# MAIN_LOG=Path(LOG_FOLDER/MAIN_SYSTEM_LOG).with_suffix(".log")
size_handler=RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes =int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT) )
logger=logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)


logger.info('{} \n{} \n{}'.format(PATH_LOG_FOLDER, PATH_MAIN_LOG, PATH_REPOSITORY, PATH_DATASET_REPOSITORY,
                                  PATH_DESCRIPTOR_REPOSITORY))



def main(argc,argv):
    args = cli_parser()
    logger.info(args)

    cp = createControlPlane(args)
    cp.magic_app_number=222

    drive_STGELPDL(cp)

    destroyControlPlane(cp)

    # logger.info(args)
    # currs_client = PredictorClient()
    # # logger.info(currs_client.get_data("Predict data"))
    #
    # while True:
    #     # run()
    #     # time.sleep(1)
    #     logger.info("Train data\n\n\n\n")
    #     time.sleep(2)
    #     currs_client.get_streaming_data("Train data")
    #     time.sleep(5)
    #     logger.info("Predict data\n\n\n\n")
    #     for i in range(5):
    #         logger.info(currs_client.get_data("Predict data"))
    #         time.sleep(2)
    #
    #
    # pass

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
    pass