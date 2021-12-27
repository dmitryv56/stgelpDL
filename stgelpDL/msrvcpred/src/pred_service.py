#!/usr/bin/env python3

""" Short-time predictor for loading green energy on base Deep Learning (service).

 """

import sys
import logging
from logging.handlers import RotatingFileHandler

from msrvcpred.cfg import MAX_LOG_SIZE_BYTES, BACKUP_COUNT, PATH_LOG_FOLDER, PATH_MAIN_LOG, PATH_REPOSITORY, \
    PATH_DATASET_REPOSITORY, PATH_DESCRIPTOR_REPOSITORY
from predictor.stgelpDL import drive_STGELPDL
from ms_util import cli_parser, createControlPlane, destroyControlPlane


""" Set main system log for all modules exclude cfg.py.
SERVER_LOG is used in server.service module"""

size_handler = RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes=int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)
logger.info('{} \n{} \n{}'.format(PATH_LOG_FOLDER, PATH_MAIN_LOG, PATH_REPOSITORY, PATH_DATASET_REPOSITORY,
                                  PATH_DESCRIPTOR_REPOSITORY))


def main(argc, argv):
    args = cli_parser()
    logger.info(args)

    cp = createControlPlane(args)
    cp.magic_app_number = 222

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
