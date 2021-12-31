#!/usr/bin/env python3
""" GUI for predictor """

import sys
import logging
from logging.handlers import RotatingFileHandler

from msrvcpred.cfg import MAX_LOG_SIZE_BYTES, BACKUP_COUNT, PATH_LOG_FOLDER, PATH_CHART_LOG
from chart_api import PlotChart

from app.clients.client_Predictor import PredictorClient

size_handler = RotatingFileHandler(PATH_CHART_LOG, mode='a', maxBytes=int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)
logger.info('Logs: {} \nChart log: {} '.format(PATH_LOG_FOLDER, PATH_CHART_LOG))


def main():
    pch = PlotChart()
    logger.info("{} initialized.".format(pch.__class__.__name__))
    try:
        pch.run()
    except KeyboardInterrupt:
        logger.info("Chart client finished")

    except Exception as ex:
        logger.error("O-oops!!! {} exception. Chart client finished".format(ex))

    finally:
        sys.exit(0)
if __name__ == "__main__":
    main()