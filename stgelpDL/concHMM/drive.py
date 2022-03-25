#!/usr/bin/env python3

""" Concurent HMM predictors"""


import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from conc_hmm import HMM

current_dir = Path(__file__).parent

MAX_LOG_SIZE_BYTES=1024*1024
BACKUP_COUNT = 2

PATH_LOG_FOLDER = Path(current_dir / "Logs")
PATH_LOG_FOLDER.mkdir(parents=True, exist_ok=True)
PATH_MAIN_LOG = Path(PATH_LOG_FOLDER/"main_HMM.log")
PATH_DATASET_REPOSITORY =""
PATH_REPOSITORY = ""
PATH_DESCRIPTOR_REPOSITORY = ""


size_handler = RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes=int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)
logger.info('Logs: {} \nMain log: {} \nRepository: {} \nDatasets:{} \nDescriptors{}'.format(PATH_LOG_FOLDER,
            PATH_MAIN_LOG, PATH_REPOSITORY, PATH_DATASET_REPOSITORY, PATH_DESCRIPTOR_REPOSITORY))



async def foo(hmm:HMM=None):
    print('Running in foo')
    if hmm is not None:
        hmm.__str__()
    await asyncio.sleep(0)
    print('Explicit context switch to foo again')


async def bar(hmm:HMM=None):
    print('Explicit context to bar')
    if hmm is not None:
        hmm.__str__()
    await asyncio.sleep(0)
    print('Implicit context switch back to bar')


def main():
    hmm2=HMM(states=['S0','S1'],pi=[0.1,0.9])
    hmm3 = HMM(states=['S0', 'S1','S2'], pi=[0.1, 0.8,0.1])
    ioloop = asyncio.get_event_loop()
    tasks =[ioloop.create_task(foo(hmm2)), ioloop.create_task(bar(hmm3))]
    wait_tasks = asyncio.wait(tasks)
    ioloop.run_until_complete(wait_tasks)
    ioloop.close()

if __name__ == "__main__":
    main()
    pass