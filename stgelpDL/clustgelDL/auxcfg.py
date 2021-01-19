#!/usr/bin/python3

import os
from datetime import timedelta
from pathlib import Path
from time import perf_counter, sleep
# log
# 'plot' item in this dictionary is specific. It points on the folder not holds a file handler.
# The functions below listLogSet, closeLogs, log2All process 'plot'-item as specific.
D_LOGS={"clargs":None,"timeexec":None, "main":None, "block":None,"except":None,"cluster":None, "nnmodel":None,
        "control":None,"train":None,"predict":None,"plot":None}


def listLogSet(folder:str=None):
    path_folder=None
    if folder is None:
        path_folder=Path("Logs")
    else:
        path_folder = Path(folder)
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    suffics=".log"
    for log_name, handler in D_LOGS.items():
        log_name_path=Path(path_folder/log_name).with_suffix(suffics)
        if log_name == "plot":
            Path(path_folder/log_name).mkdir(parents=True, exist_ok=True)
            D_LOGS[log_name] = str(Path(path_folder/log_name))
            continue
        f=open(str(log_name_path),'w+')
        if f is not None:
            D_LOGS[log_name]=f
    return

def closeLogs():
    for log_name, handler in D_LOGS.items():
        if log_name=="plot":
            continue
        if handler is not None:
            handler.close()
            handler=None
    return
# Puts a message in all log files and flushes. If no message , only flushes log files
def log2All(msg:str=None):

    for log_name, handler in D_LOGS.items():
        if log_name == "plot":
            continue
        if handler is not None:
            if msg is not None and len(msg)>0:
                handler.write("{}\n".format(msg))
            else:
                handler.flush()
    return

def logList():
    msg = ""
    for log_name, handler in D_LOGS.items():

        if log_name == "plot":
            continue
        if handler is not None:
            msg=msg + "{}: {}\n".format(log_name, os.path.realpath(handler.name))

    return msg

"""
Decorator exec_time
"""

def exec_time(function):
    def timed(*args, **kw):
        time_start = perf_counter()
        ret_value = function(*args, **kw)
        time_end = perf_counter()

        execution_time = time_end - time_start
        arguments = ", ".join([str(arg) for arg in args] + ["{}={}".format(k, kw[k]) for k in kw])
        smsg = "  {:.2f} sec  for {}({})\n".format(execution_time, function.__name__, arguments)

        D_LOGS["timeexec"].write(smsg)
        D_LOGS["timeexec"].flush()

        return ret_value

    return timed

