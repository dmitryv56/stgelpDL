#!/usr/bin/python3

""" 'idspaccl' - there is a digital twin of CANbus-FD protocol, it performs for intrusion detection in the packets
at the transport level of Can Bus-FD. The Neural Net is used for packet classification of two groups  valid and
intrused packets.
    At train stage, the packets are classified by given matched key and the probability of the class appearance is
estimated. The given matched key is one of packet fields 'ID', 'Data' or concatenation 'ID'+'Data' ('Packet').
The class probability estimation and a histogram of the delay intervals between packets being belonging to to the same
class are saved in Database (pandas DataFrame dataset).
    At the test stage, anomaly packed are detected. The checked hypothesis is packet appearances describes by poisson
probability distributions  and delay intervals describes by exponential distribution. For all classes the parameters of
distributions already estimated .

"""

from os import path
import sys
from datetime import datetime,timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.stats import poisson,ks_2samp


from predictor.utility import msg2log
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from canbus.api import readChunkFromCanBusDump,file_len,dict2csv,dict2pkl, pkl2dict,mleexp,manageDB,DB_NAME,fillQuery,\
fillQuery2Test,getSerializedModelData,KL_decision
from canbus.clparser import parserIDS,strParamIDS
from canbus.func import dsetIDSpacCl,trainIDSpacCl,detectIDSpacCl



def main(argc,argv):
    nret=0

    canbusdump = "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
    canbusdump = "/home/dmitryv/ICSim/ICSim/candump-2021-02-11_100040.log"

    repository=""
    dset_name=""
    title =""
    chunk_size =256

    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")  # using in folder/file names
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    argparse, message0 = parserIDS()
    """ command-line paremeters to local variables """
    title = argparse.cl_title
    mode = argparse.cl_mode
    method = argparse.cl_method
    n_train = int(argparse.cl_trainsize)
    n_test = int(argparse.cl_testsize)
    chunk_size = int(argparse.cl_chunk)
    l_canbusdump = list(argparse.cl_icsimdump.split(','))

    filter_size = int(argparse.cl_bfsize)
    fp_prob = float(argparse.cl_bfprob)


    title1 = "{}_{}_{}".format(title.replace(' ', '_'), mode, method)

    """ create log and repository folders """
    dir_path = path.dirname(path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(title1, date_time)
    folder_for_logging.mkdir(parents=True, exist_ok=True)
    folder_repository_data = Path(Path(dir_path) / Path("Repository") / Path(method)/Path('data'))
    folder_repository_data.mkdir(parents=True, exist_ok=True)
    folder_repository_model = Path(Path(dir_path) / Path("Repository") / Path(method) / Path('model'))
    folder_repository_model.mkdir(parents=True, exist_ok=True)
    # folder_ppd = Path(folder_repository / title1 / method)
    # folder_ppd.mkdir(parents=True, exist_ok=True)
    feature_type='prb'
    exclude_ID=['01A4,01AA']
    param = (title, mode, method, n_train, n_test, l_canbusdump, chunk_size, feature_type, exclude_ID, filter_size,
             fp_prob,  folder_for_logging, folder_repository_data,folder_repository_model)
    message2 = strParamIDS(param)

    """ init logs """
    listLogSet(str(folder_for_logging))  # A logs are creating

    msg2log(None, message0, D_LOGS['clargs'])
    msg2log(None, message1, D_LOGS['timeexec'])
    msg2log(None, message2, D_LOGS['main'])
    for canbusdump in l_canbusdump:
        n_samples = file_len(canbusdump)
        s_samples = n_samples if n_samples > 0 else '0'
        msg2log(None, "\n{} lines in {}\n\n".format(s_samples, canbusdump),D_LOGS['control'])

    subtitle = "{}".format(mode)

    if mode == "dset" or mode == "debug":
        dset_dict = dsetIDSpacCl(l_canbusdump=l_canbusdump, feature_type = feature_type, chunk_size=chunk_size,
                    title=title, repository =str(folder_repository_model), dset_name= str(folder_repository_data),
                    f = D_LOGS['control'])

    elif mode=="train" or mode=="debug":
        model_dict = trainIDSpacCl(feature_type = feature_type, repository= str(folder_repository_model),
                    chunk_size = chunk_size, exclude_ID= exclude_ID, dset_name= str(folder_repository_data),
                    title=title, f=D_LOGS['predict'])

    elif mode=="detect" or mode=="debug":
        model_dict = detectIDSpacCl(feature_type=feature_type, repository=str(folder_repository_model),
                                   chunk_size=chunk_size, exclude_ID=exclude_ID, dset_name=str(folder_repository_data),
                                   title=title, f=D_LOGS['predict'])
    else:
        pass

    return nret



if __name__ =="__main__":
    pass
    nret =main(len(sys.argv),sys.argv)

    sys.exit(nret)