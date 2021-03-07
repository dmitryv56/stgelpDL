#!/usr/bin/python3

from pathlib import Path
import re
import matplotlib.pyplot as plt
import math
import numpy as np

from clustgelDL.auxcfg  import D_LOGS,log2All,exec_time,logList
from predictor.utility import msg2log
from canbus.api import wfstreamGen,trstreamGen,SIG_,SIG_0,SIG_1, TR_DICT,transitionsPng,tr_names
from canbus.BF import BF
from canbus.digitaltwin import wfstreamGenSLD,digitalTwinTrain,DigitalTwinMLP,digitalTwinPredict

def train_path(method:str = 'BF', canbusdump:str="", bf:BF=None, chunk_size:int =16, fsample:float = 16e+6,
               bitrate:float = 125000.0, slope:float = 0.3, snr:float = 40.0, repository:dict= {}, hyperparam:tuple=(),
               f:object=None ):


    transitionsPng(fsample=fsample,bitrate=bitrate,snr=snr,slope=slope,f=f)

    if method == "BF":
        repositoryBF=repository["BF"]

        train_pathBF(canbusdump=canbusdump, bf=bf, chunk_size=chunk_size, fsample=fsample, bitrate=bitrate, slope=slope,
                     snr=snr, repository=repositoryBF,f=D_LOGS['control'])
    elif method == "DTWIN":
        repositoryDTWIN = repository["DTWIN"]

        train_pathDTWIN(canbusdump=canbusdump, chunk_size=chunk_size, fsample=fsample, bitrate=bitrate, slope=slope,
                        snr=snr, repository=repositoryDTWIN, hyperparam =hyperparam, f=D_LOGS['train'])
    else:
        pass

    return


def train_pathBF(canbusdump:str="", bf:BF=None, chunk_size:int =16, fsample:float = 16e+6,
               bitrate:float = 125000.0, slope:float = 0.3, snr:float = 40.0, repository:str="", f:object=None):

    f_json_name = Path(repository) / Path("bf").with_suffix(".json")
    offset_line = 0

    prev_state= SIG_
    n_chunk=0
    while  offset_line>=0 and n_chunk<8:
        transition_stream = trstreamGen( canbusdump=canbusdump, offset_line =offset_line,
                                        chunk_size = chunk_size, prev_state=prev_state, f=D_LOGS['train'])
        if not transition_stream:
            break
        title = "train_chunk_{}_snr_{}".format(n_chunk, snr)

        wf_dict = wfstreamGen(mode= 'train', transition_stream=transition_stream, fsample=fsample, bitrate=bitrate,
                              slope=slope, snr=snr, trwf_d = TR_DICT, bf=bf, title=title, repository=str(f_json_name),
                              f=D_LOGS['train'])

        offset_line=offset_line + chunk_size
        n_chunk+=1
        msg2log(None, "Bloom filter state after {} chunk\n{}".format(n_chunk-1, bf.__str__()),D_LOGS['control'])
    return


def train_pathDTWIN(canbusdump: str = "", chunk_size: int = 16, fsample: float = 16e+6, bitrate: float = 125000.0,
                    slope: float = 0.3, snr: float = 40.0, repository:str="", hyperparam:tuple=(), f: object = None):

    offset_line = 0
    all_train_chunks=[]
    prev_state = SIG_
    n_chunk = 0
    train_chunk_start=0
    (batch_size,epochs)=hyperparam
    while offset_line >= 0 and n_chunk < 8:
        transition_stream = trstreamGen(canbusdump=canbusdump, offset_line=offset_line,
                                        chunk_size=chunk_size, prev_state=prev_state, f=D_LOGS['train'])
        if not transition_stream:
            break
        title = "train_chunk_{}_snr_{}".format(n_chunk, snr)

        chunk_list,input_size = wfstreamGenSLD(mode='train', transition_stream=transition_stream, source='wf', fsample=fsample,
                       bitrate=bitrate, slope=slope, snr=snr, trwf_d=TR_DICT, title=title, repository=repository,
                       start_chunk=train_chunk_start, dump_chunk_size = 512, f=D_LOGS['train'])
        all_train_chunks=all_train_chunks + chunk_list
        train_chunk_start=train_chunk_start +len(chunk_list)
        offset_line = offset_line + chunk_size
        n_chunk += 1
        # msg2log(None, "Bloom filter state after {} chunk\n{}".format(n_chunk - 1, bf.__str__()), D_LOGS['control'])
    n_classes=len(tr_names)
    hyperprm = {
        'normalize':True,
        'batch_size':batch_size,
        'epochs':epochs,
        'input_size': {input_size: None},
        'n_classes': {n_classes: None},
        'hidden_layers': [{32: 'relu'}, {32: 'relu'}]
    }
    digitalTwinTrain(model= DigitalTwinMLP, name="DigitalTwinMLP",  hyperprm=hyperprm, chunk_list = all_train_chunks,
                     repository= repository, f=D_LOGS['train'])
    return





def test_path(method:str='BF',canbusdump:str="", bf:BF=None, chunk_size:int =16, fsample:float = 16e+6,
               bitrate:float = 125000.0, slope:float = 0.3, snr:float = 40.0, repository:dict={}, f:object=None):
    offset_line = 0

    if method == "BF":
        repositoryBF = repository["BF"]

        test_pathBF(canbusdump=canbusdump, bf=bf, chunk_size=chunk_size, fsample=fsample, bitrate=bitrate, slope=slope,
                     snr=snr, repository=repositoryBF,f=D_LOGS['control'])
    elif method == "DTWIN":
        repositoryDTWIN = repository["DTWIN"]

        test_pathDTWIN(canbusdump=canbusdump, chunk_size=chunk_size, fsample=fsample, bitrate=bitrate, slope=slope,
                        snr=snr, repository=repositoryDTWIN,f=D_LOGS['train'])
    else:
        pass

    # prev_state = SIG_
    # n_chunk = 0
    # while offset_line >= 0 and n_chunk<10:
    #     transition_stream = trstreamGen(canbusdump=canbusdump, offset_line=offset_line,
    #                                     chunk_size=chunk_size, prev_state=prev_state, f=f)
    #     if not transition_stream:
    #         break
    #     title="test_chunk_{}_snr_{}".format(n_chunk,snr)
    #     wf_dict = wfstreamGen(mode='test', transition_stream=transition_stream, fsample=fsample, bitrate=bitrate,
    #                           slope=slope, snr=snr, trwf_d=TR_DICT, bf=bf, title=title, f=f)
    #
    #     offset_line = offset_line + chunk_size
    #     n_chunk +=1
    #     if snr>3:
    #         snr=snr -5.0
    return

def test_pathBF(canbusdump:str="", bf:BF=None, chunk_size: int = 16, fsample: float = 16e+6, bitrate: float = 125000.0,
                    slope: float = 0.3, snr: float = 40.0, repository:str="", f: object = None):
    offset_line = 0

    f_json_name = Path(repository) / Path("bf").with_suffix(".json")
    if  not f_json_name.is_file():
        message="{} does not exist. The testing is finished"
        msg2log(None, message,f)
        log2All(message)
        return

    prev_state = SIG_
    n_chunk = 0
    while offset_line >= 0 and n_chunk<10:
        transition_stream = trstreamGen(canbusdump=canbusdump, offset_line=offset_line,
                                        chunk_size=chunk_size, prev_state=prev_state, f=f)
        if not transition_stream:
            break
        title="test_chunk_{}_snr_{}".format(n_chunk,snr)
        wf_dict = wfstreamGen(mode='test', transition_stream=transition_stream, fsample=fsample, bitrate=bitrate,
                              slope=slope, snr=snr, trwf_d=TR_DICT, bf=bf, title=title, repository=str(f_json_name),
                              f=f)

        offset_line = offset_line + chunk_size
        n_chunk +=1
        if snr>3:
            snr=snr -5.0
    return


def test_pathDTWIN(canbusdump: str = "", chunk_size: int = 16, fsample: float = 16e+6, bitrate: float = 125000.0,
                slope: float = 0.3, snr: float = 40.0, repository: str = "", f: object = None):
    offset_line = 0

    prev_state = SIG_
    n_chunk = 0
    all_test_chunks=[]
    test_chunk_start=0
    while offset_line >= 0 and n_chunk < 10:
        transition_stream = trstreamGen(canbusdump=canbusdump, offset_line=offset_line,
                                        chunk_size=chunk_size, prev_state=prev_state, f=f)
        if not transition_stream:
            break
        title = "test_chunk_{}_snr_{}".format(n_chunk, snr)
        chunk_list, input_size = wfstreamGenSLD(mode='est', transition_stream=transition_stream, source='wf',
                                                fsample=fsample, bitrate=bitrate, slope=slope, snr=snr, trwf_d=TR_DICT,
                                                title=title, repository=repository,
                                                start_chunk=test_chunk_start, dump_chunk_size=512, f=D_LOGS['train'])
        all_test_chunks = all_test_chunks + chunk_list
        test_chunk_start = test_chunk_start + len(chunk_list)
        offset_line = offset_line + chunk_size
        n_chunk += 1
        if snr > 3:
            snr = snr - 5.0
    digitalTwinPredict(model= DigitalTwinMLP, name="DigitalTwinMLP",  chunk_list=all_test_chunks[-5:],
                       repository = repository, f=D_LOGS['predict'])
    return