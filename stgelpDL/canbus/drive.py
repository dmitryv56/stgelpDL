#!/usr/bin/python3

from pathlib import Path
import re
import matplotlib.pyplot as plt
import math
import numpy as np

from clustgelDL.auxcfg  import D_LOGS,log2All,exec_time,logList
from predictor.utility import msg2log
from canbus.api import wfstreamGen,trstreamGen,SIG_,SIG_0,SIG_1, TR_DICT
from canbus.BF import BF

def train_path(canbusdump:str="", bf:BF=None, chunk_size:int =16, f:object=None ):

    offset_line = 0
    fsample = 16e+6
    bitrate = 125000.0
    slope   = 0.3
    snr     = 30


    prev_state= SIG_
    n_chunk=0
    while  offset_line>=0 and n_chunk<8:
        transition_stream = trstreamGen( canbusdump=canbusdump, offset_line =offset_line,
                                        chunk_size = chunk_size, prev_state=prev_state, f=D_LOGS['train'])
        if not transition_stream:
            break
        title = "train_chunk_{}_snr_{}".format(n_chunk, snr)
        wf_dict = wfstreamGen(mode= 'train', transition_stream=transition_stream, fsample=fsample, bitrate=bitrate,
                              slope=slope, snr=snr, trwf_d = TR_DICT, bf=bf, title=title, f=D_LOGS['train'])

        offset_line=offset_line + chunk_size
        n_chunk+=1
        msg2log(None, "Bloom filter state after {} chunk\n{}".format(n_chunk-1, bf.__str__()),D_LOGS['control'])
    return


def test_path(canbusdump:str="", bf:BF=None, chunk_size:int =16, f:object=None):
    offset_line = 0
    fsample = 16e+6
    bitrate = 125000.0
    slope = 0.1
    snr = 60

    prev_state = SIG_
    n_chunk = 0
    while offset_line >= 0 and n_chunk<10:
        transition_stream = trstreamGen(canbusdump=canbusdump, offset_line=offset_line,
                                        chunk_size=chunk_size, prev_state=prev_state, f=f)
        if not transition_stream:
            break
        title="test_chunk_{}_snr_{}".format(n_chunk,snr)
        wf_dict = wfstreamGen(mode='test', transition_stream=transition_stream, fsample=fsample, bitrate=bitrate,
                              slope=slope, snr=snr, trwf_d=TR_DICT, bf=bf, title=title, f=f)

        offset_line = offset_line + chunk_size
        n_chunk +=1
        if snr>3:
            snr=snr -5.0
    return