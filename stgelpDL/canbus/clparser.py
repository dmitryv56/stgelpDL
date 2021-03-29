#!/usr/bin/python3

import argparse
import os
import sys
from datetime import datetime,timedelta
from pathlib import Path

__version__ = '0.0.1'



def parser()->(argparse.ArgumentParser, str):

 # command-line parser
    sDescriptor         = "Digitat Twin Canbus-FD for exploring wavefor signal anomaly in physical layer. ICSim is " + \
 "used for a packet flow on transfer layer."


    smodeHelp          = " Supervised model anomaly detection (train path) or using trained models for anomaly " + \
                        "detection (test path). The 'debug' path comprises both 'trai' and 'test' paths, the dump is" +\
                        " splitted by set cut points."
    smethodHelp         = " 'BF' - Bloom Filter for fast anomaly detection. 'DTWIN' -a digital twin on the base of " + \
                         " neuron model of the signal waveforms."
    sICSimDumpHelp      = "Absolute path to a ICSim dump file. "
    sTrainSizeHelp      = "Train chunk size in lines of the dump file. Only for 'debug', 1024 by default."
    sTestSizeHelp       = "Test chunk size following by train chunk. Only for 'debug', 512 by default."
    sFsampleHelp        = "Sampling frequence, MHz. The default is 16 MHz."
    sBitrateHelp        = " Bitrate, KHz. The default is 256 KHz."
    sSNRHelp            = "Signal-Noise-Ratio, DB. The default is 40 DB."
    sSlopeHelp          = "Slope of waveform. The defaukt is 0.3."
    sChunkHelp          = "Chunk size for ICSim dump processing. The default is 32."
    sBFsizeHelp         = "Bloom Filter array size, 2048 by default."
    sBFprobHelp         = "Bloom Filter false positive probability, 2048 by default."
    sBatchSizeHelp      = "Batch size is a hyper-parameter of Neural Net training, 32 by default."
    sEpochsHelp         = "Epochs is a hyper-parameter of Neural Net training, 20 by default."
    stitleHelp          = "Title, one word using as log folder name."


    parser = argparse.ArgumentParser(description=sDescriptor)


    parser.add_argument('-m', '--mode', dest='cl_mode', action='store', default='train', choices=['train', 'test', \
                        'debug'], help=smodeHelp)
    parser.add_argument('-M', '--method', dest='cl_method', action='store', default='BF',
                        choices=['BF', 'DTWIN'], help=smethodHelp)
    parser.add_argument('-S', '--train_size',dest='cl_trainsize', action='store',default='1024', help=sTrainSizeHelp)
    parser.add_argument('-s', '--test_size', dest='cl_testsize', action='store', default='512', help=sTestSizeHelp)
    parser.add_argument('-d', '--icsim_dump',dest='cl_icsimdump', action='store',  help=sICSimDumpHelp)
    parser.add_argument('-f', '--fsample',   dest='cl_fsample',   action='store', default='16',   help=sFsampleHelp)
    parser.add_argument('-b', '--bitrate',   dest='cl_bitrate',   action='store', default='256',  help=sBitrateHelp)
    parser.add_argument('-R', '--SNR',       dest='cl_snr',       action='store', default='40',   help=sSNRHelp)
    parser.add_argument('-p', '--slope',     dest='cl_slope',     action='store', default='0.3',  help=sSlopeHelp)
    parser.add_argument('-c', '--chunk',     dest='cl_chunk',     action='store', default='32',   help=sChunkHelp)
    parser.add_argument(      '--BFsize',    dest='cl_bfsize',    action='store', default='2048', help=sBFsizeHelp)
    parser.add_argument(      '--BFprob',    dest='cl_bfprob',    action='store', default='0.05', help=sBFprobHelp)
    parser.add_argument('-z', '--batch_size',dest='cl_batch_size',action='store', default='32',   help=sBatchSizeHelp)
    parser.add_argument('-e', '--epochs',    dest='cl_epochs',    action='store', default='20',   help=sEpochsHelp)
    parser.add_argument('-t', '--title',     dest='cl_title',     action='store', default='Canbus_FD', help=stitleHelp)
    parser.add_argument('-v', '--verbose',   dest='cl_verbose',   action='count', default=0)
    parser.add_argument('--version',         action='version',    version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()
    # command-line parser
    arglist=sys.argv[0:]
    argstr=""
    for arg in arglist:
        argstr+=" " + arg
    message0=f"""  Command-line arguments

{argstr}
Title:                             : {args.cl_title}
Mode:                              : {args.cl_mode}
Method:                            : {args.cl_method}
ICSim dump file path               : {args.cl_icsimdump}
Train chunk size, in lines         : {args.cl_trainsize},  only for 'debug' mode
Test chunk size, in lines          : {args.cl_testsize},  only fot 'debug' mode
Sampling frequency                 : {args.cl_fsample} MHz
Bitrate                            : {args.cl_bitrate} KHz
Signal-Noise-Ratio (SNR)           : {args.cl_snr} DB
Waveform slope factor              : {args.cl_slope}  
Chunk for dump reading             : {args.cl_chunk}
Bloom filter bit array size        : {args.cl_bfsize}, only for 'BF' method
Bloom filter false positive prob.  : {args.cl_bfprob}, only for 'BF' method
Batch size for Neural Net training : {args.cl_batch_size}, only for 'DTWIN' method
Epochs for Neural Net training     : {args.cl_epochs}, only for 'DTWIN' method
"""

    return args, message0

def strParam(param:tuple)->str:
    (title,    mode,    method ,    n_train ,    n_test ,    canbusdump ,    fsample ,    bitrate ,    snr ,\
    slope ,    chunk_size ,    filter_size ,    fp_prob ,    batch_size, epochs,  folder_for_logging,
     d_repositories ) =param

    s="".join("{}:{}".format(key,val) for key,val in d_repositories.items())

    message= f"""
Title:                             : {title}
Mode:                              : {mode}
Method:                            : {method}
ICSim dump file path               : {canbusdump}
Train chunk size, in lines         : {n_train},  only for 'debug' mode.
Test chunk size, in lines          : {n_test},  only fot 'debug' mode.
Sampling frequency                 : {fsample} MHz.
Bitrate                            : {bitrate} KHz.
Signal-Noise-Ratio (SNR)           : {snr} DB
Waveform slope factor              : {slope}  
Chunk for dump reading             : {chunk_size}
Bloom filter bit array size        : {filter_size}, only for 'BF' method.
Bloom filter false positive prob.  : {fp_prob}, only for 'BF' method.
Batch size for Neural Net training : {batch_size}, only fo 'DTWIN' method.
epochs for Neural Net training     : {epochs}, only fo 'DTWIN' method.
Folder for logging                 : {folder_for_logging}
Repositories                       :
{s}
"""
# Digital twin repository            : {folder_dtwin}
# Bloom filter repository            : {folder_bf}           :
# """
    return message

""" Probability analysis of the the frequency of the appearance of the packets at the transport level CanBus-FD """

def parserPrAnFapTL()->(argparse.ArgumentParser, str):

 # command-line parser
    sDescriptor         = "Digital Twin Can Bus-FD for probability analysis of the the frequency of the appearance " + \
                          "of the packets at the transport level. ICSim is used for a packet flow generation. "


    smodeHelp          = " Supervised model anomaly detection (train path) or using trained models for anomaly " + \
                        "detection (test path). The 'debug' path comprises both 'train' and 'test' paths, the dump is" +\
                        " split by cut point."
    smethodHelp         = " 'ppd' - poisson probability distribution  for fast anomaly detection. By default is 'ppd'."
    sICSimDumpHelp      = "Absolute path to a ICSim dump file. "
    sMatchedKeyHelp     = "'ID', 'data' or 'packet', i.e.'ID||data' fields from CANbus packet are may use for " + \
                          " packet selection. By default, it is 'ID'. "

    sTrainSizeHelp      = "Train chunk size in lines of the dump file. Only for 'debug', 1024 by default."
    sTestSizeHelp       = "Test chunk size following by train chunk. Only for 'debug', 512 by default."
    # sFsampleHelp        = "Sampling frequence, MHz. The default is 16 MHz."
    # sBitrateHelp        = " Bitrate, KHz. The default is 256 KHz."
    # sSNRHelp            = "Signal-Noise-Ratio, DB. The default is 40 DB."
    # sSlopeHelp          = "Slope of waveform. The default is 0.3."
    sChunkHelp          = "Chunk size for ICSim dump processing. The default is 32."
    sBFsizeHelp         = "Bloom Filter array size, 2048 by default."
    sBFprobHelp         = "Bloom Filter false positive probability, 2048 by default."
    # sBatchSizeHelp      = "Batch size is a hyper-parameter of Neural Net training, 32 by default."
    # sEpochsHelp         = "Epochs is a hyper-parameter of Neural Net training, 20 by default."
    stitleHelp          = "Title, one word using as log folder name."


    parser = argparse.ArgumentParser(description=sDescriptor)


    parser.add_argument('-m', '--mode', dest='cl_mode', action='store', default='train', choices=['train', 'test', \
                        'debug'], help=smodeHelp)
    parser.add_argument('-M', '--method', dest='cl_method', action='store', default='ppd',
                        choices=['ppd', 'other'], help=smethodHelp)
    parser.add_argument('-S', '--train_size',dest='cl_trainsize', action='store',default='1024', help=sTrainSizeHelp)
    parser.add_argument('-s', '--test_size', dest='cl_testsize', action='store', default='512', help=sTestSizeHelp)
    parser.add_argument('-d', '--icsim_dump',dest='cl_icsimdump', action='store',  help=sICSimDumpHelp)
    parser.add_argument('-k', '--matched_key', dest='cl_match_key', action='store', choices=['ID', 'Packet','Data'], \
                        default='ID',help=sMatchedKeyHelp)
    # parser.add_argument('-f', '--fsample',   dest='cl_fsample',   action='store', default='16',   help=sFsampleHelp)
    # parser.add_argument('-b', '--bitrate',   dest='cl_bitrate',   action='store', default='256',  help=sBitrateHelp)
    # parser.add_argument('-R', '--SNR',       dest='cl_snr',       action='store', default='40',   help=sSNRHelp)
    # parser.add_argument('-p', '--slope',     dest='cl_slope',     action='store', default='0.3',  help=sSlopeHelp)
    parser.add_argument('-c', '--chunk',     dest='cl_chunk',     action='store', default='32',   help=sChunkHelp)
    parser.add_argument(      '--BFsize',    dest='cl_bfsize',    action='store', default='2048', help=sBFsizeHelp)
    parser.add_argument(      '--BFprob',    dest='cl_bfprob',    action='store', default='0.05', help=sBFprobHelp)
    # parser.add_argument('-z', '--batch_size',dest='cl_batch_size',action='store', default='32',   help=sBatchSizeHelp)
    # parser.add_argument('-e', '--epochs',    dest='cl_epochs',    action='store', default='20',   help=sEpochsHelp)
    parser.add_argument('-t', '--title',     dest='cl_title',     action='store', default='Canbus_FD', help=stitleHelp)
    parser.add_argument('-v', '--verbose',   dest='cl_verbose',   action='count', default=0)
    parser.add_argument('--version',         action='version',    version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()
    # command-line parser
    arglist=sys.argv[0:]
    argstr=""
    for arg in arglist:
        argstr+=" " + arg
    message0=f"""  Command-line arguments

{argstr}
Title:                             : {args.cl_title}
Mode:                              : {args.cl_mode}
Method:                            : {args.cl_method}
ICSim dump file path               : {args.cl_icsimdump}
Matched key                        : {args.cl_match_key}
Train chunk size, in lines         : {args.cl_trainsize},  only for 'debug' mode
Test chunk size, in lines          : {args.cl_testsize},  only fot 'debug' mode
Chunk for dump reading             : {args.cl_chunk}
Bloom filter bit array size        : {args.cl_bfsize}, only for 'BF' method
Bloom filter false positive prob.  : {args.cl_bfprob}, only for 'BF' method

"""

    return args, message0

def strParamPrAnFapTL(param:tuple)->str:

    (title, mode, method, n_train, n_test, canbusdump, matched_key, chunk_size, filter_size, fp_prob,
     folder_for_logging, folder_repository) =param


    message= f"""
Title:                             : {title}
Mode:                              : {mode}
Method:                            : {method}
ICSim dump file path               : {canbusdump}
Matched Key                        : {matched_key}
Train chunk size, in lines         : {n_train},  only for 'debug' mode.
Test chunk size, in lines          : {n_test},  only fot 'debug' mode.
Chunk for dump reading             : {chunk_size}
Bloom filter bit array size        : {filter_size}, only for 'BF' method.
Bloom filter false positive prob.  : {fp_prob}, only for 'BF' method.

Folder for logging                 : {folder_for_logging}
Repository                         : {folder_repository}

"""

    return message


