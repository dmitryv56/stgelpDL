#!/usr/bin/python3

""" 'andephyl' is a program for anomaly detection in the virtual model (Digital Twin -DTWIN) of the Can Bus.
 This one comprises a several modules :
 api - common api ICSim dump parsing, waveform generations and etc.
 BF - Bloom filter implementation.
 clparser - command line parser.
 digitaltwin - neural model classes implementation.
 drive - simple state-machine for code flow control.
 """

from os import path
import sys
from datetime import datetime
from pathlib import Path

from canbus.drive import train_path,test_path
from canbus.BF import BF
from canbus.clparser import parser, strParam
from clustgelDL.auxcfg import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from predictor.utility import msg2log,PlotPrintManager

def main(argc,argv)->int:

    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")  # using in folder/file names
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    argparse,message0 = parser()
    """ command-line paremeters to local variables """
    title       = argparse.cl_title
    mode        = argparse.cl_mode
    method      = argparse.cl_method
    n_train     = int(argparse.cl_trainsize)
    n_test      = int(argparse.cl_testsize)
    canbusdump  = argparse.cl_icsimdump
    fsample     = round(float(argparse.cl_fsample) * 1000000.0, 3)
    bitrate     = round(float(argparse.cl_bitrate) * 1000.0,    3)
    snr         = round(float(argparse.cl_snr), 1)
    slope       = round(float(argparse.cl_slope), 2)
    chunk_size  = int(argparse.cl_chunk)
    filter_size = int(argparse.cl_bfsize)
    fp_prob     = float(argparse.cl_bfprob)
    batch_size  = int(argparse.cl_batch_size)
    epochs      = int(argparse.cl_epochs)
    title1="{}_{}_{}_{}".format(title.replace(' ','_'),mode,method,date_time)
    """ create log and repository folders """
    dir_path = path.dirname(path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(title1, date_time)
    folder_repository=Path(dir_path) / "Repository"
    folder_bf = folder_repository / title1 / "BF"
    Path(folder_bf).mkdir(parents=True, exist_ok=True)
    folder_dtwin = folder_repository / title1 / "DTWIN"
    Path(folder_dtwin).mkdir(parents=True, exist_ok=True)
    d_repositories={"BF":folder_bf, "DTWIN":folder_dtwin}

    param = (title, mode, method, n_train, n_test, canbusdump, fsample, bitrate, snr, slope, chunk_size, filter_size,
             fp_prob, batch_size, epochs, folder_for_logging, d_repositories)
    message2=strParam(param)
    hyperparam=(batch_size,epochs)
    """ init logs """
    listLogSet(str(folder_for_logging))  # A logs are creating

    msg2log(None,message0,  D_LOGS['clargs'])
    msg2log(None, message1, D_LOGS['timeexec'])
    msg2log(None, message2, D_LOGS['main'])

    # method="BF" # BF -Bloom Filter, DTWIN - digital twin
    # method = "DTWIN"
    # canbusdump ="/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
    # fsample= 16e+6
    # bitrate: float = 125000.0
    # slope: float = 0.3
    # snr:float = 40.0
    # chunk_size=32

    bf=None
    strHeader = "\n Digital twins (Neuron model) for anomaly detection  \n"
    if method == "BF":
        strHeader="\n Bloom filter fast detection  \n"
        bf = BF(filter_size=filter_size, fp_prob=fp_prob, f=D_LOGS['block'])
    log2All(strHeader)

    if mode == "train" or mode == "debug":
        strHeader = "\n Train path  \n"
        log2All(strHeader)

        train_path(method=method, canbusdump=canbusdump, bf=bf, chunk_size=chunk_size, fsample=fsample,
                bitrate=bitrate, slope=slope, snr=snr, repository=d_repositories, hyperparam=hyperparam,
                f=D_LOGS['control'])

    if mode == "test" or mode == "debug":
        strHeader = "\n Test path  \n"
        log2All(strHeader)
        test_path(method=method,canbusdump=canbusdump, bf=bf, chunk_size=chunk_size, fsample=fsample,
                bitrate=bitrate, slope=slope, snr=snr, repository=d_repositories,f=D_LOGS['control'])

    message1 = "Time execution logging finished at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    msg2log(None, message1, D_LOGS['timeexec'])
    closeLogs()
    return 0


if __name__=="__main__":
    nret =main(len(sys.argv),sys.argv)

    sys.exit(nret)