#!/usr/bin/python3

""" 'franpatl' - frequency analysis of packet appearances at the transport level of Can Bus-FD.

"""

from os import path
import sys
from datetime import datetime
from pathlib import Path

from predictor.utility import msg2log
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs
from canbus.api import file_len
from canbus.clparser import parserPrAnFapTL,strParamPrAnFapTL
from canbus.func import trainFreqModel,testFreqModel





"""
canbusdump = "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
match_key = 'ID'  # 'ID',"Packet','Data'
"""
def main(arc,argv):
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")  # using in folder/file names
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    argparse, message0 = parserPrAnFapTL()
    """ command-line paremeters to local variables """
    title = argparse.cl_title
    mode = argparse.cl_mode
    method = argparse.cl_method
    n_train = int(argparse.cl_trainsize)
    n_test = int(argparse.cl_testsize)
    chunk_size=int(argparse.cl_chunk)
    canbusdump = argparse.cl_icsimdump

    filter_size = int(argparse.cl_bfsize)
    fp_prob = float(argparse.cl_bfprob)
    match_key=argparse.cl_match_key

    title1 = "{}_{}_{}_key_is_{}".format(title.replace(' ', '_'), mode, method, match_key)

    """ create log and repository folders """
    dir_path = path.dirname(path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(title1, date_time)
    folder_for_logging.mkdir(parents=True, exist_ok=True)
    folder_repository = Path(Path(dir_path) / Path("Repository") / Path(method))
    folder_repository.mkdir(parents=True, exist_ok=True)
    # folder_ppd = Path(folder_repository / title1 / method)
    # folder_ppd.mkdir(parents=True, exist_ok=True)


    param = (title, mode, method, n_train, n_test, canbusdump, match_key, chunk_size, filter_size, fp_prob,
             folder_for_logging, folder_repository)
    message2 = strParamPrAnFapTL(param)

    """ init logs """
    listLogSet(str(folder_for_logging))  # A logs are creating

    msg2log(None, message0, D_LOGS['clargs'])
    msg2log(None, message1, D_LOGS['timeexec'])
    msg2log(None, message2, D_LOGS['main'])
    n_samples=file_len(canbusdump)
    s_samples=n_samples if n_samples>0 else '0'
    msg2log(None,"\n{} lines in {}\n\n".format(s_samples, canbusdump))
    subtitle = "{}".format(mode)
    dset_name=Path( Path(folder_repository) / Path("{}_{}".format("dataset",match_key))).with_suffix(".csv")
    if mode != 'test':

        trainFreqModel(canbusdump= canbusdump, match_key=match_key, repository=str(folder_repository), title= subtitle,
                   dset_name=str(dset_name), f=D_LOGS['train'])

    if mode != 'train':
        testFreqModel(canbusdump= canbusdump, match_key=match_key, repository=str(folder_repository),
                      dset_name=str(dset_name), title= subtitle, f=D_LOGS['predict'])



    message1 = "Time execution logging finished at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    msg2log(None, message1, D_LOGS['timeexec'])
    closeLogs()
    return 0


if __name__=="__main__":
    # offset_line = 4
    # chunk_size = 2
    # canbusdump = "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
    # ft = open("log.log", 'w+')
    # ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=chunk_size, canbusdump=canbusdump, f=ft)
    # time=float(ld[0]['DateTime'])
    # utc_time0 = datetime.utcfromtimestamp(time)
    # utc_time = datetime.fromtimestamp(time, timezone.utc)
    # local_time=utc_time.astimezone()
    # print(utc_time.strftime("%Y-%m-%d %H:%M:%S.%f+00:00 (UTC)"))
    # print(local_time.strftime("%Y-%m-%d %H:%M:%S.%f%z (%Z)"))
    # tst=datetime.fromtimestamp(math.floor(utc_time))
    nret = main(len(sys.argv),sys.argv)

    sys.exit(nret)