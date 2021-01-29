#!/usr/bin/python3

""" Short Term (Green) Energy Load Offline Predictor by using Deep Learning.

"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from stcgelDL.cfg import GlobalConst
from predictor.utility import msg2log,PlotPrintManager
from tsstan.pltStan import setPlot, plotAll
from offlinepred.drive import drive
from offlinepred.api import PATH_REPOSITORY




__version__="0.0.1"


def parser()->(argparse.ArgumentParser, str):

 # command-line parser
    sDescriptor         = "Short Term (green) Electricity Load Processes (Time Series -TS)  Offline Predictor " +  \
                          "by using Deep Learning."

    sCSVhelp            = "Absolute path to a source dataset (csv-file)."
    sEndogenTSnameHelp  = "The TS list. A string contains the comma-separated the column names of TS  in the dataset. "
    sTimeStampHelp      = "Time Stamp (column) name in the dataset."
    sn_stepHelp         = "The step period  for Supervised Learning data creating."
    sdiscretHelp        = "TS sampling discretization(minutes)"
    snum_predictsHelp   = "Predict period. The last part of time series is used for predict."


    sevalsizeHelp       = "Evaluation sequence size. The evaluation sequence followed by test sequence."
    stestsizeHelp       = "Test sequence size. The tail of TS is used for testing."
    stitleHelp          = "Title, one word using as log folder name."

    parser = argparse.ArgumentParser(description=sDescriptor)

    parser.add_argument('-m', '--mode', dest='cl_mode', action='store', default='offlp', choices=['offlp'],
                        help='Possible modes: 1)offline predict  2) reserved for future extensions')
    parser.add_argument('-c', '--csv_dataset',  dest='cl_dset',         action='store',            help=sCSVhelp)
    parser.add_argument('-t', '--endogen',      dest='cl_endots',       action='store', default='',
                        help=sEndogenTSnameHelp)
    parser.add_argument('--timestamp',          dest='cl_timestamp',    action='store', default='Date Time',
                        help=sTimeStampHelp)
    parser.add_argument('-n', '--n_step',       dest='cl_n_step',       action='store', default=32,  help=sn_stepHelp)
    parser.add_argument('-d', '--discret',      dest='cl_discret',      action='store', default=10, help=sdiscretHelp)
    parser.add_argument('-p', '--num_predicts', dest='cl_num_predicts', action='store', default=4,
                        help=snum_predictsHelp)
    parser.add_argument('-s', '--test',         dest='cl_test_size',     action='store', default=32, help=stestsizeHelp)
    parser.add_argument('-e', '--eval',         dest='cl_eval_size',     action='store', default=256,help=sevalsizeHelp)

    parser.add_argument('-i', '--title', dest='cl_title', action='store', default='ElHiero', help=stitleHelp)
    parser.add_argument('--verbose', '-v',      dest='cl_verbose',      action='count', default=0)
    parser.add_argument('--version',                                    action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()
    # command-line parser
    arglist=sys.argv[0:]
    argstr=""
    for arg in arglist:
        argstr+=" " + arg
    message0=f"""  Command-line arguments

{argstr}
Title:                             : {args.cl_title}
Dataset path                       : {args.cl_dset}
Mode                               : {args.cl_mode}
Time Series in dataset             : {args.cl_endots}
Timestamp in dataset               : {args.cl_timestamp} 
Discretization (min)               : {args.cl_discret}
Time Series Step period            : {args.cl_n_step}
Evaluation Sequence Size           : {args.cl_eval_size}
Test Sequence Size                 : {args.cl_test_size}
Predict period                     : {args.cl_num_predicts}

"""

    return args, message0

def logParams(param:tuple)->str:
    (title, mode, csv_source, dt_col_name, ts_list, discret, n, n_step, n_eval, n_test, n_pred, folder_for_logging, \
     folder_for_train_logging, folder_for_control_logging, folder_for_predict_logging) = param

    message0 = f"""  Task parameters
Title:                             : {title}
Dataset path                       : {csv_source}
Mode                               : {mode}
Time Series in dataset             : {ts_list}
Timestamp in dataset               : {dt_col_name} 
Discretization (min)               : {discret}
Time Series length                 : {n}
Time Series step period            : {n_step}
Evaluation sequence size           : {n_eval}
Test sequence size                 : {n_test}
Predict period                     : {n_pred}
Folder for logging                 : {folder_for_logging}
Folder for train logging           : {folder_for_train_logging}
Folder for control logging         : {folder_for_control_logging}
Folder for plots                   : {D_LOGS["plot"]}
Logs                               :
{logList()}
       
    """
    return message0



def main(argc,argv):
    args, message0 = parser()

    title       = args.cl_title
    mode        = args.cl_mode
    csv_source  = args.cl_dset
    dt_col_name = args.cl_timestamp  # "Date Time"
    ts_list     = list(args.cl_endots.split(','))
    discret     = int(args.cl_discret)
    n_step      = int(args.cl_n_step)
    n_eval      = int(args.cl_eval_size)
    n_test      = int(args.cl_test_size)
    n_pred      = int(args.cl_num_predicts)

    GlobalConst.setVerbose(args.cl_verbose)
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(title, date_time)

    listLogSet(str(folder_for_logging))  # A logs are creating
    msg2log(None, message0, D_LOGS["clargs"])
    msg2log(None, message1, D_LOGS["timeexec"])
    fc = D_LOGS["control"]
    fp = D_LOGS["predict"]
    ft = D_LOGS["train"]
    folder_for_train_logging = str(Path(os.path.realpath(ft.name)).parent)
    folder_for_control_logging = str(Path(os.path.realpath(fc.name)).parent)
    folder_for_predict_logging = str(Path(os.path.realpath(fp.name)).parent)

    PlotPrintManager.set_Logfolders(folder_for_control_logging, folder_for_predict_logging)
    PlotPrintManager.set_Logfolders(D_LOGS['plot'], D_LOGS['plot'])
    if not Path(PATH_REPOSITORY).is_dir():
        Path(PATH_REPOSITORY).mkdir(parents=True, exist_ok=True)
    msg=""
    try:
        df = pd.read_csv(csv_source)
        n = len(df)
    except:
        n=0
        msg = "O-o-ops! I got an unexpected error - reason  {}\n".format(sys.exc_info())
    finally:
        if len(msg) > 0:
            msg2log(None, msg, D_LOGS["except"])
            log2All()

        param=(title ,mode, csv_source,dt_col_name,ts_list,discret,n, n_step,n_eval,n_test,n_pred,folder_for_logging, \
           folder_for_train_logging,folder_for_control_logging,folder_for_predict_logging)
        msg2log(None,logParams(param),D_LOGS["main"])
        if n==0:
            closeLogs()
            return -1

    nret   = 0
    msgErr = ""
    try:
        if mode=="offlp":
            ret = drive(df=df, title=title, dt_col_name=dt_col_name, ts_list=ts_list, discret=discret,n=n,
                        n_step=n_step, n_eval=n_eval, n_test=n_test, n_pred=n_pred,
                        folder_for_logging=folder_for_logging, folder_for_train_logging=folder_for_train_logging,
                        folder_for_control_logging=folder_for_control_logging,
                        folder_for_predict_logging=folder_for_predict_logging)
        else:
            pass #TODO
    except:
        msgErr = "O-o-ops! I got an unexpected error - reason  {}\n".format(sys.exc_info())
        nret   = -1
    finally:
        if len(msgErr)>0:
            msg2log(None,msgErr,D_LOGS['except'])

        message = "Time execution logging stoped at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
        msg2log(None, message, D_LOGS["timeexec"])
        closeLogs()  # The loga are closing
    return nret

if __name__=="__main__":
    nret = 1
    nret = main(len(sys.argv), sys.argv)

    sys.exit(nret)
