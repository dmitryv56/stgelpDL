# /usr/bin/python3

"""State classification for green electricity load by using Deep Learning - stcgelDL"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


import numpy as np
import pandas as pd

from clustgelDL.block import  createInputOutput
from clustgelDL.NNmodel import create_model,create_LSTMmodel,createTfDatasets,createTfDatasetsLSTM,fitModel,\
    predictModel,EPOCHS
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from predictor.api import chart_MAE,chart_MSE
from predictor.utility import  msg2log
from tsstan.pltStan import setPlot, plotAll


__version__ = '0.0.0'


def main(argc, argv):

    # command-line parser
    sDescriptor       = 'State Classification (green) electricity load processes(Time Series -TS) by using Deep Learning'
    sCSVhelp          = "Absolute path to a source dataset (csv-file)."
    sEndogenTSnameHelp  = "Endogenous Variable (TS,column name)  in the dataset."
    sExogenTSnameHelp = "Exogenous Variables. A string contains the comma-separated the column names of TS  in the dataset. "
    sLabelnameHelp    = "Label or desired data. The column name in the dataset. "
    sTimeStampHelp    = "Time Stamp (column) name in the dataset."
    sn_stepHelp       = "The step period for endogenious TS for Supervised Learning data creating."
    sdiscretHelp      = "Time series sampling discretization(minutes)"
    snum_clustersHelp = "A-priory set of number of classes(clusters).There is a number of neuron in output level " + \
                        "of Neuron Net."
    snum_predictsHelp = "Predict period. The last part of time series is used for predict."


    parser = argparse.ArgumentParser(description=sDescriptor)

    parser.add_argument('-m', '--mode',         dest='cl_mode',         action='store', default='stcks',
                        choices=['stcls'],
                        help='Possible modes. Only default mode enabled for current version.')
    parser.add_argument('-c', '--csv_dataset',  dest='cl_dset',         action='store',            help=sCSVhelp)
    parser.add_argument('-t', '--endogen',      dest='cl_endots',       action='store', default='Imbalance',
                        help=sEndogenTSnameHelp)
    parser.add_argument('-x', '--exogen',       dest='cl_exogts',       action='store', default='',
                        help=sExogenTSnameHelp)
    parser.add_argument('-l', '--label',        dest='cl_labels',         action='store', default='',
                        help=sLabelnameHelp)
    parser.add_argument('--timestamp',          dest='cl_timestamp',    action='store', default='Date Time',
                        help=sTimeStampHelp)
    parser.add_argument('-n', '--n_step',       dest='cl_n_step',       action='store', default=32,  help=sn_stepHelp)

    parser.add_argument('-u', '--num_clusters', dest='cl_num_clusters', action='store',default=10,help=snum_clustersHelp)
    parser.add_argument('--discret',            dest='cl_discret',      action='store', default=10, help=sdiscretHelp)
    parser.add_argument('-p', '--num_predicts', dest='cl_num_predicts', action='store', default=4,
                        help=snum_predictsHelp)
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

Dataset path                       : {args.cl_dset}
Mode                               : {args.cl_mode}
Endogenious Time Series in dataset : {args.cl_endots}
Exogenious Time Series in dataset  : 
{args.cl_exogts}
Labels (desired data) in dataset   : {args.cl_labels}
Timestamp in dataset               : {args.cl_timestamp} 
Discretization (min)               : {args.cl_discret}
Step period for endogenious TS     : {args.cl_n_step}
Number of target clusters (number of neurons in the output layer of Neuron Net)
                                   : {args.cl_num_clusters}
Predict period                     : {args.cl_num_predicts}

    """
    dt_col_name      = args.cl_timestamp #"Date Time"
    data_col_name    = args.cl_endots    # "Imbalance"
    exogenious_list  = list(args.cl_exogts.split(','))
    labels_name      = args.cl_labels
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    message1 ="Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))




    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_for_logging=Path(dir_path)/"Logs"/"{}_{}".format(data_col_name,date_time)

    listLogSet(str(folder_for_logging))  # A logs are creating
    msg2log(None, message0, D_LOGS["clargs"])
    msg2log(None, message1, D_LOGS["timeexec"])


    fc=D_LOGS["control"]
    fp=D_LOGS["predict"]
    ft=D_LOGS["train"]
    folder_for_train_logging=str(Path(os.path.realpath(ft.name)).parent)
    folder_for_control_logging=str(Path(os.path.realpath(fc.name)).parent)
    folder_for_predict_logging=str(Path(os.path.realpath(fp.name)).parent)

    setPlot()

    csv_source    = args.cl_dset   # "~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
    name          = "{}_Clusters".format(data_col_name)
    discret       = int(args.cl_discret)
    n_step        = int(args.cl_n_step)
    cluster_max   = int(args.cl_num_clusters)
    n_pred        = int(args.cl_num_predicts)

    message = f"""  Common parameters set

Dataset path                       : {csv_source}
Mode                               : {args.cl_mode}
Endogenious Time Series in dataset : {data_col_name}
Exogenious Time Series in dataset  : {exogenious_list}
Labels (desired data) in dataset   : {labels_name}
Timestamp in dataset               : {dt_col_name} 
Discretization (min)               : {discret}
Step period for endogenious TS     : {n_step}
Number of target clusters (number 
of neurons in the output layer of 
Neuron Net)                        : {cluster_max}
Predict period                     : {n_pred}
Folder for control logging         : {folder_for_control_logging}
Folder for train logging           : {folder_for_train_logging}
Folder for predict logging         : {folder_for_predict_logging}
Folder for plots                   : {D_LOGS["plot"]}
Logs                               :
{logList()}
        """
    msg2log(None, message, D_LOGS["main"])

    df = pd.read_csv(csv_source)

    # X_learning,y_desired,list_blocks, list_clusters =  createInputOutput(name, df, dt_col_name, data_col_name,
    #                                                     block_size, number_blocks, cluster_max, D_LOGS["plot"], fc)
    #
    # #Cluster properties
    # messages=f"""
    #     Cluster properties and Histograms
    # """
    # msg2log(None,messages,fc)
    # for item in list_clusters:
    #     item.blockProperties()
    #     item.printHist(D_LOGS["plot"])
    # # block properties
    # messages = f"""
    #         Block properties and Histograms
    #         (Only first 20 blocks)
    #     """
    # msg2log(None, messages, fc)
    # for item in list_blocks[:20]:
    #     item.blockProperties()
    #     item.printHist(D_LOGS["plot"])
    # # deep learning
    #
    # model = create_model(block_size, cluster_max,f=ft)
    # train_dataset, test_dataset = createTfDatasets(X_learning[:number_blocks - n_pred, :],
    #                                                y_desired[:number_blocks - n_pred], validationRatio=0.1, f=ft)
    # # model = create_LSTMmodel(block_size, cluster_max, f=ft)
    # # train_dataset, test_dataset = createTfDatasetsLSTM(X_learning[:number_blocks - n_pred, :],
    # #                                                y_desired[:number_blocks - n_pred], validationRatio=0.1, f=ft)
    #
    # history, eval_history = fitModel(model, train_dataset,  test_dataset, n_epochs=EPOCHS, f=ft)
    #
    # chart_MAE("MultiLayer Model", data_col_name, history, block_size, folder_for_train_logging,False)
    #
    # chart_MSE("MultiLayer Model", data_col_name, history, block_size, folder_for_train_logging,False)
    #
    # y_pred = predictModel(model, X_learning[number_blocks-n_pred:,:],
    #                       [ list_blocks[i].start_label for i in range(number_blocks-n_pred,number_blocks)], f=fp)
    # msg=f"""
    #     Prediction
    #     for Timestamp labels: {[ list_blocks[i].start_label for i in range(number_blocks-n_pred,number_blocks)]}
    #     predicted states: {y_pred}
    # """
    # msg2log(None,msg,fp)



    message = "Time execution logging stoped at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    msg2log(None, message, D_LOGS["timeexec"])
    closeLogs()  # The loga are closing
    return



