#!/usr/bin/python3
"""  Clusterization (green) electricity load processes by using Deep Learning """

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
from predictor.api import chart_MAE,chart_MSE
from predictor.utility import  msg2log
from tsstan.pltStan import setPlot, plotAll

__version__ = '0.0.1'

def main(argc, argv):


    # command-line parser
    sDescriptor       = 'Clusterization (green) electricity load processes(Time Series) by using Deep Learnings'
    sCSVhelp          = "Absolute path to a source dataset (csv-file)."
    sTSnameHelp       = "Time Series (column) name in the dataset."
    sTimeStampHelp    = "Time Stamp (column) name in the dataset."
    sblock_sizeHelp   = "The size of the block (or sequence of consecutive elements of the time series)." + \
                        " There is a number of neurons in the input level of Neuron Net."
    snum_blockHelp    = " Amount of randomly generated sequences. "
    sdiscretHelp      = "Time series sampling discretization(minutes)"
    snum_clustersHelp = "A-priory set of number of classes(clusters).There is a number of neuron in output level " + \
                        "of Neuron Net."
    snum_predictsHelp = "Predict period. The last part of time series is used for predict."


    parser = argparse.ArgumentParser(description=sDescriptor)

    parser.add_argument('-m', '--mode',         dest='cl_mode',         action='store', default='clust',
                        choices=['clust'],
                        help='Possible modes. Only default mode enabled for current version.')
    parser.add_argument('-c', '--csv_dataset',  dest='cl_dset',         action='store',            help=sCSVhelp)
    parser.add_argument('-t', '--tsname',       dest='cl_tsname',       action='store', default='Imbalance',
                        help=sTSnameHelp)
    parser.add_argument('--timestamp',          dest='cl_timestamp',    action='store', default='Date Time',
                        help=sTimeStampHelp)
    parser.add_argument('-b', '--block_size',   dest='cl_block_size',   action='store', default=144,
                        help=sblock_sizeHelp)
    parser.add_argument('-n', '--num_block',    dest='cl_num_block',    action='store', default=500,
                        help=snum_blockHelp)
    parser.add_argument('-l', '--num_clusters', dest='cl_num_clusters', action='store', default=3,
                        help=snum_clustersHelp)
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
    message=f"""  Command-line arguments

            {argstr}


            Dataset path           : {args.cl_dset}
            Mode                   : {args.cl_mode}
            Time Series in dataset : {args.cl_tsname}
            Discretization (min)   : {args.cl_discret}
            Timestamp in dataset   : {args.cl_timestamp}
            Block size (number of neurons in the input layer of Neuron Net)
                                   : {args.cl_block_size}
            Number of blocks being have  be randomly generated
                                   : {args.cl_num_block}
            Number of target clusters (number of neurons in the output layer of Neuron Net
                                   : {args.cl_num_clusters}
            Predict period (number of blocks for which belonging to class is estimated) 
                                   : {args.cl_num_predicts}

    """
    with open("commandline_arguments.log", "w+") as fcl:
        msg2log(None,message,fcl)
    fcl.close()

    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")

    with open("execution_time.log", 'w') as fel:
        fel.write("Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S")))

    dir_path = os.path.dirname(os.path.realpath(__file__))

    folder_for_train_logging = Path(dir_path) / "Logs" / "train" / date_time
    folder_for_control_logging = Path(dir_path) / "Logs" / "control" / date_time
    folder_for_predict_logging = Path(dir_path) / "Logs" / "predict" / date_time
    Path(folder_for_train_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_control_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_predict_logging).mkdir(parents=True, exist_ok=True)

    suffics=".log"
    dt_col_name   = args.cl_timestamp #"Date Time"
    data_col_name = args.cl_tsname  # "Imbalance"

    file_for_train_logging = Path(folder_for_train_logging, data_col_name + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_predict_logging = Path(folder_for_predict_logging, data_col_name + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_control_logging = Path(folder_for_control_logging, data_col_name + "_" + Path(__file__).stem).with_suffix(
        suffics)

    fc = open(file_for_control_logging, 'w+')
    fp = open(file_for_predict_logging, 'w+')
    ft = open(file_for_train_logging, 'w+')


    setPlot()

    csv_source    = args.cl_dset   # "~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
    name          = "{}_Clusters".format(data_col_name)
    discret       = int(args.cl_discret)
    block_size    = int(args.cl_block_size)
    number_blocks = int(args.cl_num_block)
    cluster_max   = int(args.cl_num_clusters)
    n_pred        = int(args.cl_num_predicts)

    message = f"""  Common parameters set
                Dataset path           : {csv_source}
                Mode                   : {args.cl_mode}
                Time Series in dataset : {data_col_name}
                Timestamp in dataset   : {dt_col_name}
                Discretization (min)   : {discret}
                Block size (number of neurons in the input layer of Neuron Net)
                                       : {block_size}
                Number of blocks being have  be randomly generated
                                       : {number_blocks}
                Number of target clusters (number of neurons in the output layer of Neuron Net
                                       : {cluster_max}
                Predict period (number of blocks for which belonging to class is estimated) 
                                       : {n_pred}
                Logs (control path)    : {str(folder_for_control_logging )}
                Logs (predict path)    : {str(folder_for_predict_logging )}
                Logs (train path)      : {str(folder_for_train_logging )}


    """

    msg2log(None, message, fc)

    df = pd.read_csv(csv_source)

    X_learning,y_desired,list_blocks, list_clusters =  createInputOutput(name, df, dt_col_name, data_col_name, block_size, number_blocks,
                                              cluster_max, folder_for_control_logging, fc)

    #Cluster properties
    messages=f"""
        Cluster properties and Histograms
    """
    msg2log(None,messages,fc)
    for item in list_clusters:
        item.blockProperties()
        item.printHist(folder_for_control_logging)
    # block properties
    messages = f"""
            Block properties and Histograms
            (Only first 20 blocks)
        """
    msg2log(None, messages, fc)
    for item in list_blocks[:20]:
        item.blockProperties()
        item.printHist(folder_for_control_logging)
    # deep learning

    model = create_model(block_size, cluster_max,f=ft)
    train_dataset, test_dataset = createTfDatasets(X_learning[:number_blocks - n_pred, :],
                                                   y_desired[:number_blocks - n_pred], validationRatio=0.1, f=ft)
    # model = create_LSTMmodel(block_size, cluster_max, f=ft)
    # train_dataset, test_dataset = createTfDatasetsLSTM(X_learning[:number_blocks - n_pred, :],
    #                                                y_desired[:number_blocks - n_pred], validationRatio=0.1, f=ft)

    history, eval_history = fitModel(model, train_dataset,  test_dataset, n_epochs=EPOCHS, f=ft)

    chart_MAE("MultiLayer Model", data_col_name, history, block_size, folder_for_train_logging,False)

    chart_MSE("MultiLayer Model", data_col_name, history, block_size, folder_for_train_logging,False)

    y_pred = predictModel(model, X_learning[number_blocks-n_pred:,:],
                          [ list_blocks[i].start_label for i in range(number_blocks-n_pred,number_blocks)], f=fp)
    msg=f"""
        Prediction 
        for Timestamp labels: {[ list_blocks[i].start_label for i in range(number_blocks-n_pred,number_blocks)]}
        predicted states: {y_pred}
    """
    msg2log(None,msg,fc)
    fc.close()
    fp.close()
    ft.close()
    fel.close()
    return


if __name__ == "__main__":
    pass
