#!/usr/bin/python3

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictor.utility import msg2log

def prepareLossCharts(object_history:object=None, folder:str=None,f:object=None):

    if object_history is None:
        msg2log(prepareLossCharts.__name__,"No history date available.The accurace charts will be missed",f)
        return
    for metric_name in object_history.model.metrics_names:
        chart_loss(name_model=object_history.model.name, metric_name=metric_name, history=object_history.history,
                   folder=folder, f=f)
    return


def chart_loss(name_model:str="Loss", metric_name:str="loss", history:dict=None, folder:str=None,f:object=None):

    if history is None :
        return
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    fig, ax = plt.subplots()

    key_val="val_{}".format(metric_name)
    label_train="{} (training data)".format(metric_name)
    label_val = "{} (validation data)".format(metric_name)
    ax.plot(history[metric_name], marker='', label=label_train, color=palette(0))
    ax.plot(history[key_val],   marker='',   label=label_val,   color=palette(1))

    plt.legend(loc=2, ncol=2)
    ax.set_title('{} {} function'.format(name_model, metric_name))
    ax.set_xlabel("No. epoch")
    ax.set_ylabel(metric_name)

    if folder is not None:
        file_png=Path(Path(folder)/Path("{}_{}_function".format(name_model,metric_name))).with_suffix(".png")
    else:
        file_png ="{}_{}_function.png".format(name_model, metric_name)
    plt.savefig(file_png)
    plt.close("all")
    return


def _chart_loss(object_history,name_model, name_time_series, history, n_steps, logfolder, stop_on_chart_show=False):
    # Plot history: MAE
    name_model=object_history.model.name
    d_history=object.history.history
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    fig, ax = plt.subplots()

    ax.plot(history.history['loss'], marker='', label='MAE (training data)', color=palette(0))
    ax.plot(history.history['val_loss'], marker='', label='MAE (validation data)', color=palette(1))

    plt.legend(loc=2, ncol=2)
    ax.set_title('Mean Absolute Error{} {} (Time Steps = {})'.format(name_model, name_time_series, n_steps))
    ax.set_xlabel("No. epoch")
    ax.set_ylabel("MAE value")
    # plt.show(block=stop_on_chart_show)

    if logfolder is not None:
        plt.savefig("{}/MAE_{}_{}_{}.png".format(logfolder, name_model, name_time_series, n_steps))
    plt.close("all")
    return

""" The output classes should be indesed as 0,1,..., n-1. The desired data column if dataset contains only them. But
some indexes may be missed. 
If missing states are in desired data then number output is set  max_state+1

The indexes of classes (states) and their frequences are logged.
The function returns the size of output layer of neuron net.
"""
def getOutputSize(df:pd.DataFrame =None,desired_col_name:str=None, f:object=None)->int:
    pass
    if df is None or desired_col_name is None:
        msg="Dataset object is None or desired ata name is none. Exit ..."
        msg2log(None,msg,f)
        return -1

    a=np.array(df[desired_col_name])
    uniq,count =np.unique(a,return_counts=True)
    msg2log(None,"Desired data \n{:^10s} {:^10s}".format("State","Count"),f)
    n,=uniq.shape
    for i in range(n):
        msg2log(None,"{:>10d} {:>10d}".format(uniq[i],count[i]),f)
    st_max=uniq[-1]
    if n<st_max+1:
        msg2log(None,"\nNOTE: some states are missed in desired data\n\n",f)
        n=st_max+1

    return n



