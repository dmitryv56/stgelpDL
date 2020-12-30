#!/usr/bin/python3

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def prepareLossCharts(object_history:object, folder:str=None,f:object=None):

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