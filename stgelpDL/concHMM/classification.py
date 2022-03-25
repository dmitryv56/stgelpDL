#!/usr/bin/env python3

""" Multivatiate classification of observed data by K-meabs method.
The observer data is csv-dataset contains timestamp columns and other time series columns.
"""

import logging
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from conc_hmm import DataHelper
from cfg_classification import PATH_ROOT_FOLDER,LOG_FOLDER,CLASSIFICATION_TASKS


def plotStates(ds:pd.DataFrame=None, task:str="", factor_list:list=[],state_sequence:list=[], title:str="",
               path_to_file:str="",   start_index: int=0, end_index: int = 0):

    if end_index >len(state_sequence): end_index = len(state_sequence)
    if end_index == 0:                 end_index = len(state_sequence)
    suffics='.png'
    ss="{}_{}_".format(str(start_index), str(end_index))
    file_png = Path(path_to_file, task + "_States_" + ss + Path(__file__).stem).with_suffix( suffics)
    n=len(factor_list)
    fig,axs=plt.subplots(n+1)
    fig.suptitle("{} (from {} till {}".format(title,start_index,end_index))
    for i in range(n):
        axs[i].plot(ds[factor_list[i]].values[start_index:end_index])
    axs[n].plot(state_sequence[start_index:end_index])

    plt.savefig(file_png)
    plt.close("all")
    return


def k_class(data:DataHelper = None,task:str="Demand",task_folder:str = 'Demand',factor_list:list =['Real_demand'],
            subplot_size:int = 1008):
    pass
    data.exogenious=factor_list
    data.file_png_path = task_folder
    title=task
    pca_title="PCA_{}".format(task)
    cluster_centers, cluster_labels, pca_cluster_centers, pca_cluster_labels =data.clusterData(title=title,
                                                                                               pca_title=pca_title)

    for n_clusters, clusters in cluster_labels.items():
        path_to_file = str(Path(task_folder)/Path("{}_classes".format(n_clusters)))
        Path(path_to_file).mkdir(parents=True, exist_ok=True)
        (n_size,) = clusters.shape
        start_index = 0
        while start_index<n_size:
            end_index=start_index + subplot_size
            if end_index>n_size:
                end_index=n_size
            plotStates(ds=data.df, task=task, factor_list=factor_list, state_sequence=clusters,
                       title="{}\n{} classes, start at {}".format(task, n_clusters, data.dt[start_index]),
                       path_to_file=path_to_file, start_index=start_index, end_index=end_index)
            start_index=end_index


    pass

if __name__ == "__main__":
    folder_for_predict_logging =str(Path(PATH_ROOT_FOLDER / LOG_FOLDER ))
    Path(folder_for_predict_logging).mkdir(parents=True, exist_ok=True)
    filename = str(Path(Path(folder_for_predict_logging) / Path("log")).with_suffix(".log"))
    logging.basicConfig(filename=filename, filemode='a',
                        level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    dataset = '/home/dmitry/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_20102020_conchmm.csv'
    ts = "Real_demand"
    dt = "Date Time"
    endogenious = ['Programmed_demand', 'Real_demand']
    # exogenious = ['Diesel_Power', 'WindGen_Power','HydroTurbine_Power','Pump_Power']
    exogenious = ['Diesel_Power', 'WindGen_Power', 'Hydrawlic']
    dhlp = DataHelper(dataset=dataset, timestamps=dt, endogenious=endogenious, exogenious=exogenious)
    dhlp.readData()
    for task, factor_list in CLASSIFICATION_TASKS.items():
        task_folder = str(Path(folder_for_predict_logging)/Path(task))
        Path(task_folder).mkdir(parents=True, exist_ok=True)
        k_class(data=dhlp, task=task, task_folder=task_folder, factor_list=factor_list)

