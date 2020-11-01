#!/usr/bin/python3
"""  Clusterization (green) electricity load processes by using Deep Learning """

import copy
from datetime import datetime
import os
from pathlib import Path
from random import seed, randrange

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from clustgelDL.block import createBlocks, auxData_list, listBlocks2matrix, cluster,learningData
from clustgelDL.kmean import kMeanCluster, block_to_cluster,cluster_Blocks,logClusterblock
from clustgelDL.NNmodel import create_model,create_LSTMmodel,createTfDatasets,createTfDatasetsLSTM,fitModel,predictModel,EPOCHS
from predictor.api import chart_MAE,chart_MSE
from predictor.cfg import MAGIC_SEED
from predictor.utility import logDictArima, svld2log, msg2log
from tsstan.pltStan import setPlot, plotAll


def isMidnight(ISO8601str: str) -> bool:
    bRet = False
    t = dateutil.parser.parse(ISO8601str)
    if t.hour == 0 and t.minute == 0:
        bRet = True
    return bRet


def crtRandomDictblock(df: pd.DataFrame, dt_col_name: str, data_col_name: str, block_size: int, number_blocks: int,
                       f: object = None) -> dict:
    pass
    d_data = {}
    data_list = [x for x in df[data_col_name].tolist() if np.isnan(x) == False]
    dt_list = [x for x in df[dt_col_name].tolist()]
    start_block = 0  # This block is exactly alignmented to begin of the time series
    d_data[dt_list[start_block]] = auxData_list(data_list, start_block, block_size)

    for nb in range(number_blocks - 2):
        pass
        start_block = randrange(1, len(data_list) - block_size)
        d_data[dt_list[start_block]] = auxData_list(data_list, start_block, block_size)

    start_block = len(data_list) - block_size  # This block is exactly alignmented  to end of the time series
    d_data[dt_list[start_block]] = auxData_list(data_list, start_block, block_size)

    msg2log(crtRandomDictblock.__name__, "Generated {} blocks of {} size".format(number_blocks, block_size), f)
    k = 0
    for key, val in d_data.items():
        msg2log(None, "\n{} block. {} : {}".format(k, key, val), f)
        k += 1
    return d_data


def crtDictblock(df: pd.DataFrame, dt_col_name: str, data_col_name: str, typeblock: str = 'day',
                 f: object = None) -> dict:
    d_data = {}
    data_list = [x for x in df[data_col_name].tolist() if np.isnan(x) == False]
    dt_list = [x for x in df[dt_col_name].tolist()]
    k = None
    k_prev = None
    for i in range(len(data_list)):
        if isMidnight(dt_list[i]):
            k = i
            if k_prev is not None:
                d_data[dt_list[k_prev]] = list_day_value
            list_day_value = []
            k_prev = k
        else:
            if k is None:
                continue
        list_day_value.append(data_list[i])

    pass
    return d_data


def dictLists2matrix(d_dict_lists: dict, f: object = None) -> (np.array, list):
    n = len(d_dict_lists)
    labelX = []
    # get first key in dictionary
    key_first = list(d_dict_lists.keys())[0]
    list_first = d_dict_lists[key_first]
    m = len(list_first)
    X = np.zeros((n, m), dtype=float)
    i = 0
    meanX = np.zeros((m), dtype=float)
    for key, value in d_dict_lists.items():
        j = 0
        labelX.append(key)
        for item in value:
            X[i][j] = item
            meanX[j] += item
            j += 1
        i += 1
    #
    msg = " X matrix {} x {}\n\n".format(n, m)
    msg2log(dictLists2matrix.__name__, msg, f)
    svld2log(X, labelX, 4, f)
    meanX = meanX / n
    msg = " Mean values vector {} \n\n".format(m)
    msg2log(dictLists2matrix.__name__, msg, f)
    svld2log(meanX, [n], 4, f)
    for i in range(n):
        for j in range(m):
            X[i][j] = X[i][j] - meanX[j]
    msg = " \n\n Centerd X matrix {} x {}\n\n".format(n, m)
    msg2log(dictLists2matrix.__name__, msg, f)
    svld2log(X, labelX, 4, f)

    return X, labelX

    pass


def getPComponents(eig_vals: np.array, eig_vecs: np.array, ratio: float = 0.6, f: object = None) -> (
np.array, np.array):
    pass
    sum_eig = np.sum(eig_vals)
    n, m = eig_vecs.shape
    n_PCA = 0
    sum_current = 0.0
    l_eig = []
    for i in range(n - 1, -1, -1):
        sum_current += eig_vals[i]
        n_PCA += 1
        l_eig.append(eig_vals[i])
        if sum_current > sum_eig * ratio:
            break
    ev = np.array(l_eig)
    vv = np.zeros((n, n_PCA), dtype=float)
    k = 0
    for i in range(n - 1, n - n_PCA - 1, -1):
        vv[:, k] = eig_vecs[:, i].copy()
        k += 1

    return ev, vv


def pca(X: np.array, labelX: list, f: object = None) -> (np.array, np.array):
    # Data matrix
    n, m = X.shape
    C = np.dot(X.T, X) / (n - 1)
    traceC = np.trace(C)
    # eigen decomposition
    eig_vals, eig_vecs = np.linalg.eigh(C)
    sum_eig = np.sum(eig_vals)
    msg = " \nSum of eigen values {}\n All eigen values in ascending order\n\n{}".format(sum_eig, eig_vals)
    msg2log(pca.__name__, msg, f)
    ev, vv = getPComponents(eig_vals, eig_vecs, ratio=0.6, f=f)
    m, k = vv.shape
    k = ev.shape
    message = f""" Sum of principal eigen values: {np.sum(ev)}
                    They describe                : {np.sum(ev) * 100.0 / sum_eig} % of variations
                    Max eigen value              : {ev[0]}
                    Num/ of pricipal components  : {ev.shape}
                    Principal  eigen values 
                    in descending order          : {ev}
               """
    msg2log(pca.__name__, message, f)
    # project X ontoPC space
    X_pca = np.dot(X, vv)

    msg = " X maped onto PCA  matrix {} x {}\n\n".format(n, k)
    msg2log(pca.__name__, msg, f)
    svld2log(X_pca, labelX, 4, f)

    return X_pca, ev


def kMeanCluster1(name: str, X: np.array, labelX: list, n_component: int, file_png_path: str, f: object = None) -> (
        dict, dict, dict, dict):
    pass
    n_cluster_max = 10
    n_pc = n_component
    cluster_centers = {}
    cluster_labels = {}
    cluster_contains_blocks = {}
    block_belongs_to_cluster = {}
    Path(file_png_path).mkdir(parents=True, exist_ok=True)

    for n_clusters in range(1, n_cluster_max):
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:, :n_pc])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:, :])
        message = f"""
                Data maped on {n_pc} principial components 
                Number of clusters : {n_clusters}
                Cluster centers    : {kmeans.cluster_centers_}
                Cluster labels (i-th block belongs to j-th cluster) : {kmeans.labels_}


        """
        msg2log(kMeanCluster.__name__, message, f)
        fl_name = "{}_pc{}_clusters{}.png".format(name, n_pc, n_clusters)
        file_png = Path(Path(file_png_path) / Path(fl_name))

        plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c=kmeans.labels_.astype(float), s=50, alpha=0.5)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=50)
        plt.savefig(file_png)
        plt.close("all")
        cluster_centers[n_clusters] = kmeans.cluster_centers_
        cluster_labels[n_clusters] = kmeans.labels_
        d_cluster_contains_blocks, d_block_belongs_to_cluster = \
            block_to_cluster(n_clusters, kmeans.cluster_centers_, kmeans.labels_, labelX, f)
        cluster_contains_blocks[n_clusters] = d_cluster_contains_blocks
        block_belongs_to_cluster[n_clusters] = d_block_belongs_to_cluster
    return cluster_centers, cluster_labels, cluster_contains_blocks, block_belongs_to_cluster


# def block_to_cluster(n_clusters, centeroids, cluster_labels, labelX, f=None) -> (dict, dict):
#     cluster_contains_blocks = {}
#     block_belongs_to_cluster = {}
#     for i in range(n_clusters):
#         cluster_contains_blocks[i] = []
#
#     for i in range(len(cluster_labels)):
#         lst = cluster_contains_blocks[cluster_labels[i]]
#         lst.append(labelX[i])
#         cluster_contains_blocks[cluster_labels[i]] = lst
#         block_belongs_to_cluster[labelX[i]] = \
#             "belongs to {} cluster with center {}".format(cluster_labels[i],
#                                                           np.around(centeroids[cluster_labels[i], :], 3))
#     logClusterblock(cluster_contains_blocks, block_belongs_to_cluster, f)
#
#     return cluster_contains_blocks, block_belongs_to_cluster


# def logClusterblock(cluster_contains_blocks: dict, block_belongs_to_cluster: dict, f: object = None):
#     msg = "\nNumber of clusters : {}\n".format(len(cluster_contains_blocks))
#     msg2log(logClusterblock.__name__, msg, f)
#     msg = "\n\nCluster contains the following blocks\n"
#     msg2log(None, msg, f)
#     for key, l_val in cluster_contains_blocks.items():
#         msg = "Cluster: {}   {}".format(key, l_val[0])
#         msg2log(None, msg, f)
#         for item in l_val[1:]:
#             msg = "              {}".format(item)
#             msg2log(None, msg, f)
#
#     msg = "\n\nblock belongs to cluster \n"
#     msg2log(None, msg, f)
#     for key, val in block_belongs_to_cluster.items():
#         msg = "{} {}".format(key, val)
#         msg2log(None, msg, f)
#     return


# def main_1():
#     pass
#     f = None
#     setPlot()
#     file_png_path = "DaysPlot"
#     csv_source = "~/LaLaguna/stgelpDL/dataLaLaguna/Imbalance_ElHiero_24092020_20102020.csv"
#     df = pd.read_csv(csv_source)
#     name = "Imbalance_Clusters"
#     dt_col_name = "Date Time"
#     data_column_name = "Imbalance"
#
#     pass
#     n_pc = 0
#     block_size = 288
#     number_blocks = 100
#     with open("Block{}.log".format(block_size), "w") as fl:
#         d_day_data = crtDictblock(df, dt_col_name, data_column_name, typeblock='day', f=None)
#         d_day_data = crtRandomDictblock(df, dt_col_name, data_column_name, block_size, number_blocks, fl)
#         logDictArima(d_day_data, 4, fl)
#         X, labelX = dictLists2matrix(d_day_data, fl)
#
#         X_pca, eig_vals = pca(X, labelX, fl)
#         n_pc, = eig_vals.shape
#         sum_eig_vals = np.sum(eig_vals)
#         message = f"""
#             Sum of Eigen Values : {sum_eig_vals}
#             First Eigen Value   : {eig_vals[0]}   {eig_vals[0] * 100 / sum_eig_vals} %
#             Second Eigen Value  : {eig_vals[1]}   {eig_vals[1] * 100 / sum_eig_vals} %
#         """
#         msg2log(main.__name__, message, fl)
#         plt.scatter(X_pca[:, 0], X_pca[:, 1])
#         plt.savefig("TwoComponent.png")
#         plt.close("all")
#     file_png_path = "BlockClusters"
#     with open("Block{}Clusters.log".format(block_size), 'w') as fk:
#
#         cluster_centers, cluster_labels, cluster_contains_blocks, block_belongs_to_cluster = \
#             kMeanCluster(name, X_pca, labelX, n_pc, file_png_path, fk)
#     pass
#     if False:
#         for data_date, data_list in d_day_data.items():
#             plotAll(name, data_list, "{}_{}".format(data_column_name, data_date), file_png_path, f=None)
#             pass


def createInputOutput(name, df, dt_col_name, data_col_name, block_size,number_blocks,cluster_max,file_png_path, \
                      f:object=None)->(np.array,np.array,list):


    n_pc = 0

    list_clusters = []
    list_blocks=[]

    # with open("Sequence{}.log".format(block_size), "w") as fl:

    list_blocks = createBlocks(df, dt_col_name, data_col_name, block_size, number_blocks, f)
    X, labelX = listBlocks2matrix(list_blocks, f)

    X_pca, eig_vals = pca(X, labelX, f)
    n_pc, = eig_vals.shape
    sum_eig_vals = np.sum(eig_vals)
    message = f"""
            Sum of Eigen Values : {sum_eig_vals}
            First Eigen Value   : {eig_vals[0]}   {eig_vals[0] * 100 / sum_eig_vals} %
            Second Eigen Value  : {eig_vals[1]}   {eig_vals[1] * 100 / sum_eig_vals} %
        """
    msg2log(main.__name__, message, f)
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.savefig("TwoPrincipalComponent.png")
    plt.close("all")
    # file_png_path = "SequenceClusters"
    # with open("Sequence{}Clusters.log".format(block_size), 'w') as fk:
    cluster_centers, cluster_labels, all_cluster_contains_blocks, all_block_belongs_to_cluster = \
            kMeanCluster(name, X_pca, labelX, cluster_max=cluster_max+1,type_features='pca', n_component=n_pc, \
                         file_png_path=file_png_path, f=f)
    block_belongs_to_cluster =all_block_belongs_to_cluster[cluster_max]

    cluster_contains_blocks=all_cluster_contains_blocks[cluster_max]
    for val_cluster_Id, label_list in cluster_contains_blocks.items():
        cluster_Id, cluster_center_original_space = cluster_Blocks(val_cluster_Id, label_list, df, dt_col_name, \
                                                                       data_col_name,block_size, f)
        cluster_current=cluster(name, block_size,cluster_Id,label_list, cluster_center_original_space.tolist(),f)
        list_clusters.append(cluster_current)

    X_learning, y_desired = learningData(list_blocks, block_belongs_to_cluster, f)

    return X_learning, y_desired,list_blocks

def main():
    pass

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
    dt_col_name = "Date Time"
    data_col_name = "Imbalance"

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

    csv_source = "~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
    df = pd.read_csv(csv_source)
    name = "Imbalance_Clusters"

    dt_col_name = "Date Time"
    data_col_name = "Imbalance"

    pass
    n_pc = 0
    block_size = 144
    number_blocks = 500
    cluster_max=3
    n_pred=4

    X_learning,y_desired,list_blocks =  createInputOutput(name, df, dt_col_name, data_col_name, block_size, number_blocks,
                                              cluster_max, folder_for_control_logging, fc)


    # deep learning

    model = create_model(block_size, cluster_max,f=ft)
    train_dataset, test_dataset = createTfDatasets(X_learning[:number_blocks - n_pred, :],
                                                   y_desired[:number_blocks - n_pred], validationRatio=0.1, f=ft)
    # model = create_LSTMmodel(block_size, cluster_max, f=ft)
    train_dataset, test_dataset = createTfDatasets(X_learning[:number_blocks-n_pred,:],y_desired[:number_blocks-n_pred],
                                                   validationRatio= 0.1, f=ft)
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
    seed(MAGIC_SEED)
    main()
