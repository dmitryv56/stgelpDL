#!/usr/bin/python3

""" This module contains functions for classification """

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from predictor.utility import logDictArima, svld2log, msg2log


def kMeanCluster(name: str, X: np.array, labelX: list, cluster_max:int = 4,type_features: str ='pca', n_component: int=2, \
                 file_png_path: str="", f: object = None) -> (dict, dict, dict, dict):
    """

    :param name:  - title for logs
    :param X: np.array((n_samples, n_features)
    :param labelX: -list of n_samples labels for each row of X
    :param type_features: 'pca' -principal components (default) or 'feat' -original features.
    :param n_component: number components for clusterisation
    :param file_png_path:
    :param f:
    :return:
    """
    pass
    n_cluster_max = cluster_max
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
                Data maped on {n_pc} principial components (or first features) 
                Number of clusters : {n_clusters}
                Cluster centers    : {kmeans.cluster_centers_}
                Cluster labels (i-th block belongs to j-th cluster) : {kmeans.labels_}


        """
        msg2log(kMeanCluster.__name__, message, f)
        fl_name = "{}_pc{}_clusters{}.png".format(name, n_pc, n_clusters)
        file_png = Path(Path(file_png_path) / Path(fl_name))
        plotClusters(kmeans, X, file_png)

        cluster_centers[n_clusters] = kmeans.cluster_centers_
        cluster_labels[n_clusters] = kmeans.labels_
        d_cluster_contains_blocks, d_block_belongs_to_cluster = \
            block_to_cluster(n_clusters, kmeans.cluster_centers_, kmeans.labels_, labelX, f)
        cluster_contains_blocks[n_clusters] = d_cluster_contains_blocks
        block_belongs_to_cluster[n_clusters] = d_block_belongs_to_cluster
    return cluster_centers, cluster_labels, cluster_contains_blocks, block_belongs_to_cluster

def plotClusters(kmeans: KMeans, X: np.array, file_png:Path):
    """
    The plot shows 2 first component of X
    :param kmeans: -sclearn.cluster.Kmeans object
    :param X: matrix n_samples * n_features or principal component n_samples * n_components.
    :param file_png:
    :return:
    """
    plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=50)
    plt.savefig(file_png)
    plt.close("all")
    return

def block_to_cluster(n_clusters, centeroids, cluster_labels, labelX, f=None) -> (dict, dict):

    cluster_contains_blocks = {}
    block_belongs_to_cluster = {}
    for i in range(n_clusters):
        cluster_contains_blocks[i] = []

    for i in range(len(cluster_labels)):
        lst = cluster_contains_blocks[cluster_labels[i]]
        lst.append(labelX[i])
        cluster_contains_blocks[cluster_labels[i]] = lst
        block_belongs_to_cluster[labelX[i]] = \
            "belongs to {} cluster with center {}".format(cluster_labels[i],
                                                          np.around(centeroids[cluster_labels[i], :], 3))
    logClusterblock(cluster_contains_blocks, block_belongs_to_cluster, f)

    return cluster_contains_blocks, block_belongs_to_cluster
""" This function estimates the cluster center in original future space"""
def cluster_Blocks(val_cluster_Id:int,  key_list:list, df:pd.DataFrame, dt_col_name:str, data_col_name:str, \
                   n_block_size:int, f:object =None)->(int, np.array):
    """

    :param val_cluser_Id:
    :param key_list:
    :param df:
    :param dt_col_name:
    :param data_col_name:
    :param n_block_size:
    :param f:
    :return:
    """
    pass
    cluster_center=np.zeros((n_block_size), dtype=float)
    for item in key_list:
        n_start = df[dt_col_name].loc[lambda x: x == item].index[0]
        cur_list = df[data_col_name][n_start:n_start+ n_block_size].tolist()
        current_block = np.array(cur_list)

        cluster_center =np.add(cluster_center,current_block)
    cluster_center=cluster_center/(float(n_block_size))

    message =f"""
            Cluster ID: {val_cluster_Id}
            Original cluster center: {cluster_center} 
            
            """
    msg2log(None,message,f)
    return val_cluster_Id, cluster_center


def logClusterblock(cluster_contains_blocks: dict, block_belongs_to_cluster: dict, f: object = None):
    msg = "\nNumber of clusters : {}\n".format(len(cluster_contains_blocks))
    msg2log(logClusterblock.__name__, msg, f)
    msg = "\n\nCluster contains the following blocks\n"
    msg2log(None, msg, f)
    for key, l_val in cluster_contains_blocks.items():
        msg = "Cluster: {}   {}".format(key, l_val[0])
        msg2log(None, msg, f)
        for item in l_val[1:]:
            msg = "              {}".format(item)
            msg2log(None, msg, f)

    msg = "\n\nblock belongs to cluster \n"
    msg2log(None, msg, f)
    for key, val in block_belongs_to_cluster.items():
        msg = "{} {}".format(key, val)
        msg2log(None, msg, f)
    return