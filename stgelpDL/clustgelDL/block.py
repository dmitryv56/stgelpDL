#!/usr/bin/python3

""" These classes describe the sequences of the time series"""
import sys

import copy
import dateutil
from pathlib import Path
from random import randrange

import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustgelDL.kmean import kMeanCluster, block_to_cluster,cluster_Blocks,logClusterblock
from predictor.utility import msg2log, svld2log


class baseBlock():
    _number_block = 0

    def __init__(self, block_size, start_index, start_label, data_list, f=None):
        self.block_size = block_size
        self.start_index = start_index
        self.start_label = start_label
        self.data_list = copy.copy(data_list)
        self.block_id = type(self)._number_block
        self.f =f
        type(self)._number_block = type(self)._number_block + 1


class block(baseBlock):
    def __init__(self, block_size, start_index, start_label, data_list, f=None):
        self.pca_list = []
        self.hist = None
        self.bin_edges = None
        self.signed_states = []
        self.duration = []
        self.countStates={}  #number of times stays in the one of states {-1,0,+1}
        super().__init__(block_size, start_index, start_label, data_list, f)

    def buildSignedStates(self):
        for item in self.data_list:
            if item < -1e-6:
                self.signed_states.append(-1)
            elif item < 1e-06:
                self.signed_states.append(0)
            else:
                self.signed_states.append(1)
        st = self.signed_states[0]
        nst = 1
        for item in self.signed_states[1:]:
            if item == st:
                nst += 1
            else:
                self.duration.append((st, nst))
                st = item
                nst = 1
        self.countStates[-1]=sum(1 for x in self.signed_states if x<0)
        self.countStates[0] =sum(1 for x in self.signed_states if x==0)
        self.countStates[1] = sum(1 for x in self.signed_states if x > 0)
        return

    def buildHistogram(self):
        self.hist, self.bin_edges = np.histogram(self.data_list)

    def blockProperties(self, name="", cluster_Id=None):
        self.buildSignedStates()
        self.buildHistogram()
        message=f"""
        Block Id:  {self.block_id} Type: {self.__class__.__name__} Cluster Id:{cluster_Id} Name: {name} Size: {self.block_size}
        Sequence of sign: {self.signed_states}
        Durations and transitions: {self.duration}
        Duration in '-1': {self.countStates[-1]}  Duration in '0': {self.countStates[0]}  Duration in '+1': {self.countStates[1]}
        Histogram bins: {self.bin_edges}
        Histogram: {self.hist}

        """
        msg2log(None,message,self.f)
        return

    def printHist(self, path_to_folder):

        if self.__class__.__name__ == 'cluster':
            title = "Histogram (Cluster {})".format(self.cluster_Id)
            png_file_ ="Histogram_Cluster_{}.png".format(self.cluster_Id)
        else:
            title = "Histogram (Block started at {})".format(self.start_label)
            png_file_ = "Histogram_Block_{}_start_index_{}.png".format(self.block_id, self.start_index)
        msg = ""
        try:
            png_file=Path(path_to_folder / png_file_)
            plt.hist(self.data_list,density=True,bins=int(len(self.data_list)/10), label ="Data")
            mn,mx =plt.xlim()
            kde_xs =np.linspace(mn,mx, 301)
            kde = st.gaussian_kde(self.data_list)
            plt.plot(kde_xs,kde.pdf(kde_xs), label='PDF')
            plt.legend(loc='upper left')
            plt.ylabel('Probability')
            plt.xlabel('Data')
            plt.title(title)

            plt.savefig(png_file)

        except np.linalg.LinAlgError:
            msg="Oops! raise LinAlgError(singular matrix)"
        except:
           msg ="Unexpected error: {}".format( sys.exc_info()[0])
        finally:

            msg2log(None,msg,self.f)
            plt.close("all")
        return

class cluster(block):
    def __init__(self, name, block_size, cluster_Id, label_list, cluster_center_list, f=None):

        self.name = name
        self.label_list=copy.copy(label_list)
        self.cluster_Id=cluster_Id

        super().__init__(block_size, -1, "", cluster_center_list, f)

    def blockProperties(self):
        super().blockProperties(self.name, self.cluster_Id)

# API
def auxData_list(data_list, start_block, block_size):
    return data_list[start_block:start_block + block_size ]


def createBlocks(df: pd.DataFrame, dt_col_name: str, data_col_name: str, block_size: int, number_blocks: int,
                 f: object = None) -> list:
    list_blocks = []
    data_list = [x for x in df[data_col_name].tolist() if np.isnan(x) == False]
    dt_list = [x for x in df[dt_col_name].tolist()]
    start_block = 0  # This block is exactly alignmented to begin of the time series
    list_blocks.append(
        block(block_size, start_block, dt_list[start_block], auxData_list(data_list, start_block, block_size), f))
    for nb in range(number_blocks - 2):
        pass
        start_block = randrange(1, len(data_list) - block_size)
        list_blocks.append(
            block(block_size, start_block, dt_list[start_block], auxData_list(data_list, start_block, block_size), f))
    start_block = len(data_list) - block_size  # This block is exactly alignmented  to end of the time series
    list_blocks.append(
        block(block_size, start_block, dt_list[start_block], auxData_list(data_list, start_block, block_size), f))

    msg2log(createBlocks.__name__, "Generated {} blocks of {} size".format(number_blocks, block_size), f)
    k = 0
    for item in list_blocks:
        msg2log(None, "\n{} block. {} ".format(k, item), f)
        k += 1
    return list_blocks


def listBlocks2matrix(list_blocks: list, f: object = None) -> (np.array, list):
    pass
    if list_blocks is None or not list_blocks:
        return None, None

    n = len(list_blocks)
    m = list_blocks[0].block_size
    labelX = []

    X = np.zeros((n, m), dtype=float)
    meanX = np.zeros((m), dtype=float)
    k = 0
    for item in list_blocks:
        labelX.append(item.start_label)
        X[k, :] = item.data_list
        k += 1
        meanX = meanX + item.data_list
    meanX = meanX / float(n)

    msg = " X matrix {} x {}\n\n".format(n, m)
    msg2log(None, msg, f)
    svld2log(X, labelX, 4, f)

    msg = " Mean values vector {} \n\n".format(m)
    msg2log(None, msg, f)
    svld2log(meanX, [n], 4, f)
    for k in range(n):
        X[k, :] = X[k, :] - meanX

    print(np.sum(X[:, 0]), np.sum(X[:, 1]))

    msg = " Normalized X matrix {} x {}\n\n".format(n, m)
    msg2log(None, msg, f)
    svld2log(X, labelX, 4, f)
    return X, labelX

def learningData(list_blocks: list, block_belongs_to_cluster: dict, f: object=None)->(np.array,np.array):

    pass
    if list_blocks is None or not list_blocks or block_belongs_to_cluster is None or not block_belongs_to_cluster:
        return None,None

    n=len(list_blocks)
    m=list_blocks[0].block_size
    X=np.zeros((n,m),dtype=float)
    y=np.zeros((n),dtype=float)
    k=0
    for item in list_blocks:
        ClusterId =getClusterId(item.start_label, block_belongs_to_cluster)
        if ClusterId is not None:
            X[k,:]=item.data_list
            y[k]=ClusterId
            k+=1
        else:
            msg="Cluster ID is not found for block {} with label {}. This block ignored.".format(item.block_id,item.start_label)
            msg2log(None,msg,f)
        pass
    return X,y

def getClusterId(labelBlock,block_belongs_to_cluster):

    if labelBlock in block_belongs_to_cluster:
        str_value=block_belongs_to_cluster[labelBlock]
        # string parsing to fetch cluster id 'belongs to 3 cluster with center [-0.25   .... ]'
        key=int(str_value.split(' ')[2])
    else:
        key= None
    return key

def isMidnight(ISO8601str: str) -> bool:
    bRet = False
    t = dateutil.parser.parse(ISO8601str)
    if t.hour == 0 and t.minute == 0:
        bRet = True
    return bRet

def getPComponents(eig_vals: np.array, eig_vecs: np.array, ratio: float = 0.6, f: object = None) -> (np.array, np.array):
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

def createInputOutput(name, df, dt_col_name, data_col_name, block_size,number_blocks,cluster_max,file_png_path, \
                      f:object=None)->(np.array,np.array,list,list):


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
    msg2log(None, message, f)
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

    return X_learning, y_desired,list_blocks,list_clusters


