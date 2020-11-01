#!/usr/bin/python3

""" These classes describe the sequences of the time series"""
import copy
from random import randrange

import numpy as np
import pandas as pd

from predictor.utility import msg2log, svld2log


class baseBlock():
    _number_block = 0

    def __init__(self, block_size, start_index, start_label, data_list, f=None):
        self.block_size = block_size
        self.start_index = start_index
        self.start_label = start_label
        self.data_list = copy.copy(data_list)
        self.block_id = type(self)._number_block
        type(self)._number_block = type(self)._number_block + 1


class block(baseBlock):
    def __init__(self, block_size, start_index, start_label, data_list, f=None):
        self.pca_list = []
        self.hist = None
        self.bin_edges = None
        self.states = []
        self.duration = []
        super().__init__(block_size, start_index, start_label, data_list, f)

    def buildStates(self):
        for item in self.list_data:
            if item < -1e-6:
                self.states.append(-1)
            elif item < 1e-06:
                self.states.append(0)
            else:
                self.states.append(1)
        st = self.states[0]
        nst = 1
        for item in self.states[1:]:
            if item == st:
                nst += 1
            else:
                self.duration.append((st, nst))
                st = item
                nst = 1
        return

    def buildHistogram(self):
        self.hist, self.bin_edges = np.histogram(self.list_data)


class cluster(block):
    def __init__(self, name, block_size, cluster_Id, label_list, cluster_center_list, f=None):

        self.name = name
        self.label_list=copy.copy(label_list)
        self.cluster_Id=cluster_Id

        super().__init__(block_size, -1, "", cluster_center_list, f)



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


