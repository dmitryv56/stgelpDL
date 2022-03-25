#!/usr/bin/env python3

import logging
from random import seed, randint
from math import exp
from time import perf_counter
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

from conc_cfg import SEED
from plt_api import plotClusters

logger = logging.getLogger(__name__)

MEAN_COL = 0
STD_COL = 1

seed(SEED)

def execution_time(function):

    def timed(*args,**kw):
        time_start =perf_counter()
        ret_value=function(*args,**kw)
        time_end = perf_counter()

        msg = "\n\n {:.2f} sec for {}({},{})\n".format(time_end - time_start, function.__name__, args,kw)
        logger.info(msg)
        return ret_value

    return timed


""" This function estimates emission probabilities. The normal distributions with 'mean' and 'std' is postulated.
The train data y[t], state sequence and list of states are passed as arguments.
"""


def emisMLE(y: np.array = None, state_sequence: list = [], states: list = []) -> np.array:
    """

    :param y: observations
    :param state_sequence:  state sequences . The size of observations and size of sequence are equal.
    :param states:
    :return:
    """

    if y is None or len(states) == 0 or len(state_sequence) ==0:
        logger.error("{} invalid arguments".format(emisMLE.__name__))
        return None

    emisDist = np.zeros((len(states), 2), dtype=float)    # allocation matrix n_states *2 . The state points on the row.
    # Each row contains a 'mean' and 'std' for this state
    (n,) = y.shape
    msg = ""
    for state in states:
        a_aux = []
        for i in range(n):
            if state_sequence[i] == state:
                a_aux.append(y[i])
        if not a_aux:
            emisDist[state][MEAN_COL] = 0.0
            emisDist[state][STD_COL] = 1e-06
            logger.error("No observations for {} state".format(state))
        elif len(a_aux) == 1:
            emisDist[state][MEAN_COL] = a_aux[0]
            emisDist[state][STD_COL] = 1e-06
        else:
            a = np.array(a_aux, dtype=float)
            emisDist[state][MEAN_COL] = round(a.mean(), 4)
            emisDist[state][STD_COL] = round(a.std(), 6)
        msg = msg + "{}: {} {}\n".format(state, emisDist[state][MEAN_COL], emisDist[state][STD_COL])
    message = f"""
    Emission Prob. Distribution
State Mean  Std 
{msg}

"""
    logger.info(message)
    return emisDist


""" This function estimates the pai (initial) distribution.
The estimate initial pay[state] is proportional of the occurence times for this state.
"""


def paiMLE(states: list = [], count: np.array = None, train_sequence_size: int = 1) -> np.array:
    """

    :param states:list of possible states like as [0,1, ... n_states-1]
    :param count: list of number occurinces for each state in states sequences
    :param train_sequence_size: size of states sequences
    :return:
    """

    if  count is None or len(states)==0 or train_sequence_size == 1:
        logger.error("{} invalid arguments".format(paiMLE.__name__))
        return None

    pai = np.array([round(count[state]/train_sequence_size, 4) for state in states])
    msg = "".join(["{}: {}\n".format(state,pai[state]) for state in states])
    message = f"""
State  Pai
{msg}

"""
    logger.info(message)

    return pai


def transitionsMLE(state_sequence: list = [], states: list = []) -> np.array:
    """

    :param state_sequence:  list of states (states sequence)  corresponding to observations.
    :param states: list of possible states like as [0,1, ... n_states-1]
    :return: transition matrix len(states) * len(states)
    """
    if len(state_sequence) == 0 or len(states) == 0:
        logger.error("{} invalid arguments".format(transitionsMLE.__name__))
        return None


    # Denominators are counts of state occurence along sequence without last item.
    _, denominators = np.unique(state_sequence[:-1], return_counts=True)
    # Note: If some state appears only once one as last item in seqiuence then this state will loss.
    transDist = np.zeros((len(states), len(states)), dtype=float)

    for statei in states:
        denominator = denominators[statei]
        msg = "State {} : ".format(statei)
        for statej in states:
            nominator =0
            for k in range(len(state_sequence)-1):
                if state_sequence[k] == statei and state_sequence[k+1] == statej:
                    nominator += 1
            transDist[statei][statej] = round(float(nominator)/float(denominator), 6)
            msg = msg + "{} ".format(transDist[statei][statej])
        message = f"""{msg}"""
        logger.info(message)
    return transDist

def getEvalSequenceStartIndex(n_train:int=64, n_eval:int=36, n_test:int=0)->int:
    rn = randint(0, n_train -(n_eval+n_test))
    msg = "Evaluation sequence begins at {}, test sequence begins at {} ".format(rn,rn+n_eval)
    if n_test == 0:
        msg = "Evaluation sequence begins at {}".format(rn)

    logger.info(msg)
    return rn


def logViterbiPath(title:str,viterbi_path:np.array,timestamp:np.array,hidden_states:list)->(int, float):

    if viterbi_path is None:
        return
    if hidden_states is None or len(hidden_states)==0:
        logger.info("{} -viterby path\n{}".format(title,viterbi_path))
        return
    msg="{} -viterby_path\nTimestamp  Viterby Path Hidden State No atch".format(title)
    cnt = 0
    rate = 0.0
    for i in range(len(viterbi_path)):
        s='no'
        if viterbi_path[i] == hidden_states[i]:
           s='yes'
           cnt = cnt +1
        msg = msg + "{} {} {} {} {}\n".format(i,timestamp[i],viterbi_path[i],hidden_states[i], s)
    rate = round((cnt / len(viterbi_path) * 100), 2)
    msg = msg + "\n\nMatches: {}\nRate:{}%\n".format(cnt, rate)

    logger.info(msg)
    return cnt, rate

def logMarginPrb(title:str, logits:np.array, timestamp:np.array):
    pass

    (n,n_states) = logits.shape
    msg="{} margin probability\nTimestamp "
    for i in range(n_states):
        msg=msg + " State {} ".format(i)
    msg=msg+"\n"
    for i in range(n):
        msg = msg + "\n{} ".format(timestamp[i])
        for j in range(n_states):
            msg = msg + " {} ".format(1.0/(1.0+exp(logits[i][j])))
    logger.info(msg)
    return


""" K-mean classificator for state set extraction from dataset. 
X matrix (dataset) contains observation for endogen and exogenious features. Its size is N *M, where N -observations 
number (row in matrix), M-features number(columns in matrix).

"""

@execution_time
def getClusters(file_png_path:str="",title:str="",X:np.array = None, labels:np.array="", min_clusters:int =2, max_clusters:int=5) -> \
        (dict,dict):


    cluster_centers = {}
    cluster_labels =  {}
    (n,m) = X.shape
    for n_clusters in range(min_clusters, max_clusters + 1):
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:, :n_pc])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:, :])
        message = f"""
                Data maped on {m} features (or principial components) 
                Number of clusters : {n_clusters}
                Cluster centers    : {kmeans.cluster_centers_}
                Cluster labels (i-th observation belongs to j-th cluster) : {kmeans.labels_}


        """
        logger.info( message)
        fl_name = "{}_{}_clusters{}.png".format(title, m, n_clusters)
        file_png = Path(Path(file_png_path) / Path(fl_name))
        plotClusters(kmeans, X, str(file_png))

        cluster_centers[n_clusters] = kmeans.cluster_centers_
        cluster_labels[n_clusters] = kmeans.labels_

    return cluster_centers, cluster_labels

def getPCAClusters(file_png_path:str = "", title:str="",X:np.array = None, labels:np.array = "", min_clusters:int = 2,
                   max_clusters:int = 5) -> (dict,dict):


    cluster_centers = {}
    cluster_labels = {}
    pca=PCA(2)
    (n,m)=X.shape
    m=2
    df=pca.fit_transform(X)


    for n_clusters in range(min_clusters, max_clusters + 1):
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:, :n_pc])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
        message = f"""
                Data maped on 2 principial components
                Number of clusters : {n_clusters}
                Cluster centers    : {kmeans.cluster_centers_}
                Cluster labels (i-th observation belongs to j-th cluster) : {kmeans.labels_}


        """
        logger.info(message)
        fl_name = "{}_{}_clusters{}.png".format(title, m, n_clusters)
        file_png = Path(Path(file_png_path) / Path(fl_name))
        plotClusters(kmeans, X, str(file_png))

        cluster_centers[n_clusters] = kmeans.cluster_centers_
        cluster_labels[n_clusters] = kmeans.labels_

    return cluster_centers, cluster_labels



if __name__ == "__main__":
    pass
