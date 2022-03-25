#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def pltViterbi(file_png_path:str = "", pref_file_name:str = "HMM", subtitle:str = "", observations:np.array = None,
                    viterbi_path:np.array = None, hidden_sequence:list = []):

    """

    :param pref_file_name: i.e. 'Imbalance'
    :param subtitle:  i.e. 'start at <data time>'
    :param observations:
    :param viterbi_path:
    :param hidden_sequence:
    :return:
    """


    suffics = ".png"
    if hidden_sequence is not None:
        file_name = "{}_viterbi_sequence_vs_hidden_sequence".format(pref_file_name)
    else:
        file_name = "{}_viterbi_sequence".format(pref_file_name)
    vit_png = Path(Path(file_png_path)/Path(file_name)).with_suffix(suffics)
    # vit_png = Path(cp.folder_predict_log / file_png)

    try:
        # plt.plot(observations, viterbi_path, label="Viterbi path")
        # plt.plot(observations, hidden_sequence, label="Hidden path")
        plt.plot(viterbi_path, label="Viterbi path")

        if hidden_sequence is not None:
            plt.plot(hidden_sequence, label="Hidden path")

        else:
            pass
        numfig = plt.gcf().number
        fig = plt.figure(num=numfig)
        fig.set_size_inches(18.5, 10.5)
        if hidden_sequence is not None:
            fig.suptitle("Viterby optimal path vs hidden path for dataset\n{}".format(subtitle), fontsize=24)
        else:
            fig.suptitle("Viterby optimal path for dataset\n{}".format(subtitle), fontsize=24)
        plt.ylabel("States", fontsize=18)
        plt.xlabel("Observation timestamps", fontsize=18)
        plt.legend()
        plt.savefig(vit_png)

    except:
        pass
    finally:
        plt.close("all")
    return


def plotClusters(kmeans: KMeans=None, X: np.array=None, file_png:str = "Clusters.png")->str:
    """
    The plot shows 2 first component of X
    :param kmeans: -sclearn.cluster.Kmeans object
    :param X: matrix n_samples * n_features or principal component n_samples * n_components.
    :param file_png:
    :return:
    """
    ret_status = "The cluster plot saved in {}".format(file_png)
    if kmeans is None or X is None:
        ret_status = "plotClusters: Error - no data"
        return ret_status
    (n,m)=X.shape
    if m<2:
        ret_status = "plotClusters: Error -X should have 2 features."
        return ret_status
    plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=50)
    plt.savefig(file_png)
    plt.close("all")
    return ret_status

if __name__ == "__main__":
    pass