#!/usr/bin/python3

"""This module contains the functions and classes for desired data building according by given dataset.
For automatically creating desired data are a some ways. Now automatically classification by k-means is
implemented.

Another methods will be implemented in the future as well.

"""

import os
import sys
import copy

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA,PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from stcgelDL.cfg import GlobalConst
from clustgelDL.kmean import kMeanCluster
from predictor.utility import msg2log,vector_logging

@exec_time
def driveDesiredDataBuild(df:pd.DataFrame, method:str="k-mean",  num_class:int = 3, title:str ='ElHierro',
                 dt_col_name:str="Date Time", endogen_col_name:str="Imbalance", exogen_list:list=None,  f:object=None):

    desdata = DesiredData(df, exogen_list=exogen_list, f=f)
    desdata.bldDesiredData()
    desdata.save_dataset()
    desdata.centerClusters()
    del desdata
    desdata=None
    return




class DesiredData():

    def __init__(self, df:pd.DataFrame, method:str="k-mean",  num_class:int = 3, title:str ='ElHierro',
                 dt_col_name:str="Date Time", endogen_col_name:str="Imbalance", exogen_list:list=None,  f:object=None):
        self.df          = df
        self.method      = method
        self.max_cluster = num_class
        self.title       = title
        self.dt_col_name = dt_col_name
        self.endogen_col_name=endogen_col_name
        self.exogen_list = exogen_list
        self.f           = f
        self.X           = None
        self.m           = len(self.exogen_list)+1
        self.n           = len(self.df)
        self.desired     = None
        self.counts      = None
        self.cluster_centers = None
        self.all_names()

    def all_names(self):
        l_names=copy.copy(self.exogen_list)
        l_names.insert(0,self.endogen_col_name)
        self.names=''.join(['{:10s} '.format(item) for item in l_names])

    def save_dataset(self):
        folder =str(os.path.dirname(self.f.name))
        p=Path(Path(folder)/Path("{}_DesiredGen".format(self.endogen_col_name))).with_suffix(".csv")
        msg2log(None,"Updated dataset is {}".format(p), self.f)
        self.df.to_csv(p, index=False)

    @exec_time


    @exec_time
    def crtXmatrix(self):
        if self.X is not None:
            del self.X
        msg=""
        try:
            self.X=np.zeros((self.n,self.m),dtype=float)
            for i in range(self.n):
                self.X[i,0]=self.df[self.endogen_col_name][i]
            k=1
            for item in self.exogen_list:
                for i in range(self.n):
                    self.X[i,k]=self.df[item][i]
                k+=1
            if GlobalConst.getVerbose()==1:
                msg2log(None,self.X,self.f)
            elif GlobalConst.getVerbose()>1:
                title="{} {} {}".format("Index ",self.endogen_col_name,[item for item in self.exogen_list])
                logMatrix(self.X, title=title, f=self.f)
        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}\n{}".format(str(e),sys.exc_info())
        except:
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
        finally:
            if len(msg) > 0:
                msg2log(type(self).__name__, msg, f=D_LOGS["except"])
        return
    @exec_time
    def bldDesiredData(self):
        print_weigth=8
        store_title=self.title
        self.crtXmatrix()
        # 5 scenario
        d_seqs={self.endogen_col_name:  self.df[self.endogen_col_name]}
        d_seqs["allFeat_list"]        = self.allFeaturesScenario()
        d_seqs["PCA_list"]            = self.pcaScenario()
        d_seqs["standardScaler_list"] = self.standadScalerScenario()
        d_seqs["minmaxScaler_list"]   = self.minmaxScalerScenario()
        d_seqs["exogeniousFeat_list"] = self.exogeniousFeaturesScenario()
        n_seq=len(d_seqs["exogeniousFeat_list"])
        # endogeniousFeat = self.endogeniousFeatureScenario()
        d_seqs["total_list"] = []
        for i in range(n_seq):
            d_seqs["total_list"].append( d_seqs["allFeat_list"] [i] + d_seqs["PCA_list"][i]*2 +
                                         (1-d_seqs["standardScaler_list"][i])*4 +d_seqs["minmaxScaler_list"] [i] *8 +
                                         d_seqs["exogeniousFeat_list"][i]*16)

        plotsequences(d_seqs, title= "Belong to desired class", folder=D_LOGS["plot"], f=self.f)
        self.s_uniqval, uniqval, self.counts = self.code2state(d_seqs["total_list"])
        for key,val in d_seqs.items():
            self.df[key]=copy.copy(val)

        self.desired = [uniqval.tolist().index(i) for i in d_seqs["total_list"] ]
        self.df["desired"]=copy.copy(self.desired)

        return

    @exec_time
    def centerClusters(self):
        if self.counts is None:
            return

        n_cluster, =self.counts.shape
        self.cluster_centers = np.zeros((n_cluster,self.m),dtype=float)
        for k in range (n_cluster):

            for i in range(len(self.desired)):
                if self.desired[i] == k:
                    self.cluster_centers[k,:]+=self.X[i,:]
            self.cluster_centers[k,:]=self.cluster_centers[k,:]/self.counts[k]
        title="Cluster Centers\n{:<10s} {}".format("Cluster",self.names)
        logMatrix(self.cluster_centers, title=title, f = self.f)

        for i in range((self.m)-1):
            for j in range(i+1,self.m ):
                self.plotClusters(i,j)
        return

    def plotClusters(self,i:int=0,j:int=1):
        """
        The plot shows 2 first component of X
        :param kmeans: -sclearn.cluster.Kmeans object
        :param X: matrix n_samples * n_features or principal component n_samples * n_components.
        :param file_png:
        :return:
        """
        msg=""
        try:
            k,=self.counts.shape
            l_names=copy.copy(self.exogen_list)
            l_names.insert(0,self.endogen_col_name)
            file_pngname="{}_clusters_{}_{}_projection".format(k,l_names[i],l_names[j])
            file_png=Path(Path(D_LOGS['plot'])/Path(file_pngname)).with_suffix(".png")
            plt.scatter(self.X[:, i].tolist(), self.X[:, j].tolist(), c=np.array(self.desired).astype(float), s=50, alpha=0.5)
            plt.scatter(self.cluster_centers[:, i], self.cluster_centers[:, j], c='red', s=50)
            plt.savefig(file_png)
        except:
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
        finally:
            if len(msg)>0:
                msg2log("plotClusters",msg,D_LOGS["except"])
            plt.close("all")
        return


    @exec_time
    def allFeaturesScenario(self):
        store_title = self.title
        self.title = "{}_{}".format(self.title, "AllFeatures_path_")
        msg = ""
        try:
            desired_list = self.kMean(self.X)

        except:
            desired_list=None
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
        finally:
            if len(msg) > 0:
                msg2log("allFeaturesScenario", msg, f=D_LOGS["except"])
        self.title = store_title
        return desired_list

    @exec_time
    def pcaScenario(self):
        store_title = self.title
        self.title = "{}_{}".format(self.title, "PCA_path_")
        msg = ""
        try:
            pca = PCA(n_components='mle', svd_solver='full')
            X_new = pca.fit_transform(self.X)
            msg2log("Principial Component Analysis", "n_components={}".format(pca.n_components_), f=self.f)
            vector_logging("mean", pca.mean_, 8, f=self.f)
            vector_logging("Explained Variance", pca.explained_variance_, 8, f=self.f)
            vector_logging("Explained Variance Ratio", pca.explained_variance_ratio_, 8, f=self.f)
            vector_logging("Singular Values", pca.singular_values_, 8, f=self.f)

            desired_list=self.kMean(X_new)
        except:
            desired_list = None
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())

        finally:
            if len(msg) > 0:
                msg2log("pcaScenario", msg, f=D_LOGS["except"])
        self.title=store_title
        return desired_list

    @exec_time
    def standadScalerScenario(self):

        store_title = self.title
        self.title = "{}_{}".format(self.title, "StandardScaler_path_")
        # scaler
        msg=""
        try:
            standard_scaler = StandardScaler()
            scaler = standard_scaler.fit(self.X)
            X_scaled = scaler.transform(self.X)
            msg2log("StandardScaler", "", self.f)

            desired_list = self.kMean(X_scaled)
        except:
            desired_list = None
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())

        finally:
            if len(msg) > 0:
                msg2log("standadScalerScenario", msg, f=D_LOGS["except"])

        self.title = store_title
        return desired_list

    @exec_time
    def minmaxScalerScenario(self):
        store_title = self.title
        self.title = "{}_{}".format(self.title, "MinMaxScaler_path_")
        # scaler
        msg = ""
        try:
            minmax_scaler = MinMaxScaler()
            scaler = minmax_scaler.fit(self.X)
            X_scaled = scaler.transform(self.X)
            msg2log("MinMaxScaler", "", self.f)

            desired_list = self.kMean(X_scaled)
        except:
            desired_list = None
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())

        finally:
            if len(msg) > 0:
                msg2log("minmaxScalerScenario", msg, f=D_LOGS["except"])

        self.title = store_title
        return desired_list

    @exec_time
    def exogeniousFeaturesScenario(self):
        store_title = self.title
        self.title = "{}_{}".format(self.title, "ExogeniusFeatures_path_")
        msg=""
        try:
            desired_list = self.kMean(self.X[:,1:])
        except:
            desired_list = None
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())

        finally:
            if len(msg) > 0:
                msg2log("exogeniousFeaturesScenario", msg, f=D_LOGS["except"])

        self.title = store_title
        return desired_list

    @exec_time
    def endogeniousFeatureScenario(self):
        store_title = self.title
        self.title = "{}_{}".format(self.title, "EndogeniusFeature_path_")
        msg = ""
        try:
            desired_list = self.kMean(self.X[:,0])
        except:
            desired_list = None
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())

        finally:
            if len(msg) > 0:
                msg2log("endogeniousFeatureScenario", msg, f=D_LOGS["except"])
        self.title = store_title
        return desired_list

    @exec_time
    def kMean(self, X:np.array)->list:
        pass
        (n,m)=X.shape
        f=open("temp.log",'w')
        d1,d2,d3,d4 =kMeanCluster(self.title, X, self.df[self.dt_col_name], cluster_max=self.max_cluster,
                                  type_features='pca', n_component = m, file_png_path= D_LOGS["plot"], f=f)
        f.close()
        os.remove("temp.log")
        pass
        opt_class_number = len(d2)
        desired_list=copy.copy(d2[opt_class_number])
        return desired_list

    def code2state(self, seq: list):
        a = np.array(seq)
        uniqval, counts = np.unique(a, return_counts=True)
        s_uniqcode=[hex(i) for i in uniqval.tolist()]
        msg = f"""
uniqal values:{s_uniqcode}
counts       :{counts}
    """
        msg2log(None, msg, self.f)
        return s_uniqcode,uniqval,counts

@exec_time
def logMatrix(X:np.array,title:str=None,f:object = None):
    if title is not None:
        msg2log(None,title,f)
    (n,m)=X.shape
    z=np.array([i for i in range(n)])
    z=z.reshape((n,1))
    a=np.append(z,X,axis=1)
    s = '\n'.join([''.join(['{:10.4f}'.format(item) for item in row]) for row in a])
    if f is not None:
        f.write(s)

    return

@exec_time
def plotsequences(d_seqs: dict, title:str=None, folder: str = None, f: object = None):

    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        path_path_png = Path(folder / Path("{}_.png".format(title)))
        path_png = str(path_path_png)
    else:
        path_png = "{}_.png".format(title)

    n = len(d_seqs)
    fig, aax = plt.subplots(n + 1)
    k=0

    for item,data_list in d_seqs.items():
        n_size = len(data_list)
        x = [i for i in range(n_size)]
        aax[k].set_title('{}'.format(item))
        aax[k].plot(x,data_list)

        k=k+1

    plt.savefig(path_png)
    plt.close("all")
    return

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


if __name__ == "__main__":
    logMatrix(np.array([[1,2,3],[4.0,5.0,6.0],[7,8,9],[10,11,12]]))



