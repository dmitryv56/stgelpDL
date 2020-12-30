#!/usr/bin/python3

""" This module contains a functions and classes for input data for classification NN modeles creating, models training
and the state (situations) forecasting on base them.

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
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy,KLDivergence,MeanAbsoluteError,MeanSquaredError,\
                                     MeanSquaredLogarithmicError,CosineSimilarity
from tensorflow.keras.metrics import SparseCategoricalAccuracy,Accuracy,BinaryAccuracy,BinaryCrossentropy,\
    CosineSimilarity,KLDivergence,MeanAbsoluteError,MeanSquaredError,SparseCategoricalCrossentropy

from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from stcgelDL.cfg import GlobalConst
from stcgelDL.api import prepareLossCharts
from clustgelDL.kmean import kMeanCluster
from predictor.utility import msg2log,vector_logging

@exec_time
def driveNNmodelBuild(df:pd.DataFrame,  n_pred:int = 3, title:str ='ElHierro', dt_col_name:str="Date Time",
                      endogen_col_name:str="Imbalance", exogen_list:list=None,  labels_name="desired",lag:int=32,
                      f:object=None):

    sld = SupervisedLearningData(df,title=title,dt_col_name=dt_col_name,shifted_list=[endogen_col_name],
                                 exogen_list=exogen_list,desired_col_name=labels_name, lag=lag, f=f)
    sld.prepareSLD()
    # train model
    model=MLP(title="MLP_Imbalance", n_input_size=sld.m, n_output_size=sld.num_classes,f=f)
    model.epochs=32
    model.validation_split=0.5
    model.buildModel()
    model.compile()
    n,m=sld.X.shape
    cut_of_test=n-n_pred
    object_history = model.trainFit(sld.X[:cut_of_test,:],sld.desired[:cut_of_test])
    prepareLossCharts(object_history, folder=D_LOGS["plot"], f=f)

    # examine model
    Z=sld.preparePredictSLD(n_pred=n_pred)
    model.predict(Z,sld.scalerY)
    log2All()



    return

class SupervisedLearningData():

    def __init__(self, df:pd.DataFrame,title:str='ElHierro', dt_col_name:str="Date Time",shifted_list:list=[],
                 exogen_list:list=[],desired_col_name:str="desired", lag:int=32, folder:str="", f:object=None):
        pass

        self.df               = df
        self.title            = title
        self.dt_col_name      = dt_col_name
        self.shifted_list     = shifted_list
        self.exogen_list      = exogen_list
        self.desired_col_name = desired_col_name
        self.n_steps          = lag
        self.folder           = folder
        self.f                = f
        self.n                = len(self.df)-self.n_steps
        self.m                = len(self.exogen_list) + self.n_steps * len(self.shifted_list)
        self.X                = None
        self.desired          = None
        self.num_classes      = 4
        self.scaler           = MinMaxScaler(feature_range=(0,1))
        self.scalerY          = MinMaxScaler(feature_range=(0, 1))

    """ This method prepares Supervised learning Data from sources time series (self.X) and desired data (self.desired).
    For TS from shifted list the lag values are added
    [ y1[t-n_step], y1[t-n_step+1], ... ,y1[t-1], y1[t], ...,yk[t-n_step], yk[t-n_step+1], ...., yk[t-1],yk[t], x1[t], 
     x2[t],  ,xp[t]] - the t-th row of Superviswd Learning data(matrix),
     where y1,...,yk belong to 'shifted' list, x1,...,xp belong to exogenious list.
     SLD matrix scaled by MinMaxScaler
     The size of row vector is n*step *len('shifted' list) + len(exogenious list).
     The number of rows in the matrix self.X is n-n_step. The size of self.desired is n_n_step
    """
    @exec_time
    def prepareSLD(self):

        if self.X is not None:
            del self.X
            self.X=None
        if self.desired is not None:
            del self.desired
            self.desired = None

        X=np.zeros((self.n,self.m),dtype=float)
        desired=np.zeros((self.n),dtype=float)
        for i in range (self.n):
            i1=i+self.n_steps-1
            k=0
            for item in self.shifted_list:
                for j in range(self.n_steps):
                    X[i,k]=self.df[item][i1-j]
                    k+=1
            for item in self.exogen_list:
                X[i,k]=self.df[item][i1]
                k+=1

            desired[i] = self.df[self.desired_col_name] [i1]
        labels = np.unique(np.array(self.df[self.desired_col_name]))
        self.num_classes, = labels.shape
        self.scalerY.fit(desired.reshape(-1,1))
        self.desired=self.scalerY.transform(desired.reshape(-1,1)).reshape(-1,)
        # self.desired=copy.copy(desired)
        self.scaler.fit(X)
        self.X = self.scaler.transform(X)
        return

    """ This method prepares predicting Supervised learning Data from sources time series Z .
    For TS from shifted list the lag values are added
    [ y1[t-n_step], y1[t-n_step+1], ... ,y1[t-1], y1[t], ...,yk[t-n_step], yk[t-n_step+1], ...., yk[t-1],yk[t], x1[t], 
     x2[t],  ,xp[t]] - the t-th row of Superviswd Learning data(matrix),
     where y1,...,yk belong to 'shifted' list, x1,...,xp belong to exogenious list.
     The size of row vector is n*step *len('shifted' list) + len(exogenious list).
     The number of rows in the matrix Z is n_pred """
    @exec_time
    def preparePredictSLD(self,n_pred:int=1)->np.array:

        Z=np.zeros((n_pred,self.m), dtype=float)

        for i in range(n_pred):
            i1 = i + self.n_steps - 1
            k = 0
            for item in self.shifted_list:
                for j in range(self.n_steps):
                    Z[i, k] = self.df[item][i1-j]
                    k += 1
            for item in self.exogen_list:
                Z[i, k] = self.df[item][i1]
                k += 1
        Zsc=self.scaler.transform(Z)
        return Zsc


class NNmodel():

    def __init__(self,title:str="", status:int=-1, n_input_size:int=16, n_output_size:int=10, f:object=None):
        self.title                = title
        self.status               = status
        self.f                    = f
        self.model                = None
        self.epochs               = 100
        self.optimizer            = 'adam'
        # self.metrics              = ['accuracy']
        self.metrics              = [tf.keras.metrics.SparseCategoricalCrossentropy(),
                                     tf.keras.metrics.KLDivergence(),
                                     tf.keras.metrics.MeanAbsoluteError(),
                                     tf.keras.metrics.MeanSquaredError()]

        self.loss                 = {0:tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                     1:tf.keras.losses.KLDivergence(),
                                     2:tf.keras.losses.MeanAbsoluteError(),
                                     3:tf.keras.losses.MeanSquaredError(),
                                     4:tf.keras.losses.MeanSquaredLogarithmicError(),
                                     5:tf.keras.losses.CosineSimilarity()
                                     }
        self.validation_split     = 0.3
        self.n_input_size         = n_input_size
        self.n_output_size        = n_output_size
        self.hidden_neuron_number = 128
        self.dropout_factor1      = 0.1
        self.dropout_factor2      = 0.2
        self.d_pred               = {}

    @exec_time
    def compile(self, loss_key:int=1):
        if self.model is None:
            msg2log(self.__name__,"The {} model is not defined".format(self.title),self.f)
        msg=""
        try:
            msg2log("Compile","loss is {}\nmetrics are {}".format(self.loss[loss_key], self.metrics),self.f)
            self.model.compile(optimizer=self.optimizer, loss=self.loss[loss_key], metrics=self.metrics)
            self.status = 1
        except ValueError as e:
            msg = "O-o-ops! I got a ValueError - reason  {}\n{}".format(str(e),sys.exc_info())
        except :
            msg = "O-o-ops! I got an unexpected error - reason  {}\n".format(sys.exc_info())
        finally:
            if len(msg)>0:
                msg2log("compile",msg,D_LOGS["except"])
                log2All()
        return

    @exec_time
    def trainFit(self,X_train:np.array, Y_train:np.array)->object:
        if self.model is None or self.status<1:
            msg2log(self.__name__, "The {} model is not compiled".format(self.title), self.f)
        msg=""
        history=None
        try:
            history = self.model.fit(X_train,Y_train,epochs=self.epochs,validation_split=self.validation_split)
            message="{} model History\n{}".format(self.model.name,history.history)
            msg2log(None,message,D_LOGS["train"])
            self.status=2
        except ValueError as e:
            msg = "O-o-ops! I got a ValueError - reason  {}\n{}".format(str(e),sys.exc_info())
        except RuntimeError as e:
            msg = "O-o-ops! I got a RuntimeError - reason  {}\n{}".format(str(e),sys.exc_info())
        except :
            msg = "O-o-ops! I got an unexpected error - reason  {}\n".format(sys.exc_info())
        finally:
            if len(msg)>0:
                msg2log("trainFit",msg,D_LOGS["except"])
                log2All()
        return history

    def predict(self, X:np.array, scaler:MinMaxScaler):
        if self.model is None or self.status < 2:
            msg2log(type(self).__name__, "The {} model is not feed".format(self.title), self.f)
        n_pred,m=X.shape
        msg = ""
        try:
            predict = self.model.predict(X)
            message = "{} model predict\n{}".format(self.model.name, predict)
            msg2log(None, message, D_LOGS["predict"])
            self.status = 3
            message="\nState Predict by {} model\n{:<10s} {:<10s}".format(self.model.name,"Period","State label")
            msg2log(None,message,D_LOGS["predict"])
            for i in range(len(predict)):
                self.d_pred[i]=np.argmax(predict[i])
                #inv=scaler.inverse_transform(self.pred[i])
                message="{:>10d} {:<10.4f}".format(i, self.d_pred[i])
                msg2log(None, message, D_LOGS["predict"])

        except ValueError as e:
            msg = "O-o-ops! I got a ValueError - reason  {}\n{}".format(str(e), sys.exc_info())
        except RuntimeError as e:
            msg = "O-o-ops! I got a RuntimeError - reason  {}\n{}".format(str(e), sys.exc_info())
        except:
            msg = "O-o-ops! I got an unexpected error - reason  {}\n".format(sys.exc_info())
        finally:
            if len(msg) > 0:
                msg2log("predict", msg, D_LOGS["except"])
                log2All()

        return





class MLP(NNmodel):

    def __init__(self, title:str="MLP", status:int=-1,n_input_size:int=16, n_output_size:int=10,f:object=None):

        super().__init__(title=title,status=status,n_input_size=n_input_size, n_output_size=n_output_size,f=f)

    def buildModel(self ):

        self.model = Sequential(name=self.title)
        # model.add(tf.keras.Input(shape=( n_steps,1)))
        self.model.add(layers.Dense(self.hidden_neuron_number, activation='tanh', input_dim=self.n_input_size,
                                    name='Layer_0'))

        # self.model.add(layers.Dropout(self.dropout_factor2, name='Layer_1'))
        self.model.add(layers.Dense(64, activation='tanh',name='Layer_2'))
        # self.model.add(layers.Dropout(self.dropout_factor1, name='Layer_3'))
        self.model.add(layers.Dense(32, activation='relu',name='Layer_4'))
        self.model.add(layers.Dense(16, activation='relu',name='Layer_5'))
        self.model.add(layers.Dense(self.n_output_size,  name='Layer_output'))
        self.model.add(layers.Softmax(name='Probability-predictions'))
        self.status=0
        return






