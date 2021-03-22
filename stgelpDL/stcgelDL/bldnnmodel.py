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
from stcgelDL.api import prepareLossCharts,getOutputSize
from clustgelDL.kmean import kMeanCluster
from predictor.utility import msg2log,vector_logging, logMatrix

@exec_time
def driveNNmodelBuild(df:pd.DataFrame,  n_pred:int = 3, title:str ='ElHierro', dt_col_name:str="Date Time",
                      endogen_col_name:str="Imbalance", exogen_list:list=None,  labels_name="desired",n_steps:int=32,
                      test_size:int=16, f:object=None):

    sld = SupervisedLearningData(df,title=title,dt_col_name=dt_col_name,shifted_list=[endogen_col_name],
                                 exogen_list=exogen_list,desired_col_name=labels_name, n_steps=n_steps,
                                 test_size=test_size, f=f)
    sld.prepareSLD()
    msg2log(None,sld.__str__(),D_LOGS["control"])
    # train model
    modelMLP=MLP(title="MLP_Imbalance", n_input_size=sld.m, n_output_size=sld.num_classes,f=f)
    modelMLP.epochs=10
    modelMLP.validation_split=0.3
    modelMLP.buildModel()
    modelMLP.compile()
    n,m=sld.X.shape
    # cut_of_test=n-n_pred
    # object_history = model.trainFit(sld.X[:cut_of_test,:],sld.desired[:cut_of_test])
    object_historyMLP = modelMLP.trainFit(sld.X, sld.desired)
    prepareLossCharts(object_historyMLP, folder=D_LOGS["plot"], f=f)

    # modelLSTM = LSTM(title="LSTM_Imbalance", input_dim=n, n_input_size=sld.m, n_output_size=sld.num_classes, f=f)
    #
    # modelLSTM.epochs = 3
    # modelLSTM.validation_split = 0.3
    # modelLSTM.buildModel()
    # modelLSTM.compile()
    # n, m = sld.X.shape
    # object_historyLSTM = modelLSTM.trainFit(sld.X, sld.desired)
    # prepareLossCharts(object_historyLSTM, folder=D_LOGS["plot"], f=f)

    sld.prepareTestSLD()
    modelMLP.test(sld.X_test,sld.desired_test)
    # modelLSTM.test(sld.X_test,sld.desired_test)
    # examine model
    Z=sld.preparePredictSLD(df=sld.df,n_pred=n_pred)
    modelMLP.predict(Z,sld.scalerY)
    # modelLSTM.predict(Z, sld.scalerY)
    log2All()



    return

class SupervisedLearningData():

    def __init__(self, df:pd.DataFrame,title:str='ElHierro', dt_col_name:str="Date Time",shifted_list:list=[],
                 exogen_list:list=[],desired_col_name:str="desired", n_steps:int=32, test_size:int=16, folder:str="",
                 f:object=None):
        pass

        self.df               = df
        self.title            = title
        self.dt_col_name      = dt_col_name
        self.shifted_list     = shifted_list
        self.exogen_list      = exogen_list
        self.desired_col_name = desired_col_name
        self.test_size        = test_size
        self.n_steps          = n_steps
        self.folder           = folder
        self.f                = f
        self.n                = len(self.df)-self.n_steps-self.test_size
        self.m                = len(self.exogen_list) + (self.n_steps+1) * len(self.shifted_list)
        self.X                = None
        self.X_test           = None
        self.desired          = None
        self.desired_test     = None
        self.num_classes      = 4
        self.isScaling        = False
        self.scaler           = MinMaxScaler(feature_range=(0,1))
        self.scalerY          = MinMaxScaler(feature_range=(0, 1))


    def __str__(self):
        message=f"""
Dataset:                   {self.df.__str__}
Title:                     {self.title}
Date Column Name:          {self.dt_col_name}
Lagged Endogenous Columns: {self.shifted_list}
Exogenous Columns:         {self.exogen_list}
Labels Column:             {self.desired_col_name}
Length Of Dataset :        {len(self.df)} 
Train Data Shape :         {self.n} {self.m}
Test Sequence Length:      {self.test_size}
Lag (n_steps):             {self.n_steps}
Scaling:                   {self.isScaling}        

"""
        return message



    """ This method prepares Supervised learning Data (self.X) and desired data (self.desired).
    [ y1[t-n_step], y1[t-n_step+1], ... ,y1[t-1], y1[t], ...,yk[t-n_step], yk[t-n_step+1], ...., yk[t-1],yk[t], x1[t], 
     x2[t],  ,xp[t]] - the t-th row of Supervised Learning data(matrix),
     where y1,...,yk belong to 'shifted' list, x1,...,xp belong to exogenious list.
     SLD matrix and desired labeks (output) may be scaled by MinMaxScaler.
     The size of row vector is n*step *len('shifted' list) + len(exogenious list).
     The number of rows in the matrix self.X is n-n_step. The size of self.desired is n - n_step.
    """
    @exec_time
    def prepareSLD(self):
        if self.X is not None:
            del self.X
            self.X = None
        if self.desired is not None:
            del self.desired
            self.desired = None
        N=len(self.df)-self.test_size
        X=[]
        desired=[]

        for i in range(N):
            end_ix=i +self.n_steps
            if end_ix>N-1:
                break
            seq_x=[]
            for item in self.shifted_list:
                seq_x=seq_x + list(self.df[item][i:end_ix])

            for item in self.exogen_list:
                seq_x=seq_x +list(self.df[item][end_ix:end_ix+1])
            seq_labeled=self.df[self.desired_col_name][end_ix:end_ix+1].values[0]
            X.append(seq_x)
            desired.append(seq_labeled)
        X=np.array(X)
        desired=np.array(desired)
        self.logSLD(title="Train Data ",X=X,wnd_size=32)

        self.num_classes = getOutputSize(df=self.df, desired_col_name=self.desired_col_name, f=D_LOGS["control"])

        if self.isScaling:
            self.scalerY.fit(desired.reshape(-1, 1))
            self.desired = self.scalerY.transform(desired.reshape(-1, 1)).reshape(-1, )
            # self.desired=copy.copy(desired)
            self.scaler.fit(X)
            self.X = self.scaler.transform(X)
        else:
            self.X = X
            self.desired = desired

        (self.n,self.m)=self.X.shape   # update  params
        return

    """ This method prepares testing Supervised learning Data and label data(output) from sources time series, exogenous
       factors and labesls.  
       [ y1[t-n_step], y1[t-n_step+1], ... ,y1[t-1], y1[t], ...,yk[t-n_step], yk[t-n_step+1], ...., yk[t-1],yk[t], x1[t], 
        x2[t],  ,xp[t]] - the t-th row of Supervised Learning data(matrix),
        where y1,...,yk belong to 'shifted' list, x1,...,xp belong to exogenious list.
        The size of row vector is n*step *len('shifted' list) + len(exogenious list).
        The number of rows in the matrix  is defined by self.test_size value.
        The labels (output) vector has self.test_size length 
    """
    @exec_time
    def prepareTestSLD(self):


        N = len(self.df)
        start_ix = N - self.test_size
        X = []
        desired = []

        for i in range(start_ix,N):
            end_ix = i + self.n_steps
            if end_ix > N - 1:
                break
            seq_x = []
            for item in self.shifted_list:
                seq_x = seq_x + list(self.df[item][i:end_ix])

            for item in self.exogen_list:
                seq_x = seq_x + list(self.df[item][end_ix:end_ix+1])
            seq_labeled = self.df[self.desired_col_name][end_ix:end_ix+1].values[0]
            X.append(seq_x)
            desired.append(seq_labeled)
        X = np.array(X)
        desired = np.array(desired)
        self.logSLD(title="Test Data ", X=X, wnd_size=32)

        # scalerY and scaler are already fitted
        if self.isScaling:
            self.desired_test = self.scalerY.transform(desired.reshape(-1, 1)).reshape(-1, )
            self.X_test = self.scaler.transform(X)
        else:
            self.X_test = X
            self.desired_test = desired

        (n, m) = self.X_test.shape  # check params
        return

    """ This method prepares predicting Supervised learning Data  Z from sources time series and exogenous factors.
        [ y1[t-n_step], y1[t-n_step+1], ... ,y1[t-1], y1[t], ...,yk[t-n_step], yk[t-n_step+1], ...., yk[t-1],yk[t], x1[t], 
        x2[t],  ,xp[t]] - the t-th row of Supervised Learning data(matrix),
        where y1,...,yk belong to 'shifted' list, x1,...,xp belong to exogenious list.
        The size of row vector is n*step *len('shifted' list) + len(exogenious list).
        The number of rows in the matrix Z is n_pred """
    @exec_time
    def preparePredictSLD(self, df:pd.DataFrame=None, n_pred: int = 1) -> np.array:

        if df is None:
            return None

        N = len(df)
        start_ix = N - n_pred-self.n_steps
        X = []


        for i in range(start_ix, N):
            end_ix = i + self.n_steps
            if end_ix > N - 1:
                break
            seq_x = []
            for item in self.shifted_list:
                seq_x = seq_x + list(self.df[item][i:end_ix])

            for item in self.exogen_list:
                seq_x = seq_x + list(self.df[item][end_ix:end_ix+1])

            X.append(seq_x)

        X = np.array(X)

        self.logSLD(title="Predict Data ", X=X, desired=None, wnd_size=32)

        # scalerY and scaler are already fitted
        if self.isScaling:

            Z = self.scaler.transform(X)
        else:
            Z = X


        (n, m) = Z.shape  # check params
        return Z


    @exec_time
    def preparePredictSLD_(self, n_pred: int = 1) -> np.array:

        Z = np.zeros((n_pred, self.m), dtype=float)

        for i in range(n_pred):
            i1 = i + self.n_steps
            k = 0
            for item in self.shifted_list:
                for j in range(self.n_steps + 1):
                    Z[i, k] = self.df[item][i1 - j]
                    k += 1
            for item in self.exogen_list:
                Z[i, k] = self.df[item][i1]
                k += 1
        logMatrix(Z, title="Predict Data", f=D_LOGS["control"])
        msg2log(None, "\n\n", f=D_LOGS["control"])
        logMatrix(Z, title="Predict Data", f=D_LOGS["predict"])
        msg2log(None, "\n\n", f=D_LOGS["predict"])
        # without scaler
        if True:
            Zsc = self.scaler.transform(Z)
            logMatrix(Zsc, title="Scaled Predict Data", f=D_LOGS["control"])
            msg2log(None, "\n\n", f=D_LOGS["control"])
            logMatrix(Zsc, title="Scaled Predict Data", f=D_LOGS["predict"])
            msg2log(None, "\n\n", f=D_LOGS["predict"])
        else:
            Zsc = Z
        return Zsc

    def logSLD(self,title:str="Train Data", X:np.array=None, desired:np.array=None, wnd_size:int=32):
        if X is None:
            return
        (n, m) = X.shape
        msg2log(None, "{} (input) Shape: ({},{})".format(title, n, m), f=D_LOGS["control"])
        logMatrix(X[:wnd_size, :], title="{} (first {} vectors)".format(title,wnd_size), f=D_LOGS["control"])
        msg2log(None, "\n\n", f=D_LOGS["control"])
        msg="\n\n{} first {} labels\n".format(title,wnd_size)
        if desired is not None:
            for i in range(wnd_size):
                msg=msg + "{:<2d} ".format(desired[i])
            msg2log(None,msg,D_LOGS["control"])

        logMatrix(X[n - wnd_size + 1:, :], title="{} (last {} vectors)".format(title,wnd_size), f=D_LOGS["control"])
        msg = "\n\n{} last {} labels\n".format(title, wnd_size)
        if desired is not None:
            for i in range(n-wnd_size+1,n):
                msg = msg + "{:<2d} ".format(desired[i])
            msg2log(None, msg, D_LOGS["control"])

        msg2log(None, "\n\n{} last row".format(title), f=D_LOGS["control"])
        msg2log(None, "{} {} {} ... {} {} {}\n".format(X[n - 1, 0], X[n - 1, 1], X[n - 1, 2], X[n - 1, m - 3],
                                                       X[n - 1, m - 2], X[n - 1, m - 1]), f=D_LOGS["control"])
        return




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
        self.hidden_neuron_number = 48
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
            self.model.summary()

            if self.f is not None:
                self.f.write('\n{}  \n'.format(self.title))
                self.model.summary(print_fn=lambda x: self.f.write(x + '\n'))

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
                message="{:<10d} {:<10d}".format(i, self.d_pred[i])
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

    def test(self,X_test:np.array,desired_test:np.array):
        if self.model is None or self.status < 2:
            msg2log(type(self).__name__, "The {} model is not feed".format(self.title), self.f)
        n_pred,m=X_test.shape
        msg = ""
        try:
            predict = self.model.predict(X_test)
            message = "{} model test\n{}".format(self.model.name, predict)
            msg2log(None, message, D_LOGS["predict"])
            self.status = 3
            message="\nState Testing by {} model\n{:<10s} {:<10s} {:<10s}".format(self.model.name,"Period",
                                                                                  "State label","Desired label")
            msg2log(None,message,D_LOGS["predict"])
            for i in range(len(predict)):
                self.d_pred[i]=np.argmax(predict[i])
                #inv=scaler.inverse_transform(self.pred[i])
                message="{:<10d} {:<10d} {:<10d}".format(i, self.d_pred[i],desired_test[i])
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
        self.model.add(layers.Dense(self.hidden_neuron_number, activation='tanh',name='Layer_2'))
        # self.model.add(layers.Dropout(self.dropout_factor1, name='Layer_3'))
        self.model.add(layers.Dense(self.hidden_neuron_number, activation='tanh',name='Layer_4'))
        self.model.add(layers.Dense(self.hidden_neuron_number, activation='tanh',name='Layer_5'))
        # self.model.add(layers.Dense(16, activation='relu', name='Layer_6'))
        self.model.add(layers.Dense(self.n_output_size,  activation='softmax',name='Layer_output'))
        # self.model.add(layers.Softmax(name='Probability-predictions'))
        self.status=0
        return

class LSTM(NNmodel):

    def __init__(self, title: str = "LSTM", status: int = -1, input_dim:int=128, n_input_size: int = 16,
                 n_output_size: int = 10, f: object = None):
        self.input_dim=input_dim
        super().__init__(title=title, status=status, n_input_size=n_input_size, n_output_size=n_output_size, f=f)

    def buildModel(self):
        self.model = Sequential(name=self.title)

        self.model.add(layers.LSTM(64, activation='relu', input_shape=(self.n_input_size, 1), name='Layer_0'))
        self.model.add(layers.Dense(self.n_output_size, activation='softmax',name='Layer_1'))



        # self.model.add(layers.Embedding(self.input_dim, 1, input_length=self.n_input_size,
        #                                 name='Layer_Embedding'))
        # self.model.add(layers.LSTM(64))
        #
        # self.model.add(layers.LSTM(self.hidden_neuron_number, activation='relu',dropout=0.3,
        #                            recurrent_activation='sigmoid',use_bias=True,return_sequences=True))
        #
        # self.model.add(layers.LSTM(self.hidden_neuron_number,dropout=0.3,return_sequences=True))
        # self.model.add(layers.LSTM(64,dropout=0.2,activation='relu'))
        # self.model.add(layers.Dense(self.hidden_neuron_number, activation='tanh', input_dim=64,
        #                                 name='Layer_2'))


        # self.model.add(layers.Dense(self.n_output_size, activation='softmax', name='Layer_output'))
        # self.model.add(layers.Softmax(name='Probability-predictions'))
        self.status = 0
        return






