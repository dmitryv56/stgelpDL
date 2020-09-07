#!/usr/bin/python3


import pandas as pd
from datetime import timedelta
import copy

from predictor.utility import tsBoundaries2log, tsSubset2log, dataset_properties2log,exec_time
from predictor.api import get_scaler4train, scale_sequence, TimeSeries2SupervisedLearningData

class Dataset():

    df       = None      # pandas DataFrame
    df_train = None
    df_val   = None
    df_test  = None

    _data_for_predict = None
    _predict_date = None

    def __init__(self, csv_path, dt_dtset, rcpower_dset, discret, f = None):

        self.csv_path = csv_path
        self.dt_dset = dt_dtset
        self.rcpower_dset = rcpower_dset
        self.discret = discret
        self.f = f
        self.simple_bias = 0
        self.simple_scale =1

        self.test_cut_off=60
        self.val_cut_off=600
        self.scaler = None
        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.rcpower = None




    def __str__(self):
        s = 'csv-file:' + str(self.csv_path)  +  '\n Date Time : ' + self.dt_dset + "  Time Series Name : " \
            + self.rcpower_dset + "\n discretization: " + str(self.discret)
        if self.f is not None:
            self.f.write(s)
        return s


    # getter setter
    def get_data_for_predict(self):
        return type(self)._data_for_predict

    def set_data_for_predict(self, n_len):
        type(self)._data_for_predict = copy.copy(self.rcpower[:n_len])
    data_for_predict = property(get_data_for_predict, set_data_for_predict)

    def set_predict_date(self, date_time = None):
        type(self)._predict_date = self.df[self.dt_dset].max() + timedelta(self.discret) if date_time is None else date_time

    def get_predict_date(self):
        return type(self)._predict_date

    predict_date = property(get_predict_date, set_predict_date)

    @exec_time
    def readDataSet(self ):
        """

        self.csv_path :  csv -file was made by excel export.It can contain data on many characteristics. For time series ,
                            we need data about date and time and actual data about some feature, like as Imbalance in the
                            power grid. If we consider cvs-dataset as a matrix, then the first row or header contains the
                            names of the columns. The samples must de equidistance
        self.dt_dset:     name of date/time column
        self.rcpower_dset:name of actual characteristic.
        self.discret :          discretization, this used for logging
        self.f :                log file handler
        :return:            df -pandas DataFrame object
        """
        self.df = pd.read_csv(self.csv_path)
        self.df.head()

        # %%time   T.B.D.

        # This code is copied from https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba
        # with a few minor changes.
        #


        self.df[self.rcpower_dset] = pd.to_numeric(self.df[self.rcpower_dset], errors='coerce')
        # for i in range(490):
        #    self.df[rcpower_dset][i]=i
        self.df = self.df.dropna(subset=[self.rcpower_dset])

        self.df[self.dt_dset] = pd.to_datetime(self.df[self.dt_dset], dayfirst=True)

        self.df = self.df.loc[:, [self.dt_dset, self.rcpower_dset]]
        self.df.sort_values(self.dt_dset, inplace=True, ascending=True)
        self.df = self.df.reset_index(drop=True)

        # simple scaling and biasing of the time series
        self.df[self.rcpower_dset] *= self.simple_scale
        self.df[self.rcpower_dset]+=self.simple_bias


        self.df.info()
        self.df.head(10)
        title ='Number of rows and columns after removing missing values'
        tsBoundaries2log(title, self.df, self.dt_dset, self.rcpower_dset, self.f)

        return
    """
    This method is used for split dataset on 2 (or 3) subdatasets : training, evaliation (and testing)
    """

    @exec_time
    def set_train_val_test_sequence(self):
        """

        self.df: DataFrame object
        self.dt_dset: - date/time  header name, i.e. "Date Time"
        self.rcpower_dset: - actual characteristic header name, i.e. "Imbalance"
        self.test_cut_off: - value to pass time delta value in the 'minutes' resolution or None, like as
                               'minutes=<value>.' NOte: the timedelta () function does not accept string as parameter, but
                               as value timedelta(minutes=value)
                               The last sampled values before time cutoff represent the test sequence.
        self.val_cut_off: -  value to pass time delta value in the 'minutes' resolution or None, like as
                               'minutes=<value>.'
                               The last sampled values before the test sequence.

        self.f:            - log file hadler
        :return:
        """
        if self.test_cut_off is None or self.test_cut_off == "":
            test_cutoff_date = self.df[self.dt_dset].max()
            self.df_test = None
        else:
            test_cutoff_date = self.df[self.dt_dset].max() - timedelta(minutes=self.test_cut_off)
            self.df_test = self.df[self.df[self.dt_dset] > test_cutoff_date]

        if self.val_cut_off is None or self.val_cut_off == "":
            self.df_val = None
        else:
            val_cutoff_date = test_cutoff_date - timedelta(minutes=self.val_cut_off)
            self.df_val = self.df[(self.df[self.dt_dset] > val_cutoff_date) & (self.df[self.dt_dset] <= test_cutoff_date)]

        self.df_train = self.df[self.df[self.dt_dset] <= val_cutoff_date]

        tsSubset2log(self.dt_dset, self.rcpower_dset, self.df_train, self.df_val, self.df_test, self.f)

        datePredict = self.df_test[self.dt_dset].values[0]
        actvalPredict = self.df_test[self.rcpower_dset].values[0]

        return  datePredict, actvalPredict

    @exec_time
    def dset2arrays(self, n_steps, n_features, n_epochs):


        dataset_properties2log(self.csv_path, self.dt_dset, self.rcpower_dset, self.discret, self.test_cut_off, \
                               self.val_cut_off, n_steps, n_features,n_epochs, self.f)
        # read dataset
        df = self.readDataSet()

        # set training, validation and test sequence
        datePredict, actvalPredict = self.set_train_val_test_sequence()


        # scaling time series
        self.scaler, rcpower_scaled, rcpower = get_scaler4train(self.df_train, self.dt_dset, self.rcpower_dset, self.f)

        rcpower_val_scaled,  rcpower_val  = scale_sequence(self.scaler, self.df_val,  self.dt_dset, self.rcpower_dset, self.f)
        rcpower_test_scaled, rcpower_test = scale_sequence(self.scaler, self.df_test, self.dt_dset, self.rcpower_dset, self.f)

        # time series is transformed to supevised learning data
        self.X,      self.y      = TimeSeries2SupervisedLearningData( rcpower_scaled,      n_steps, self.f )
        self.X_val,  self.y_val  = TimeSeries2SupervisedLearningData( rcpower_val_scaled,  n_steps, self.f )
        self.X_test, self.y_test = TimeSeries2SupervisedLearningData( rcpower_test_scaled, n_steps, self.f )

        self.rcpower = copy.copy(rcpower)

        return





