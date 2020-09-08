#!/usr/bin/env python3
'''
This module used for allow the Control Plane functionality. It creates the ControlPlane object, checks a mode real time
or static. The members of ControlPlane class are set according by the mode.
In static modes like as "training" or "predicting" will able to use Training Plane or Predicting Plane functionality.
In the auto real time mode the Managment Plane uses together  functionality of the Training Plane and Predicting Plane.
For this, the member of class rtDescriptor of type dictionary is used for data sharing between Control, Training and
Predict Plane.
The internal structures fo this dictionary is following
    rtDescriptor={
    'state':{id:<string, name of state>}
    'csvDataset':<path to dataset csv file>,
    'modelRepository':<path to repository  where trained model saved>
    'typeID':<id of datasource received by GET -request>
    'title':<name of datasource received by GET -request
    'lastTime':<string, last time  in timeseries>
    'misc':<string, free text>

    }

The dictionary is serialized in json format. The path to the dictionary  holds in the configuration file (cfg.py) and
supports in Control Plane object.


'''
import copy
from predictor.Statmodel import tsARIMA
from pathlib import Path
import json
import os
from predictor.utility import msg2log
from pickle import dump, load


class ControlPlane():

    _csv_path = None
    _modes = None
    _actual_mode = None
    _auto_path = None
    _train_path = None
    _predict_path = None
    _control_path = None
    _dt_dset = "Date Time"
    _rcpower_dset = "Imbalance"
    _rcpower_dst_auto = None
    _discret = 10
    _test_cut_off = None
    _val_cut_off = None
    _log_file_name = None
    _stop_on_chart_show = False
    _path_repository = None
    _all_models = {}
    _epochs = 100
    _n_steps = 64
    _n_features = 1
    _units = 32
    _filters = 64
    _kernel_size = 2
    _pool_size = 2
    _hidden_neyrons = 16
    _dropout = 0.2
    _folder_control_log = None
    _folder_train_log   = None
    _folder_predict_log = None
    _folder_auto_log    = None
    _folder_forecast    = None
    _folder_rt_datasets = None
    _folder_descriptor  = None
    _file_descriptor    = None
    _seasonaly_period   = 144
    _predict_lag        = 20
    _max_p              = 5
    _max_q              = 5
    _max_d              = 5

    _scaled_data_4_auto = False
    _start_date_4_auto  = None
    _end_data_4_auto    = None
    _time_trunc_4_auto  = 'hour'
    _geo_limit_4_auto   = None
    _geo_ids_4_auto     = None
    _last_time_auto     = None

    _forecast_number_step = 0


    def __init__(self):
        # log file handlers
        self.fc = None
        self.fp = None
        self.ft = None
        self.fa = None
        self.ff = None
        self.state = None
        self.drtDescriptor = {}
        pass


    # getters/setters
    def set_csv_path(self, val):
        type(self)._csv_path = copy.copy(val)

    def get_csv_path(self):
        return type(self)._csv_path

    csv_path = property(get_csv_path, set_csv_path)

    def set_modes(self, val):
        type(self)._modes = copy.deepcopy(val)

    def get_modes(self):
        return type(self)._modes

    modes = property(get_modes, set_modes)

    def set_actual_mode(self, val):
        type(self)._actual_mode = copy.copy(val)

    def get_actual_mode(self):
        return type(self)._actual_mode

    actual_mode = property(get_actual_mode, set_actual_mode)

    def set_auto_path(self, val):
        type(self)._auto_path = copy.copy(val)

    def get_auto_path(self):
        return type(self)._auto_path

    auto_path = property(get_auto_path, set_auto_path)

    def set_train_path(self, val):
        type(self)._train_path = copy.copy(val)

    def get_train_path(self):
        return type(self)._train_path

    train_path = property(get_train_path, set_train_path)

    def set_predict_path(self, val):
        type(self)._predict_path = copy.copy(val)

    def get_predict_path(self):
        return type(self)._predict_path

    predict_path = property(get_predict_path, set_predict_path)

    def set_control_path(self, val):
        type(self)._control_path = copy.copy(val)

    def get_control_path(self):
        return type(self)._control_path

    control_path = property(get_control_path, set_control_path)

    def set_dt_dset(self, val):
        type(self)._dt_dset = copy.copy(val)

    def get_dt_dset(self):
        return type(self)._dt_dset

    dt_dset = property(get_dt_dset, set_dt_dset)

    def set_rcpower_dset(self, val):
        type(self)._rcpower_dset = copy.copy(val)

    def get_rcpower_dset(self):
        return type(self)._rcpower_dset

    rcpower_dset = property(get_rcpower_dset, set_rcpower_dset)

    def set_rcpower_dset_auto(self, val):
        type(self)._rcpower_dset_auto = copy.copy(val)

    def get_rcpower_dset_auto(self):
        return type(self)._rcpower_dset_auto

    rcpower_dset_auto = property(get_rcpower_dset_auto, set_rcpower_dset_auto)

    def set_discret(self, val):
        type(self)._discret = val

    def get_discret(self):
        return type(self)._discret

    discret = property(get_discret, set_discret)

    def set_test_cut_off(self, val):
        type(self)._test_cut_off = val

    def get_test_cut_off(self):
        return type(self)._test_cut_off

    test_cut_off = property(get_test_cut_off, set_test_cut_off)

    def set_val_cut_off(self, val):
        type(self)._val_cut_off = val

    def get_val_cut_off(self):
        return type(self)._val_cut_off

    val_cut_off = property(get_val_cut_off, set_val_cut_off)

    def set_log_file_name(self, val):
        type(self)._log_file_name = copy.copy(val)

    def get_log_file_name(self):
        return type(self)._log_file_name

    log_file_name = property(get_log_file_name, set_log_file_name)

    def set_stop_on_chart_show(self, val):
        type(self)._stop_on_chart_show = val

    def get_stop_on_chart_show(self):
        return type(self)._stop_on_chart_show

    stop_on_chart_show = property(get_stop_on_chart_show, set_stop_on_chart_show)

    def set_path_repository(self, val):
        type(self)._path_repository = copy.copy(val)

    def get_path_repository(self):
        return type(self)._path_repository

    path_repository = property(get_path_repository, set_path_repository)

    def set_all_models(self, val):
        type(self)._all_models = copy.deepcopy(val)

    def get_all_models(self):
        return type(self)._all_models

    all_models = property(get_all_models, set_all_models)

    def set_epochs(self, val):
        type(self)._epochs = val

    def get_epochs(self):
        return type(self)._epochs

    epochs = property(get_epochs, set_epochs)

    def set_n_steps(self, val):
        type(self)._n_steps = val

    def get_n_steps(self):
        return type(self)._n_steps

    n_steps = property(get_n_steps, set_n_steps)

    def set_n_features(self, val):
        type(self)._n_features = val

    def get_n_features(self):
        return type(self)._n_features

    n_features = property(get_n_features, set_n_features)

    def set_units(self, val):
        type(self)._units = val

    def get_units(self):
        return type(self)._units

    units = property(get_units, set_units)

    def set_filters(self, val):
        type(self)._filters = val

    def get_filters(self):
        return type(self)._filters

    filters = property(get_filters, set_filters)

    def set_kernel_size(self, val):
        type(self)._kernel_size = val

    def get_kernel_size(self):
        return type(self)._kernel_size

    kernel_size = property(get_kernel_size, set_kernel_size)

    def set_pool_size(self, val):
        type(self)._pool_size = val

    def get_pool_size(self):
        return type(self)._pool_size

    pool_size = property(get_pool_size, set_pool_size)

    def set_hidden_neyrons(self, val):
        type(self)._hidden_neyrons = val

    def get_hidden_neyrons(self):
        return type(self)._hidden_neyrons

    hidden_neyrons = property(get_hidden_neyrons, set_hidden_neyrons)

    def set_dropout(self, val):
        type(self)._dropout = val

    def get_dropout(self):
        return type(self)._dropout

    dropout = property(get_dropout, set_dropout)

    def set_folder_control_log(self, val):
        type(self)._folder_control_log = copy.copy(val)

    def get_folder_control_log(self):
        return type(self)._folder_control_log

    folder_control_log = property(get_folder_control_log, set_folder_control_log)

    def set_folder_train_log(self, val):
        type(self)._folder_train_log = copy.copy(val)

    def get_folder_train_log(self):
        return type(self)._folder_train_log

    folder_train_log = property(get_folder_train_log, set_folder_train_log)

    def set_folder_predict_log(self, val):
        type(self)._folder_predict_log = copy.copy(val)

    def get_folder_predict_log(self):
        return type(self)._folder_predict_log

    folder_predict_log = property(get_folder_predict_log, set_folder_predict_log)

    def set_folder_auto_log(self, val):
        type(self)._folder_auto_log = copy.copy(val)

    def get_folder_auto_log(self):
        return type(self)._folder_auto_log

    folder_auto_log = property(get_folder_auto_log, set_folder_auto_log)

    def set_folder_forecast(self, val):
        type(self)._folder_forecast = copy.copy(val)

    def get_folder_forecast(self):
        return type(self)._folder_forecast

    folder_forecast = property(get_folder_forecast, set_folder_forecast)

    def set_folder_rt_datasets(self, val):
        type(self)._folder_rt_datasets = copy.copy(val)

    def get_folder_rt_datasets(self):
        return type(self)._folder_rt_datasets

    folder_rt_datasets = property(get_folder_rt_datasets, set_folder_rt_datasets)

    def set_folder_descriptor(self, val):
        type(self)._folder_descriptor = copy.copy(val)

    def get_folder_descriptor(self):
        return type(self)._folder_descriptor

    folder_descriptor = property(get_folder_descriptor, set_folder_descriptor)

    def set_file_descriptor(self, val):
        type(self)._file_descriptor = copy.copy(val)

    def get_file_descriptor(self):
        return type(self)._file_descriptor

    file_descriptor = property(get_file_descriptor, set_file_descriptor)

    def set_seasonaly_period(self, val):
        type(self)._seasonaly_period = val

    def get_seasonaly_period(self):
        return type(self)._seasonaly_period

    seasonaly_period = property(get_seasonaly_period, set_seasonaly_period)

    def set_predict_lag(self, val):
        type(self)._predict_lag = val

    def get_predict_lag(self):
        return type(self)._predict_lag

    predict_lag = property(get_predict_lag, set_predict_lag)

    def set_max_p(self, val):
        type(self)._max_p = val

    def get_max_p(self):
        return type(self)._max_p

    max_p = property(get_max_p, set_max_p)

    def set_max_q(self, val):
        type(self)._max_q = val

    def get_max_q(self):
        return type(self)._max_q

    max_q = property(get_max_q, set_max_q)

    def set_max_d(self, val):
        type(self)._max_d = val

    def get_max_d(self):
        return type(self)._max_d

    max_d = property(get_max_d, set_max_d)

    def set_scaled_data_4_auto(self, val):
        type(self)._scaled_data_4_auto = val

    def get_scaled_data_4_auto(self):
        return type(self)._scaled_data_4_auto

    scaled_data_4_auto = property(get_scaled_data_4_auto, set_scaled_data_4_auto)

    def set_start_date_4_auto(self, val):
        type(self)._start_date_4_auto = val

    def get_start_date_4_auto(self):
        return type(self)._start_date_4_auto

    start_date_4_auto = property(get_start_date_4_auto, set_start_date_4_auto)

    def set_end_data_4_auto(self, val):
        type(self)._end_data_4_auto = val

    def get_end_data_4_auto(self):
        return type(self)._end_data_4_auto

    end_data_4_auto = property(get_end_data_4_auto, set_end_data_4_auto)

    def set_time_trunc_4_auto(self, val):
        type(self)._time_trunc_4_auto = val

    def get_time_trunc_4_auto(self):
        return type(self)._time_trunc_4_auto

    time_trunc_4_auto = property(get_time_trunc_4_auto, set_time_trunc_4_auto)

    def set_geo_limit_4_auto(self, val):
        type(self)._geo_limit_4_auto = val

    def get_geo_limit_4_auto(self):
        return type(self)._geo_limit_4_auto

    geo_limit_4_auto = property(get_geo_limit_4_auto, set_geo_limit_4_auto)

    def set_geo_ids_4_auto(self, val):
        type(self)._geo_ids_4_auto = val

    def get_geo_ids_4_auto(self):
        return type(self)._geo_ids_4_auto

    geo_ids_4_auto = property(get_geo_ids_4_auto, set_geo_ids_4_auto)

    def set_last_time_auto(self, val):
        type(self)._last_time_auto = val

    def get_last_time_auto(self):
        return type(self)._last_time_auto

    last_time_auto = property(get_last_time_auto, set_last_time_auto)

    def set_forecast_number_step(self, val):
        type(self)._forecast_number_step = val

    def get_forecast_number_step(self):
        return type(self)._forecast_number_step

    forecast_number_step = property(get_forecast_number_step, set_forecast_number_step)
    """
    common Control Plane methods
    """
    def save_descriptor(self):
        folder_descr = Path(self.folder_descriptor)
        Path(folder_descr).mkdir(parents=True, exist_ok=True)

        file_pickle = Path(folder_descr, self.file_descriptor)

        with open(file_pickle,'wb') as outfile:
            # json.dump(self.drtDescriptor, outfile)
            dump(self.drtDescriptor,outfile)
            msg="Descriptor had already serialized in {}".format(file_pickle)
            msg2log(self.save_descriptor.__name__, msg, self.fc)
        with open(file_pickle, 'rb') as infile:
            new_dict=load(infile)
            print(new_dict == self.drtDescriptor)
            pass
        return

    def load_descriptor(self):

        bRet = False
        try:

            file_pickle=Path(self.folder_descriptor) / Path(self.file_descriptor)
            if (not os.path.exists(file_pickle)) or (not os.path.isfile(file_pickle)) :
                msg = "No saved descriptors"
                msg2log(self.load_descriptor.__name__, msg, self.fc)
                return bRet

            with open(file_pickle,'rb') as infile:
                # self.drtDescriptor =json.load(infile)
                self.drtDescriptor = load(infile)
                msg = "Descriptor had been deserialized from {}".format(file_pickle)
                msg2log(self.load_descriptor.__name__, msg, self.fc)
                bRet = True
                self.logDescriptor()
        except:
            pass
        return bRet

    def logDescriptor(self):
        for k,v in self.drtDescriptor.items():
            pass
            if type(v) is dict:
                for k1,v1 in v:
                    msg="{} : {} :{}".format(k,k1,v1)
                    msg2log("",msg,self.f)
            msg=("{} : {}".format( k,v))
            msg2log("", msg, self.fc)

        return


    """
    Control Plane method for time series analysis
  
    """
    def ts_analysis(self,ds ):
        pass
        arima = tsARIMA("control_arima", "tsARIMA", 32, 100, self.fc)
        arima.param =(0, 0, 0, self.max_p, self.max_d, self.max_q, self.predict_lag, self._seasonaly_period, self.discret, ds.df[self.rcpower_dset].values)
        arima.path2modelrepository = self.path_repository
        arima.timeseries_name = self.rcpower_dset
        arima.control_arima()
        arima.ts_analysis()

        return
