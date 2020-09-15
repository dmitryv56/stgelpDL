#!/usr/bin/python3

import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Statmodel import tsARIMA
from NNmodel import MLP, LSTM, CNN

from pathlib import Path
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pickle import dump, load
import copy

from datetime import timedelta
from utility import msg2log, chunkarray2log, svld2log, vector_logging, shift,exec_time, PlotPrintManager
from time import sleep

def chart_MAE(name_model, name_time_series, history, n_steps, logfolder, stop_on_chart_show=False):
    # Plot history: MAE


    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    fig, ax = plt.subplots()

    ax.plot(history.history['loss'], marker='', label='MAE (training data)', color=palette(0))
    ax.plot(history.history['val_loss'], marker='', label='MAE (validation data)', color=palette(1))

    plt.legend(loc=2, ncol=2)
    ax.set_title('Mean Absolute Error{} {} (Time Steps = {})'.format(name_model, name_time_series, n_steps))
    ax.set_xlabel("No. epoch")
    ax.set_ylabel("MAE value")
    plt.show(block=stop_on_chart_show)


    if logfolder is not None:
        plt.savefig("{}/MAE_{}_{}_{}.png".format(logfolder, name_model, name_time_series, n_steps))
    if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")
    return


def chart_MSE(name_model, name_time_series, history, n_steps, logfolder, stop_on_chart_show=False):
    # Plot history: MSE



    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    fig, ax = plt.subplots()

    ax.plot(history.history['mean_squared_error'],     marker='', label='MSE (training data)',   color=palette(0))
    ax.plot(history.history['val_mean_squared_error'], marker='', label='MSE (validation data)', color=palette(1))
    plt.legend(loc=2, ncol=2)
    ax.set_title('Mean Square Error  {} {} (Time Steps = {})'.format(name_model, name_time_series, n_steps))
    ax.set_xlabel("No. epoch")
    ax.set_ylabel("MSE value")
    plt.show(block=stop_on_chart_show)


    if logfolder is not None:
        plt.savefig("{}/MSE_{}.png".format(logfolder, name_model, name_time_series, n_steps))
    if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")
    return


def chart_2series(df, title, Y_label, dt_dset, array_pred, array_act, n_pred, logfolder, stop_on_chart_show=False):
    """

    :param df: - pandas dataset that contains of time series
    :param title: -title for chart
    :param Y_label: -time series name, i.e. rcpower_dset for df
    :param dt_dset:  - date/time, i.e.dt_dset dor df
    :param array_pred: - predict numpy vector
    :param array_act:  - actual values numpy vector
    :param n_pred:     - length of array_pred and array_act
    :param logfolder   - log folder path
    :param stop_on_chart_show: True if stop on the chart show and wait User' action
    :return:
    """

    times = mdates.drange(df[dt_dset][len(df[dt_dset]) - n_pred].to_pydatetime(),
                          df[dt_dset][len(df[dt_dset]) - 1].to_pydatetime(),
                          timedelta(minutes=10))  # m.b import timedate as td; td.timedelta(minutes=10)

    plt.plot(times, array_pred, label='Y pred')

    plt.plot(times, array_act, label='Y act')
    title_com='{} (Length of series   {})'%(title, n_pred)
    plt.title(title_com)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    plt.gca().xaxis.set_label_text("Date Time")
    plt.gca().yaxis.set_label_text(Y_label)

    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show(block=False)


    title.replace(" ", "_")
    if logfolder is not None:
        plt.savefig("{}/{}-{}_samples.png".format(logfolder, title, n_pred))
    if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")
    return


def chart_predict(dict_predict, n_predict, cp, ds, title, Y_label ):
    """

    :param dict_predict:    {act.model name>:<pred. vector>}
    :param n_predict:
    :param cp:
    :param ds:
    :param title:
    :param Y_label:
    :return:
    """

    # times = mdates.drange(df[dt_dset][len(df[dt_dset]) - n_pred].to_pydatetime(),
    #                       df[dt_dset][len(df[dt_dset]) - 1].to_pydatetime(),
    #                       timedelta(minutes=10))  # m.b import timedate as td; td.timedelta(minutes=10)
    # times = mdates.drange (ds.df[ds.dt_dset][len(ds.df[ds.dt_dset])].to_pydatetime(),
    #                        ds.df[ds.dt_dset][len(ds.df[ds.dt_dset] + n_predict)].to_pydatetime(),
    #                                    timedelta(minutes=cp.discret))
    times = mdates.drange( (ds.df[ds.dt_dset][len(ds.df[ds.dt_dset]) - 1] + timedelta( minutes = cp.discret )).to_pydatetime(),
                           (ds.df[ds.dt_dset][len(ds.df[ds.dt_dset]) - 1] + timedelta( minutes = cp.discret * (n_predict+1) )).to_pydatetime(),
                           timedelta(minutes=cp.discret))
    ndelta =  len(times)-n_predict

    msg="\nlen(times)={} n_predict={}\n".format(len(times),n_predict)
    msg2log(chart_predict.__name__, msg, None)

    try:
        for k,vector in dict_predict.items():

            plt.plot(times[ndelta:], vector, label=k)

        # sfile = "predict_{}.png".format(title.replace(' ', '_'))
        # sFolder = PlotPrintManager.get_PredictLoggingFolder()
        # filePng = Path(sFolder) / (sfile)
        # plt.savefig(filePng)
        # if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")

    except ValueError:
        msg="\nOoops! That was not valid value\n {}\n{}".format(sys.exc_info()[0],sys.exc_info()[1])
        msg2log(chart_predict.__name__, msg,cp.fp)
        plt.close("all")
        return
    except:
        msg = "\nOoops! Unexpected error: {}\n{}\n}".format(sys.exc_info()[0],sys.exc_info()[1])
        msg2log(chart_predict.__name__, msg, cp.fp)
        plt.close("all")
        return
    # plt.plot(times, array_pred, label='Y pred')
    #
    # plt.plot(times, array_act, label='Y act')

    # plt.title(title)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=cp.discret))
    plt.gca().xaxis.set_label_text("Date Time")
    plt.gca().yaxis.set_label_text(Y_label)

    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show(block=cp.stop_on_chart_show)

    # title.replace(" ", "_")
    # if cp.folder_predict_log is not None:
    #     plt.savefig("{}/{}-{}_steps_{}.png".format(cp.folder_forecast, title, n_predict, cp.forecast_number_step))
    # if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")

    sfile = "predict_{}.png".format(title.replace(' ', '_'))
    sFolder = PlotPrintManager.get_PredictLoggingFolder()
    filePng = Path(sFolder) / (sfile)
    plt.savefig(filePng)
    if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")
    return


"""
Prints the table of forecasts where the columns are actual model names and rows are forecasted values at t+1, t+2,...

"""

def tbl_predict(dict_predict, n_predict,cp, ds, title):
    """

    :param dict_predict: the dictionary with key is actual model name and value is a vector of predicted values, i.e.
                         {<act/model name>:<predict array>, <act.model name>: <predict array>, ... }
    :param n_predict: number of predicts. There is a length of predict array in dict_predict
    :param cp:  -ControlPlane object
    :param ds:  - dataset object
    :param title: - title, i.e. "Imbalance forecasting"
    :return:
    """

    n_row = n_predict
    n_col = len(dict_predict)
    head_list = []
    atemp = np.zeros((n_row, n_col), dtype=np.float32)
    i = j = 0
    for k, v in dict_predict.items():
        head_list.append(k)
        i = 0
        for elem in range(len(v)):
            atemp[i][j] = v[i]
            i += 1
        j += 1
    #   print

    ds.predict_date = ds.df[ds.dt_dset].max()                            # set predict date on the last sample into dataset
    dt=ds.predict_date + timedelta(minutes=cp.discret)  # first predict date/time
    date_time = dt.strftime("%Y-%m-%d %H:%M")

    print('Predict step {}\n\n'.format(cp.forecast_number_step))
    print(str(title).center(80))

    if cp.ff is not None:
        cp.ff.write('\n\n')
        cp.ff.write('Predict step {}\n\n'.format(cp.forecast_number_step))
        cp.ff.write(str(title).center(80))
        cp.ff.write('\n')
        sprt="Date Time".ljust(18)
        cp.ff.write(sprt)
        s=sprt
        for elem in range(len(head_list)):
            sprt = head_list[elem].center(18)
            s = s + sprt
            cp.ff.write(sprt)

        cp.ff.write('\n')
        print('{}\n'.format(s))


        for i in range(atemp.shape[0]):

            sprt = date_time.ljust(18)
            cp.ff.write(sprt)
            s = sprt
            for j in range(atemp.shape[1]):
                sprt = "{0:18.3f}".format(atemp[i][j])
                cp.ff.write(sprt)
                s = s + sprt
            cp.ff.write('\n')
            print(s)
            dt = dt + timedelta(minutes=cp.discret)
            date_time = dt.strftime("%Y-%m-%d %H:%M")

    return

########################################################################################################################
"""
Data scaling functions.  MinMaxScaler object is used
    Xstd = (X - X.min(axis=0))/(X.max(axs=0) -X.min(axis=0))
    Xscaled = Xstd*(max-min) + min
This  get_scaler4train ()function  creates  MinMaxScaler object over train sequence and scales it.
"""


def get_scaler4train(df_train, dt_dset, rcpower_dset, f=None):
    """

    :param df_train:
    :param dt_dset:
    :param rcpower_dset:
    :param f:
    :return:
    """

    rcpower = df_train[rcpower_dset].values

    # Scaled to work with Neural networks.
    scaler = MinMaxScaler(feature_range=(0, 1))
    rcpower_scaled = scaler.fit_transform(rcpower.reshape(-1, 1)).reshape(-1, )

    # train and scaled train data logging
    chunkarray2log("Train data array", rcpower, 8, f)
    chunkarray2log("Scaled Train data", rcpower_scaled, 8, f)

    return scaler, rcpower_scaled, rcpower


"""
This function uses the before created MinMaxScale oblect to scaling the validation or test
sequence
"""


def scale_sequence(scaler, df_val_or_test, dt_dset, rcpower_dset, f=None):
    """

    :param scaler:  - scaler object created by get_scaler4train
    :param df_val_or_test:  - dataFrame val or test sequences
    :param dt_dset:  - name of 'Date time' in DataFrame object
    :param rcpower_dset: - name of time series characteristic in DataFrame
    :param f: - log file handler
    :return: rcpower_val_or_test_scaled -scalled time series
             rcpower_val_or_test - natural time series
    """
    pass
    rcpower_val_or_test = df_val_or_test[rcpower_dset].values
    rcpower_val_or_test_scaled = scaler.transform(rcpower_val_or_test.reshape(-1, 1)).reshape(-1, )

    chunkarray2log("Data array", rcpower_val_or_test, 8, f)
    chunkarray2log("Scaled dataarray", rcpower_val_or_test_scaled, 8, f)

    return rcpower_val_or_test_scaled, rcpower_val_or_test


""" This function transforms the entire time series to the Supervised Learning Data
    i.e., each subsequence of time series x[t-k], x[t-k+1],..,x[t-1],x[t] transforms to the row of matrix X(samples,k)
    [x[t-k] x[t-k+1] .... x[t-2] x[t-1]] and element x[t] of vector y(samples)
"""


def TimeSeries2SupervisedLearningData(raw_seq, n_steps, f=None):
    """

    :param raw_seq:
    :param n_steps:
    :param f:
    :return:
    """

    chunkarray2log("Data array ", raw_seq, 32, f)

    X, y = split_sequence(raw_seq, n_steps)

    svld2log(X, y, 32, f)

    return X, y


"""
split a untivariate sequence into supervised data
"""


def split_sequence(sequence, n_steps):
    """

    :param sequence:
    :param n_steps:
    :return:
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def prepareDataset(cp, ds, f=None):
    pass
    str(ds)
    ds.dset2arrays(cp.n_steps, cp.n_features, cp.epochs)

    show_autocorr(ds.rcpower, 144, cp.rcpower_dset, cp.folder_control_log, cp.stop_on_chart_show, cp.fc)

    return

"""
The d_models dictionary is filled by pair index:wrapper for tensorflow model
The wrapper is an instance of one of classes (MAP, LSTM,CNN)/
For each class (MLP,LSTM, CNN) can be created one or more models with different structure( actual models).
Those actual models are hardcoded in the MLP, LSTM or CNN class definition.
Now are exists:
    MLP -> mlp_1, mlp_2
    CNN -> univar_cnn
    LSTM -> vanilla_LSTM, stacked_LASM, bdir_LSTM
    tsARIMA -> seasonal_arima, besr_arima

"""
@exec_time
def d_models_assembly(d_models, keyType, valueList, cp, ds):
    """

    :param d_models: dictionary {<index>:<wrapper for model>.
                     Through parameter <wrapper>.model, access to tensorflow model  is given.
    :param keyType: string value, type of NN model like as 'MLP','CNN','LSTM'
    :param valueList: tuple(index,model name) , i.e. (0,'mlp_1'),(1,'mlp_2') like as all models defined in cfg.py
    :param cp: instance of ControlPlane class
    :param ds: instance of Dataset class
    :return:
    """
    for tuple_item in valueList:
        index_model, name_model = tuple_item
        if keyType == "MLP":
            curr_model = MLP(name_model, keyType, cp.n_steps, cp.epochs, cp.fc)
            curr_model.param = (cp.n_steps, cp.n_features, cp.hidden_neyrons, cp.dropout)
        elif keyType == "LSTM":
            curr_model = LSTM(name_model, keyType, cp.n_steps, cp.epochs, cp.fc)
            curr_model.param = (cp.units, cp.n_steps, cp.n_features)
        elif keyType == "CNN":
            curr_model = CNN(name_model, keyType, cp.n_steps, cp.epochs, cp.fc)
            curr_model.param = (cp.n_steps, cp.n_features)
        elif keyType == "tsARIMA":
            curr_model = tsARIMA(name_model, keyType, cp.n_steps, cp.epochs, cp.fc)
            if name_model == 'seasonal_arima':
                curr_model.param =(1, 1, 1,  1, 1, 1, cp.seasonaly_period, cp.predict_lag, cp.discret * 60, \
                                   ds.df[cp.rcpower_dset].values)

            elif name_model == 'best_arima':
                curr_model.param =(1, 1, 1,  cp.max_p, cp.max_q, cp.max_d, cp.predict_lag, cp.discret * 60, \
                                   ds.df[cp.rcpower_dset].values)

            else:
                smsg = "Undefined name of ARIMA {}\n It is not supported by STGELDP!".format(keyType)
                print(smsg)
                if cp.fc is not None:
                    cp.fc.write(smsg)
                return
        else:
            smsg = "Undefined type of Neuron Net or ARIMA {}\n It is not supported by STGELDP!".format(keyType)
            print(smsg)
            if cp.fc is not None:
                cp.fc.write(smsg)
            return
        curr_model.path2modelrepository = cp.path_repository
        curr_model.timeseries_name = cp.rcpower_dset
        if keyType != "tsARIMA":   # no scaler for ARIMA
            curr_model.scaler = ds.scaler

        funcname = getattr(curr_model, name_model)
        curr_model.set_model_from_template(funcname)

        print (str(curr_model))
        msg2log(d_models_assembly, str(curr_model), cp.ft)

        d_models[index_model] = curr_model
    return

@exec_time
def fit_models(d_models,  cp, ds):
    pass
    histories = {}
    for k, v in d_models.items():
        curr_model = v

        X = copy.copy(ds.X)
        X_val = copy.copy(ds.X_val)
        # #LSTM
        if curr_model.typeModel == "CNN" or curr_model.typeModel == "LSTM":
            X = X.reshape((X.shape[0], X.shape[1], cp.n_features))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], cp.n_features))

        curr_model.param_fit = (
            X, ds.y, X_val, ds.y_val, cp.n_steps, cp.n_features, cp.epochs, cp.folder_train_log, cp.ft)
        msg2log(fit_models.__name__, "\n\n {} model  fitting\n".format(curr_model.nameModel), cp.ft)
        history = curr_model.fit_model()

        if curr_model.typeModel == "CNN" or curr_model.typeModel == "LSTM" or curr_model.typeModel == "MLP":

            chart_MAE(curr_model.nameModel, cp.rcpower_dset, history, cp.n_steps, cp.folder_train_log, cp.stop_on_chart_show)
            chart_MSE(curr_model.nameModel, cp.rcpower_dset, history, cp.n_steps, cp.folder_train_log, cp.stop_on_chart_show)
        elif curr_model.typeModel == "tsARIMA":
            curr_model.fitted_model_logging()

        histories[k] = history

    return histories

"""
The trained models saved in the 'Model Repository'.
The path saved model is following
    <Model Repository>/<Time Series Name>/<Model Type like as LSTM,MPL or CNN>/<actual model>
Below the folder tree for Imbalance prediction sample.    
F:\MODEL_REPOSITORY
└───Imbalance
    ├───CNN
    │   └───univar_cnn
    │       ├───assets
    │       └───variables
    ├───LSTM
    │   ├───bidir_lstm
    │   │   ├───assets
    │   │   └───variables
    │   ├───stacked_lstm
    │   │   ├───assets
    │   │   └───variables
    │   └───vanilla_lstm
    │       ├───assets
    │       └───variables
    └───MLP
        ├───mlp_1
        │   ├───assets
        │   └───variables
        └───mlp_2
            ├───assets
            └───variables
    
    
    
"""
def save_modeles_in_repository(d_models,  cp):
    """

    :param d_models:
    :param cp:
    :return:
    """
    # save modeles
    dict_srz={}
    lst_srz =[]
    for k, v in d_models.items():
        curr_model = v

        filepath_to_save_model = Path(curr_model.path2modelrepository) / curr_model.timeseries_name / curr_model.typeModel/curr_model.nameModel

        if curr_model.typeModel == "CNN" or curr_model.typeModel=="LSTM" or curr_model.typeModel == "MLP":
            curr_model.save_model_wrapper()
        elif curr_model.typeModel == "tsARIMA":
            curr_model.save_model()

        title = "Fitted model {} : {} saved".format(curr_model.typeModel, curr_model.nameModel)
        msg = "Path to repository is {}".format(filepath_to_save_model)
        msg2log(title, msg, cp.ft)
        lst_srz.append(filepath_to_save_model)

    dict_srz[cp.rcpower_dset]=lst_srz

    serialize_lst_trained_models(dict_srz, cp)

    return

def serialize_lst_trained_models(dict_srz, cp):
    pass
    filepath_to_serialize = Path( Path(cp.path_repository) / cp.rcpower_dset / cp.rcpower_dset).with_suffix('.pkl')

    with open(filepath_to_serialize,'wb') as fw:
        dump(dict_srz, fw)
    msg ="Trained models list serialized in {}\n".format(filepath_to_serialize)
    msg2log(serialize_lst_trained_models.__name__, msg, cp.ft)

    return


def deserialize_lst_trained_models(cp):

    pass
    filepath_to_serialize = Path(Path(cp.path_repository) / cp.rcpower_dset / cp.rcpower_dset).with_suffix('.pkl')

    if not Path(filepath_to_serialize).exists():
        msg = "Trained models missed or not found for {} Time Series.\nProcesing terminate\n".format(cp.rcpower_dset)
        print(msg)
        msg2log(deserialize_lst_trained_models.__name__, msg, cp.fp)
        return None
    # if sys.platform == 'win32':
    #     filep = str(filepath_to_serialize).replace('/', '\\')
    # else:
    #     filep = filepath_to_serialize
    with open(filepath_to_serialize, 'rb') as fr:
        dict_srz = load(fr)

    return dict_srz


def get_list_trained_models(cp):
    """

    :param cp:
    :return: dict_model ={<act.model name>:(<model type>,<path_model in repository>)
    """
    pass
    dict_model = {}
    dict_srz = deserialize_lst_trained_models(cp)
    if dict_srz is None:
        msg = "Trained models missed or not found for {} Time Series.\nProcesing terminate\n".format(cp.rcpower_dset)
        msg2log(get_list_trained_models.__name__, msg, cp.fp)
        return None
    for k, value_list in dict_srz.items():
        msg = "For {} Time series, the following trained models found\n ".format(cp.rcpower_dset)
        print(msg)
        msg2log("", msg, cp.fp)
        for path_model_name in value_list:
            msg = "{}".format(path_model_name)
            print(msg)
            msg2log("", msg, cp.fp)
            p = Path(path_model_name)
            ppar = p.parts
            ll = len(ppar)
            act_model_name = ppar[ll - 1]
            act_model_type = ppar[ll - 2]
            dict_model[act_model_name] = (act_model_type, path_model_name)
    return dict_model
"""
From dictionary dict_model with items <act. model>:(<type  model or class >, <path to saved trained model>), the classes are 
instantiated and predict processed.
The type model belongs to 'LSTM','CNN','MLP'
"""
@exec_time
def predict_model(dict_model,  cp, ds, n_predict = 1):

    dict_predict= {}
    vec_4_predict = copy.copy(ds.data_for_predict)
    vector_logging("The tail of {} time series for short term prediction\n".format(cp.rcpower_dset), vec_4_predict, 8, cp.fp)
    pass
    for key, value in dict_model.items():
        (model_type, path_model_name) = value
        if model_type == "MLP":
            curr_model = MLP(  key, model_type, cp.n_steps, cp.epochs, cp.fp)
        elif model_type == "CNN":
            curr_model = CNN(  key, model_type, cp.n_steps, cp.epochs, cp.fp)
        elif model_type == "LSTM":
            curr_model = LSTM( key, model_type, cp.n_steps, cp.epochs, cp.fp)
        elif model_type == "tsARIMA":
            curr_model = tsARIMA(key, model_type, cp.n_steps, cp.epochs, cp.fc)
            if key == 'seasonal_arima':
                curr_model.param = (1, 1, 1, 1, 1, 1, cp.seasonaly_period, cp.predict_lag, cp.discret * 60, \
                                    ds.df[cp.rcpower_dset].values)

            elif key == 'best_arima':
                curr_model.param = (1, 1, 1, cp.max_p, cp.max_q, cp.max_d, cp.predict_lag, cp.discret * 60, \
                                    ds.df[cp.rcpower_dset].values)

            else:
                smsg = "Undefined name of ARIMA {}\n It is not supported by STGELDP!".format(key)
                print(smsg)
                if cp.fc is not None:
                    cp.fc.write(smsg)
                return
        else:
            smsg = "Undefined type of Neuron Net or ARIMA {}\n It is not supported by STGELDP!".format(key)
            print(smsg)
            if cp.fc is not None:
                cp.fc.write(smsg)
            return

        curr_model.timeseries_name = cp.rcpower_dset
        curr_model.path2modelrepository = cp.path_repository
        if model_type == "MLP" or model_type == "CNN" or model_type == "LSTM":
            curr_model.scaler = ds.scaler

        print(str(curr_model))
        msg2log(predict_model.__name__, str(curr_model), cp.fp)

        status = curr_model.load_model_wrapper()

        y=np.zeros(n_predict)
        ins_index =0
        if curr_model.scaler is not None:
            vec_4_predict_sc = curr_model.scaler.transform(vec_4_predict.reshape(-1, 1)).reshape(-1, )
            vector_logging("After scaling\n", vec_4_predict_sc, 16, cp.fp)
        else:
            vec_4_predict_sc = copy.copy(vec_4_predict)
        if model_type == "MLP" or  model_type == "CNN" or  model_type == "LSTM":
            for k in range(n_predict):
                y_sc =curr_model.predict_one_step(vec_4_predict_sc)

                y_pred=y_sc
                if curr_model.scaler is not None:
                    y_pred = curr_model.scaler.inverse_transform((y_sc))
                y[ins_index]=y_pred
                ins_index+=1

                vec_4_predict_sc = shift(vec_4_predict_sc, -1, y_sc)
                # vector_logging("After shift\n", vec_4_predict_sc, 16, cp.fp)
        elif model_type == "tsARIMA":   # saved ARIMA models contain the time seies into. So in order to predict ,
                                        # is need to pass the forcasting lag
                                        #
            y = curr_model.predict_n_steps( n_predict)

        vector_logging("{} Short Term Forecasting\n".format(curr_model.nameModel), y, 4, cp.fp)
        dict_predict[curr_model.nameModel]=copy.copy(y)

    return dict_predict




###################################################################################################################

def autocorr(x, lags, f=None):
    chunkarray2log(" Time Series ...", x[:16], 8, f)
    mean = np.mean(x)
    var = np.var(x)
    xp = x - mean

    corr = [1.0 if l == 0 else np.sum(xp[l:] * xp[:-l]) / (len(x) * var) for l in lags]
    chunkarray2log("Autocorrelation ...", corr[:16], 8, f)
    return np.array(corr)


def autocorr_firstDiff(x, lags,  f=None):
    x_fd = np.array([])
    for i in range(len(x) - 1):
        x_fd = np.append(x_fd, x[i+1] - x[i])
    chunkarray2log("First Diff. Time Series ...", x_fd[:16], 8, f)
    mean = np.mean(x_fd)
    var = np.var(x_fd)
    xp = x_fd - mean

    corr = [1.0 if l == 0 else np.sum(xp[l:] * xp[:-l]) / (len(x_fd) * var) for l in lags]
    chunkarray2log("First Diff. Autocorrelation ...", corr[:16], 8, f)
    return np.array(corr)


def autocorr_secondDiff(x, lags, f=None):
    x_sd = np.array([])
    for i in range(len(x) - 2):
        x_sd = np.append(x_sd, x[i+2] - 2 * x[i +1] + x[i])
    chunkarray2log("Second Diff. Time Series ...", x_sd[:16], 8, f)
    mean = np.mean(x_sd)
    var = np.var(x_sd)
    xp = x_sd - mean

    corr = [1.0 if l == 0 else np.sum(xp[l:] * xp[:-l]) / (len(x_sd) * var) for l in lags]
    chunkarray2log("Second Diff. Autocorrelation ...", corr[:16], 8, f)
    return np.array(corr)

def autocorr_firstDiffSeasonDiff(x, lags, f=None):
    x_sd = np.array([])
    for i in range(len(x) - 144 -1):
        x_sd = np.append(x_sd, x[i+145] -  x[i + 144] - x[i + 1] +x[i])
    chunkarray2log("(First Diff. * Season Diff) Time Series ...", x_sd[:16], 8, f)
    mean = np.mean(x_sd)
    var = np.var(x_sd)
    xp = x_sd - mean

    corr = [1.0 if l == 0 else np.sum(xp[l:] * xp[:-l]) / (len(x_sd) * var) for l in lags]
    chunkarray2log("(First Diff. * Season Diff) Autocorrelation ...", corr[:16], 8, f)
    return np.array(corr)


def show_autocorr(y, lag_max, title, logfolder, stop_on_chart_show=False, f=None):
    y = np.array(y).astype('float')
    lags = range(lag_max)

    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=16)

    for funcii, labelii in zip([autocorr, autocorr_firstDiff, autocorr_secondDiff,autocorr_firstDiffSeasonDiff], \
                               ['Time Series Autocorrelation', 'First Diff. Autocorrelation',
                                'Second Diff. Autocorrelation','(First Diff * Season Diff) Autokorrrelation ']):
        cii = funcii(y, lags, f)
        print(labelii)
        print(cii)
        ax.plot(lags, cii, label=labelii)

    ax.set_xlabel('lag')
    ax.set_ylabel('Correlation Coefficient')
    ax.legend()
    plt.show(block=stop_on_chart_show)


    filePng = Path(PlotPrintManager.get_ControlLoggingFolder()) / (
        "autocorrelation_{}.png".format(title))
    plt.savefig(filePng)
    if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")

    return

########################################################################################################################




if __name__ == "__main__":
    pass