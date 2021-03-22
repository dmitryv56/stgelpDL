#!/usr/bin/python3

import sys
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX,SARIMAXResults,SARIMAXParams
from statsmodels.tsa.statespace.mlemodel import MLEResults

from predictor.utility import msg2log
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from offlinepred.api import logMatrix, plotPredictDF


def readDataset(csv_file:str="", endogen:list=None, title:str="Time Series",f:object=None)->pd.DataFrame:

    my_csv=Path(csv_file)
    if my_csv.is_file():
        df1=pd.read_csv(csv_file)
        if endogen is not None and len(endogen)>0:
            plot = df1.plot(y=endogen,figsize=(14,8),legend=True,title=title)
            title1=title.replace(' ','_')
            file_png=Path(D_LOGS['plot'] / Path(title1)).with_suffix('.png')
            fig=plot.get_figure()
            fig.savefig(str(file_png))

    else:
        df1=None
    return df1

def checkStationarity(df:pd.DataFrame=None, data_col_name:str=None,title:str="Time Series",f:object=None):

    if df is None or data_col_name is None:
        return

    series=df[data_col_name].values
    # ADF Test
    result = adfuller(series, autolag='AIC')
    msg2log(None, f'ADF Statistic: {result[0]}',f)
    msg2log(None, f'n_lags: {result[1]}',f)
    msg2log(None,f'p-value: {result[1]}',f)
    for key, value in result[4].items():
        msg2log(None,'Critial Values:',f)
        msg2log(None,f'   {key}, {value}',f)

    plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})


    # Original Series
    fig, axes = plt.subplots(3, 3, sharex=True)
    axes[0, 0].plot(df[data_col_name]); axes[0, 0].set_title('Original Series')
    plot_acf(df[data_col_name], ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df[data_col_name].diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df[data_col_name].diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df[data_col_name].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df[data_col_name].diff().diff().dropna(), ax=axes[2, 1])
    title1 = "{}_{}".format(title.replace(' ', '_'),data_col_name)
    file_png = Path(D_LOGS['plot'] / Path(title1)).with_suffix('.png')
    plt.savefig(file_png)
    plt.close("all")

    return

def arima_order(df:pd.DataFrame=None, data_col_name:str=None,training_size:int=512,title:str="Time Series",
                max_order:tuple=(2,2,2), max_seasonal_order:tuple=(1,0,1,6),
                f:object=None)->((int,int,int),(int,int,int,int)):

    n = len(df[data_col_name])
    start_index=0
    if n>training_size:
        start_index=n-training_size
    (P,D,Q)=max_order
    (SP,SD,SQ,S)=max_seasonal_order
    opt_order=(0,0,0)
    opt_aic=1e+12
    for p in range(P+1):
        for d in range(D+1):
            for q in range(Q+1):
                order=(p,d,q)
                seasonal_order = (0, 0, 0, 0)
                errmsg=""
                try:
                    model = SARIMAX(df[data_col_name][start_index:], order=order, seasonal_order=seasonal_order)
                    model_fit = model.fit(disp=0)
                except:
                    errmsg = f""" SARIMA optimal order searching
({p},{d},{q})X(0,0,0)
Oops!! Unexpected error...
Error : {sys.exc_info()[0]}
"""
                finally:
                    if len(errmsg) > 0:
                        msg2log(None, errmsg, D_LOGS['except'])
                        break
                if model_fit.aic<opt_aic:
                    opt_aic=model_fit.aic
                    opt_order=(p,d,q)
                msg2log(None,"ARIMA({},{},{}): AIC={}".format(p,d,q,model_fit.aic))
    opt_seasonal_order = (0, 0, 0, S)
    opt_seasonal_aic = 1e+12
    opt_sarima_aic   = opt_aic+1.0
    if S>0:
        opt_seasonal_order = (0, 0, 0,S)
        opt_seasonal_aic = 1e+12
        for sp in range(SP+1):
            for sd in range(SD+1):
                for sq in range(SQ+1):
                    seasonal_order=(sp,sd,sq,S)
                    order=(0,0,0)
                    errmsg=""
                    try:
                        model = SARIMAX(df[data_col_name][start_index:], order=order, seasonal_order=seasonal_order)
                        model_fit = model.fit(disp=0)
                    except:
                        errmsg = f""" SARIMA optimal order searching
(0,0,0)X({sp},{sd},{sq}):{S}
Oops!! Unexpected error...
Error : {sys.exc_info()[0]}
"""
                    finally:
                        if len(errmsg)>0:
                            msg2log(None,errmsg,D_LOGS['except'])
                            break

                    if model_fit.aic < opt_seasonal_aic:
                        opt_seasonal_aic = model_fit.aic
                        opt_seasonal_order = (sp, sd, sq,S)
                    msg2log(None, "ARIMA(0,0,0)x({},{},{},{}): AIC={}".format(sp, sd, sq, S, model_fit.aic))
                    seasonal_order = (0, 0, 0, 0)
        opt_sarima_aic=1e+12
        model = SARIMAX(df[data_col_name][start_index:], order=opt_order, seasonal_order=opt_seasonal_order)
        model_fit = model.fit(disp=0)
        opt_sarima_aic=model_fit.aic

    message=f"""SARIMA models comparison 
SARIMA({opt_order})x(0,0,0,0) : AIC={opt_aic}
SARIMA(0,0,0)x({opt_seasonal_order}) : AIC={opt_seasonal_aic}
SARIMA({opt_order})x({opt_seasonal_order}) : AIC={opt_sarima_aic}
"""
    msg2log(None,message,f)

    if opt_aic<opt_seasonal_aic and opt_aic<opt_sarima_aic:
        order=opt_order
        seasonal_order=(0,0,0,0)
    elif opt_seasonal_aic<opt_aic and opt_seasonal_aic<opt_sarima_aic:
        opder=(0,0,0)
        seasonal_order=opt_seasonal_order
    elif opt_sarima_aic<opt_aic and opt_sarima_aic < opt_seasonal_aic:
        order=opt_order
        seasonal_order=opt_seasonal_order

    return order,seasonal_order

def arima_run(df:pd.DataFrame=None, data_col_name:str=None,dt_col_name:str="Date Time",chunk_offset:int=0,
              chunk_size:int=8192, in_sample_start:int=0,in_sample_size:int=512, forecast_period:int=4,
              title:str="Time Series", order:tuple=(1,0,0),seasonal_order:tuple=(0,0,0,0),  f:object=None):
    pass
    n=len(df[data_col_name])

    if chunk_size+chunk_offset>n:
        chunk_size=n-chunk_offset
    cho=chunk_offset
    tcho=df[dt_col_name][chunk_offset]
    chs=chunk_size
    tchs=df[dt_col_name][chunk_offset+chunk_size-1]

    if in_sample_start + in_sample_size>n:
        in_sample_size=n-in_sample_start
    iss=in_sample_start
    tiss=df[dt_col_name][iss]
    isl=in_sample_start+in_sample_size-1
    tisl=df[dt_col_name][isl]
    message=f"""{data_col_name}
TS length: {n}
The chunk of TS for ARIMA estimation:
start offset :                {cho}  timestamp:     {tcho}
chunk size   :                {chs}  last timesamp: {tchs}
In-sample predict from index: {iss}, timestamp {tiss} 
                  till index: {isl}, last timestamp {tisl}
ARIMA  order: p = {order[0]} d = {order[1]} q = {order[2]}
SARIMA order: P = {seasonal_order[0]} D = {seasonal_order[1]} Q ={seasonal_order[2]} 
              Seasonal Period = {seasonal_order[3]}

"""
    msg2log(None,message,f)
    msg2log(None,message,D_LOGS['main'])
    log2All()

    model = SARIMAX(df[data_col_name][chunk_offset:], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=0)
    msg2log(None,model_fit.summary(),D_LOGS['predict'])
    msg2log(None, model_fit.param_names, D_LOGS['predict'])

    y_pred_series = model_fit.predict(start=in_sample_start, end=in_sample_start + in_sample_size-1)
    y_pred=np.array(y_pred_series)
    y=np.array(df[data_col_name][in_sample_start:in_sample_start+in_sample_size])
    err = np.round(np.subtract(y, y_pred),decimals=4)
    X,predict_dict=predict_bundle(y = y, y_pred = y_pred, err = err, start_index=in_sample_start,f = f)
    title = "{:^60s}\n{:^10s} {:^10s} {:^10s} {:^10s} {:^10s} {:^10s}".format(data_col_name, "NN","Index","Obs Value",
                                                                              "Predict","Error","Abs.Err")
    logMatrix(X, title= title, wideformat = '10.4f', f=D_LOGS['predict'])
    plotPredict(predict_dict=predict_dict, data_col_name=data_col_name, title= "in_sample_predict", f=D_LOGS['predict'])

    forecast_arr_series=model_fit.forecast(steps=forecast_period)
    forecast_arr=np.array(forecast_arr_series)
    X1,forecast_dict = predict_bundle(y=forecast_arr, y_pred=None, err=None, start_index=n, f=f)
    title = "{:^60s}\n{:^14s} {:^14s} {:^14s} ".format(data_col_name, "NN", "Index", "Forecast")
    logMatrix(X1, title=title, wideformat='14.8f', f=D_LOGS['predict'])
    plotPredict(predict_dict=forecast_dict, data_col_name=data_col_name, title="forecasting", f=D_LOGS['predict'])
    return

def predict_bundle(y:np.array=None, y_pred:np.array=None, err:np.array=None,start_index:int=0,f:object=None)->(np.array,
        dict):
    predict_dict={}
    (n,) = y.shape
    z = np.array([i for i in range(start_index, start_index + n)])
    predict_dict["ind"] = copy.copy(z)

    if y_pred is None:
        predict_dict["forecast"]=y
    else:
        predict_dict["observation"]=y
        predict_dict["in_sample_predict"]=y_pred


    y1 = y.reshape((n, 1))
    err_abs = None
    pred_err_abs = None
    y2=None
    if err is not None:
        abserr=np.round(np.absolute(err),decimals=4)
        predict_dict["error"]=copy.copy(err)
        predict_dict["abserror"] = copy.copy(abserr)
        err=err.reshape((n,1))
        abserr=abserr.reshape((n,1))
        err_abs=np.append(err,abserr,axis=1)

    if y_pred is not None and err_abs is not None:
       y2=y_pred.reshape((n,1))
       pred_err_abs = np.append(y2, err_abs, axis=1)
    elif y_pred is not None and err_abs is None:
        pred_err_abs = y_pred.reshape((n,1))
    if pred_err_abs is not None:
        y_pred_err_abs = np.append(y1, pred_err_abs, axis=1)
    else:
        y_pred_err_abs=y1

    z = np.array([i for i in range(start_index,start_index+n)])
    predict_dict["ind"] = copy.copy(z)
    z = z.reshape((n, 1))
    X = np.append(z, y_pred_err_abs, axis=1)
    return X, predict_dict

def plotPredict(predict_dict:dict=None, data_col_name:str="",title:str="in_sample_predict",f:object=None):
    df = pd.DataFrame(predict_dict)
    sFolder = Path(D_LOGS['plot'] / Path(data_col_name) / Path(title))
    sFolder.mkdir(parents=True, exist_ok=True)
    title1 = "{}_{}".format(title, data_col_name)
    test_predictions_file = Path(sFolder / Path(title1)).with_suffix('.csv')
    df.to_csv(test_predictions_file, index=False)
    msg = "{} test sequence predict by {} ARIMA model saved in \n{}\n".format(data_col_name, title,
                                                                            test_predictions_file)
    msg2log(None, msg, D_LOGS['predict'])

    plotPredictDF(test_predictions_file, data_col_name, title=title1)
    return





