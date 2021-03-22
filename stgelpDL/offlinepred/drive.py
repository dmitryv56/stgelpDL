#!/usr/bin/python3

""" The drive functions and functional tests functions for offline prediction.
drive()
drive_feature()
SLD_Test()
drive_test()

"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np


from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from offlinepred.api import tsSLD, hmaggSLD,logMatrix, logMatrixVector, d_models_assembly, fit_models, PATH_REPOSITORY,\
    bundlePredict
from predictor.utility import vector_logging,msg2log,PlotPrintManager,shift
from predictor.cfg import ALL_MODELS
from simTM.auxapi import dictIterate,listIterate

@exec_time
def drive(df:pd.DataFrame=None, title:str="ElHiero" , dt_col_name:str="Date Time",  ts_list:list=None, discret:int=None,
          n:int=0, n_step:int=32, n_eval:int=256, n_test:int=32, n_pred:int=4, folder_for_logging:str=None,
          folder_for_train_logging:str=None, folder_for_control_logging:str=None,
          folder_for_predict_logging:str=None)->int:

    if df is None or ts_list is None or not ts_list:
        return -1

    s72=''.join(["*" for i in range(72)])

    for ts_item in ts_list:
        log2All("\n{}\nStart {} processing...\n{}\n\n".format(s72,ts_item,s72))
        ret = drive_feature(df=df, title=ts_item, dt_col_name=dt_col_name, data_col_name=ts_item, discret=discret,
                            n=n, n_step=n_step, n_eval=n_eval, n_test=n_test, n_pred=n_pred,
                            folder_for_logging=folder_for_logging, folder_for_train_logging=folder_for_train_logging,
                            folder_for_control_logging=folder_for_control_logging,
                            folder_for_predict_logging=folder_for_predict_logging)
        log2All("\n\n{}\nFinish {} processing...\n{}\n{}\n\n".format(s72, ts_item, s72,s72))
        log2All()
    pass
    return 0

@exec_time
def drive_hmaggregated(df:pd.DataFrame=None, title:str="ElHiero" , dt_col_name:str="Date Time",  ts_list:list=None,
                       discret:int=None,  n:int=0, n_step:int=32, n_eval:int=256, n_test:int=32, n_pred:int=4,
                       daylight_begin:int =7,daylight_end:int=19, minute:int=0, folder_for_logging:str=None,
                       folder_for_train_logging:str=None, folder_for_control_logging:str=None,
                       folder_for_predict_logging:str=None)->int:

    if df is None or ts_list is None or not ts_list:
        return -1
    if daylight_begin is None or daylight_end is None:
        return -2
    if daylight_begin>daylight_end:
        return -3


    s72=''.join(["*" for i in range(72)])

    for ts_item in ts_list:
        log2All("\n{}\nStart {} processing...\n{}\n\n".format(s72,ts_item,s72))
        ret = drive_feature_hmaggregated(df=df, title=ts_item, dt_col_name=dt_col_name, data_col_name=ts_item,
                                         discret=discret, n=n, n_step=n_step, n_eval=n_eval, n_test=n_test,
                                         n_pred=n_pred, daylight_begin=daylight_begin, daylight_end=daylight_end,
                                         minute=minute, folder_for_logging=folder_for_logging,
                                         folder_for_train_logging=folder_for_train_logging,
                                         folder_for_control_logging=folder_for_control_logging,
                                         folder_for_predict_logging=folder_for_predict_logging)
        log2All("\n\n{}\nFinish {} processing...\n{}\n{}\n\n".format(s72, ts_item, s72,s72))
        log2All()
    pass
    return 0

@exec_time
def drive_feature(df:pd.DataFrame=None, title:str="ElHiero" , dt_col_name:str="Date Time",  data_col_name:str=None,
                  discret:int=None, n:int=0, n_step:int=32, n_eval:int=256, n_test:int=32, n_pred:int=4,
                  folder_for_logging:str=None,  folder_for_train_logging:str=None, folder_for_control_logging:str=None,
                  folder_for_predict_logging:str=None)->int:

    if data_col_name is None:
        return -1

    # create Supervised Learning Data Set
    sld=tsSLD(df=df, data_col_name=data_col_name, dt_col_name=dt_col_name, n_step=n_step, n_eval=n_eval,n_test=n_test,
                 f=D_LOGS['control'])

    sld.crtSLD()
    X_predict = sld.predSLD(start_in=sld.n_test) #start_in=n_step+2)
    sld.ts_analysis()

    d_models = {}
    message=dictIterate(ALL_MODELS)
    msg2log(None, "Used models\n{}".format(message), D_LOGS['control'])
    for keyType, valueList in ALL_MODELS.items():
        print('{}->{}'.format(keyType, valueList))
        # if keyType=="LSTM": continue
        status = d_models_assembly(d_models, keyType, valueList, sld=sld)

    message = dictIterate(d_models)
    msg2log(None, "Model dictionary (d_models)\n{}".format(message), D_LOGS['control'])

    histories, dict_predict =fit_models(d_models,sld,X_predict=X_predict)
    message = dictIterate(histories)
    msg2log(None, "History fit  dictionary (d_models)\n{}".format(message), D_LOGS['control'])

    if dict_predict is not None and len(dict_predict)>0:
        for key,val in dict_predict.items():
            vector_logging("{} Short Term Forecasting\n".format(key), val, 4, D_LOGS['predict'])
    bundlePredict(sld, dict_predict)
    return

@exec_time
def drive_feature_hmaggregated(df:pd.DataFrame=None, title:str="ElHiero" , dt_col_name:str="Date Time",  data_col_name:str=None,
                  discret:int=None, n:int=0, n_step:int=32, n_eval:int=256, n_test:int=32, n_pred:int=4,
                  daylight_begin:int =7,daylight_end:int=19, minute:int=0,
                  folder_for_logging:str=None,  folder_for_train_logging:str=None, folder_for_control_logging:str=None,
                  folder_for_predict_logging:str=None)->int:

    if data_col_name is None:
        return -1

    # create Supervised Learning Data Set
    sld=hmaggSLD(df=df, data_col_name=data_col_name, dt_col_name=dt_col_name, n_step=n_step, n_eval=n_eval,n_test=n_test,
                 f=D_LOGS['control'])
    for hour in range(daylight_begin,daylight_end+1):
        sld.hour=hour
        sld.minute=minute
        addtitle="H{}_M{}".format(sld.hour,sld.minute)
        sld.title="{}_TS_aggregated_along_{}".format(sld.data_col_name,addtitle)
        log2All("{} time series aggregated along the time {} started".format(sld.data_col_name,addtitle))
        msgErr=""
        try:
            sld.crtSLD()
            X_predict = sld.X_test[-1,:]
            sld.psd_segment_size = 64
            sld.ts_analysis(addtitle=addtitle)

            d_models = {}
            message=dictIterate(ALL_MODELS)
            msg2log(None, "Used models {}\n{}".format(addtitle,message), D_LOGS['control'])
            for keyType, valueList in ALL_MODELS.items():
                print('{}->{}'.format(keyType, valueList))

                status = d_models_assembly(d_models, keyType, valueList, sld=sld)

            message = dictIterate(d_models)
            msg2log(None, "Model dictionary (d_models) {}\n{}".format(addtitle,message), D_LOGS['control'])
            X_predict=X_predict.reshape((1,sld.n_step)) # from vector to matrix
            histories, dict_predict =fit_models(d_models,sld,X_predict=X_predict, addtitle=addtitle)
            message = dictIterate(histories)
            msg2log(None,
                    "History fit  dictionary (d_models).Aggregated TS along {}\n{}".format(addtitle,message),
                    D_LOGS['control'])
            msgErr1=""
            try:
                if dict_predict is not None and len(dict_predict)>0:
                    for key,val in dict_predict.items():
                        msgErr2 = ""
                        try:
                            vector_logging("{} Short Term Forecasting {}\n".format(key,addtitle), val, 4, D_LOGS['predict'])
                        except:
                            msgErr2 = "O-o-ops! I got an unexpected error at vector_logging - reason  {}\n".format(
                                sys.exc_info())

                        finally:
                            if len(msgErr2)>0:
                                msg2log(drive_feature_hmaggregated.__name__,msgErr2,D_LOGS['except'])
                msgErr3=""
                try:
                    bundlePredict(sld, dict_predict,addtitle=addtitle)
                except:
                    msgErr3 = "O-o-ops! I got an unexpected error at bundlePredict - reason  {}\n".format(
                        sys.exc_info())
                finally:
                    if len(msgErr2) > 0:
                        msg2log(drive_feature_hmaggregated.__name__, msgErr3, D_LOGS['except'])

            except:
                msgErr1 = "O-o-ops! I got an unexpected error at predict print prepare - reason  {}\n".format(sys.exc_info())
            finally:
                if len(msgErr1)>0:
                    msg2log(drive_feature_hmaggregated.__name__,msgErr1,D_LOGS['except'])
        except:
            msgErr = "O-o-ops! I got an unexpected error - reason  {}\n".format(sys.exc_info())

        finally:
            if len(msgErr)>0:
                msg2log(drive_feature_hmaggregated.__name__, msgErr,D_LOGS['except'])
            log2All(
                "{} time series aggregated along the time {} finished".format(sld.data_col_name, addtitle))
            log2All()
    return


""" Functional tests """


def SLD_test(df):
    sld = tsSLD(df=df, data_col_name="test", dt_col_name="Date Time", n_step=4, n_eval=6, n_test=2, bscaled=True,
                f=D_LOGS['main'])
    sld.crtSLD()
    logMatrixVector(sld.X_train, sld.y_train, "Training sequence", f=D_LOGS["main"])
    logMatrixVector(sld.X_eval, sld.y_eval, "Evaluation sequence", f=D_LOGS["main"])
    logMatrixVector(sld.X_test, sld.y_test, "Test sequence", f=D_LOGS["main"])

    # logMatrix(sld.X_test, "X_test",D_LOGS["main"])
    log2All()
    X_pred = sld.predSLD()
    if X_pred is not None:
        logMatrix(X_pred, "X_pred", D_LOGS["main"])
    X_pred1 = sld.predSLD(start_in=8, ts=np.array([3.0, 2.0]))
    if X_pred1 is not None:
        logMatrix(X_pred1, "X_pred1", D_LOGS["main"])
    X_pred2 = sld.predSLD(start_in=8)
    if X_pred2 is not None:
        logMatrix(X_pred2, "X_pred2", D_LOGS["main"])
    X_pred3 = sld.predSLD(ts=np.array([3.0, 2.0]))
    if X_pred3 is not None:
        logMatrix(X_pred3, "X_pred3", D_LOGS["main"])
    return

def drive_test(df):
    PlotPrintManager.set_Logfolders(D_LOGS['plot'], D_LOGS['plot'])
    Path(PATH_REPOSITORY).mkdir(parents=True, exist_ok=True)
    ret= drive(df=df, title = "Drive_test", dt_col_name= "Date Time", ts_list=['test','test1'],
              discret=10, n=len(df), n_step=4, n_eval=6, n_test=2, n_pred = 4)

    closeLogs()



if __name__== "__main__":
    csv_path="/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/test_dataset.csv"
    df=pd.read_csv(csv_path)
    listLogSet("Debug")
    #tsSLD test
    # SLD_test(df)
    #drive_test
    drive_test(df)



