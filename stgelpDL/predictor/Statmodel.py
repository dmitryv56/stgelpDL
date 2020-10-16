#!/usr/bin/python3
""" This module contains the Statmodel class that is the base class for tsARIMA class/"""

import sys
import copy

import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump, load
from pathlib import Path

from predictor.utility import exec_time, msg2log, PlotPrintManager,psd_logging,logDictArima,vector_logging
from predictor.predictor import Predictor

""" Statmodel class """

class Statmodel(Predictor):
    _param = ()
    _param_fit = ()
    model = None

    def __init__(self,nameM, typeM, n_steps, n_epochs, f=None):
        super().__init__(nameM, typeM, n_steps, n_epochs, f)
    #getter/seter

    def set_param(self, val):
        type(self)._param = copy.deepcopy(val)

    def get_param(self):
        return type(self)._param

    param = property(get_param, set_param)

    def set_param_fit(self, val):
        type(self)._param_fit = copy.copy(val)

    def get_param_fit(self):
        return type(self)._param_fit

    param_fit = property(get_param_fit, set_param_fit)

    def set_model_from_saved(self, path_2_saved_model):

        with open(path_2_saved_model, 'rb') as pkl:
            self.model = load(pkl)


        return self.model

    def set_model_from_template(self, func):
        self.model = func()
        if self.f is not None:
            #self.f.write("\n\n{} was sucessfully set}\n".format(self.nameModel))
            pass

        return
    @exec_time
    def fit_model(self):
        """

        :return:
        """
        try:
            history=[]
            self.updateTS_data()
            self.model.fit(self.ts_data)
            self.predict = self.model.predict(self.n_predict)
            title = 'ARIMA: {} predict values after fitting '.format(self.nameModel)
            vector_logging(title, self.predict, 16, self.f)
            msg = "\n{} was sucessfully fitted".format(self.nameModel)
            msg2log(self.fit_model.__name__, msg, self.f)
            dct = self.model.to_dict()
            logDictArima(dct, 0, self.f)
        except Exception as e:
            message = f"""
                                           Oops!! Unexpected error when  ARIMA was estimated...
                                           Error        : {sys.exc_info()[0]}
                                           Description  : {sys.exc_info()[1]}
                                           Excption as e: {e}
                           """

            msg2log(self.fit_model.__name__, message, self.f)
        return history

    @exec_time
    def predict_n_steps(self, n_predict_number):
        """

        :param n_predict_number:
        :return:
        """
        try:
            y=self.model.predict(n_predict_number)
        except Exception as e:
            y=np.zeros(n_predict_number)
            message = f"""
                                                       Oops!! Unexpected error when predicted by ARIMA...
                                                       Error        : {sys.exc_info()[0]}
                                                       Description  : {sys.exc_info()[1]}
                                                       Excption as e: {e}
                                       """

            msg2log(self.predict_n_steps.__name__, message, self.f)
        return y

    def load_model_wrapper(self):
        self.load_model()
        return True

    def load_model(self):
        # TODO Debug
        model_folder = Path(self.path2modelrepository) / self.timeseries_name / self.typeModel / self.nameModel
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        arima_saved_file = Path(model_folder) / "arima.pkl"
        with open(arima_saved_file, 'rb') as pkl:
            self.model = load( pkl)
        msg="Model loaded from {} ".format(model_folder)
        msg2log(self.load_model.__name__, msg, self.f)

            # if not sys.platform == "win32":
            #     # WindowsPath is not iterable
            #     for currFile in currDir:
            #         self.f.write("  {}\n".format(currFile))
        return


    def save_model(self):
        # TODO Debug
        model_folder = Path(self.path2modelrepository) / self.timeseries_name / self.typeModel / self.nameModel
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        model_saving_file = Path(model_folder) / "arima.pkl"
        with open(model_saving_file, 'wb') as pkl:
            dump(self.model, pkl)
        msg ="Model saved in {} ".format(model_folder)
        msg2log(self.save_model.__name__, msg, self.f)

        return
    def updateTS_data(self):
        pass
        (_, _, _, _, _, _, _, _,_,tmp_ts_data) = self.param

        if self.ts_data.shape != tmp_ts_data.shape or np.sum(self.ts_data!=tmp_ts_data)!=0:

            self.ts_data=copy.copy(tmp_ts_data)
            self.model.arima_res_.data.endog=copy.copy(tmp_ts_data)
        return

""" tsARIMA class """

class tsARIMA(Statmodel):
    pass
    _ar_order  = 0
    _ma_order  = 0
    _d_order   = 0
    _AR_order  = 0
    _MA_order  = 0
    _D_order   = 0
    _period    = 0
    _n_predict = 4
    _discret   = 600 # sec
    _ts_data   = None
    _param     = None

     # static member
    _p_ARIMA = -1
    _d_ARIMA = -1
    _q_ARIMA = -1
    _P_ARIMA = -1
    _D_ARIMA = -1
    _Q_ARIMA = -1
    _p_arima = -1
    _d_arima = -1
    _q_arima = -1

    def __init__(self,nameM, typeM, n_steps, n_epochs, f=None):
        self.predict=None
        super().__init__(nameM, typeM, n_steps, n_epochs, f)
        pass



    @staticmethod
    def set_SARIMA(val):
        (tsARIMA._p_ARIMA, tsARIMA._d_ARIMA, tsARIMA._q_ARIMA, tsARIMA._P_ARIMA,tsARIMA._D_ARIMA,tsARIMA._Q_ARIMA) = val

    @staticmethod
    def get_SARIMA():
        return (tsARIMA._p_ARIMA, tsARIMA._d_ARIMA, tsARIMA._q_ARIMA, tsARIMA._P_ARIMA,tsARIMA._D_ARIMA,
                tsARIMA._Q_ARIMA)

    @staticmethod
    def set_ARIMA(val):
        (tsARIMA._p_arima, tsARIMA._d_arima, tsARIMA._q_arima) = val

    @staticmethod
    def get_ARIMA():
        return (tsARIMA._p_arima, tsARIMA._d_arima, tsARIMA._q_arima)

    # getter/setter
    def set_ar_order(self,val):
        type(self)._ar_order = val

    def get_ar_order(self):
        return type(self)._ar_order

    ar_order = property(get_ar_order, set_ar_order)

    def set_ma_order(self, val):
        type(self)._ma_order = val

    def get_ma_order(self):
        return type(self)._ma_order

    ma_order = property(get_ma_order, set_ma_order)

    def set_d_order(self, val):
        type(self)._d_order = val

    def get_d_order(self):
        return type(self)._d_order

    d_order = property(get_d_order, set_d_order)

    def set_AR_order(self, val):
        type(self)._AR_order = val

    def get_AR_order(self):
        return type(self)._AR_order

    AR_order = property(get_AR_order, set_AR_order)

    def set_MA_order(self, val):
        type(self)._MA_order = val

    def get_MA_order(self):
        return type(self)._MA_order

    MA_order = property(get_MA_order, set_MA_order)

    def set_D_order(self, val):
        type(self)._D_order = val

    def get_D_order(self):
        return type(self)._D_order

    D_order = property(get_D_order, set_D_order)

    def set_period(self, val):
        type(self)._period= val

    def get_period(self):
        return type(self)._period

    period = property(get_period, set_period)

    def set_n_predict(self, val):
        type(self)._n_predict= val

    def get_n_predict(self):
        return type(self)._n_predict

    n_predict = property(get_n_predict, set_n_predict)

    def set_discret(self, val):
        type(self)._discret= val

    def get_discret(self):
        return type(self)._discret

    discret = property(get_discret, set_discret)

    def set_ts_data(self, val):
        type(self)._ts_data = None
        type(self)._ts_data= copy.copy(val)

    def get_ts_data(self):
        return type(self)._ts_data

    ts_data = property(get_ts_data, set_ts_data)

    def set_param(self, val):
        type(self)._param = val

    def get_param(self):
        return type(self)._param

    param = property(get_param, set_param)

    def __str__(self):
        return "{} TimeSeries\nARIMA {}".format( self.timeseries_name, self.nameModel )

    @exec_time
    def seasonal_arima(self):
        """

        :return:
        """
        title=self.nameModel

        (self.ar_order,self.d_order,self.ma_order,self.P_order,self.D_order,self.MA_order) =  tsARIMA.get_SARIMA()
        if self.ar_order>-1 and self.d_order>-1 and self.ma_order >-1 and self.P_order >-1 and self.D_order >-1 \
            and self.MA_order>-1 :

            self.param = (self.ar_order,self.d_order,self.ma_order, self.P_order, self.D_order, self.MA_order, \
                          self.period, self.n_predict, self.discret, self.ts_data)
        else:
            (self.ar_order, self.d_order, self.ma_order, self.P_order, self.D_order, self.MA_order, \
                          self.period, self.n_predict, self.discret, self.ts_data) =self.param
            tsARIMA.set_SARIMA((self.ar_order, self.d_order, self.ma_order, self.P_order, self.D_order, self.MA_order))

        p_order=np.array([self.ar_order,self.d_order,self.ma_order])
        P_order = np.array([self.AR_order, self.D_order, self.MA_order, self.period])
        model = pm.arima.ARIMA( p_order,P_order)

        self.model = model

        print("{}:{}".format(self.nameModel, id(self)))

        return model

    @exec_time
    def best_arima(self):
        """

        :return:
        """
        title=self.nameModel

        (self.ar_order, self.d_order, self.ma_order) = tsARIMA.get_ARIMA()
        if self.ar_order>-1 and self.d_order>-1 and self.ma_order>-1 :
            self.param=(self.ar_order, self.d_order, self.ma_order, self.ar_order, self.d_order, self.ma_order, \
                        self.period, self.n_predict, self.discret, self.ts_data)
        else:

            (self.ar_order, self.d_order, self.ma_order, max_p_,  max_d_, max_q_, self.period, self.n_predict ,\
             self.discret, self.ts_data) = self.param
            tsARIMA.set_ARIMA((self.ar_order, self.d_order, self.ma_order))
        p_order = np.array([self.ar_order, self.d_order, self.ma_order])
        model = pm.arima.ARIMA(p_order)
        self.model = model
        print("{}:{}".format(self.nameModel, id(self)))
        return model

    @exec_time
    def control_arima(self):
        """ This method performs the ARIMA(p,d,q) and ARIMA(p,d,q)xARIMA(P,D,Q)[S] estimation/
        These estimated models are forfuture using/

        :return:
        """

        title=self.__str__()

        (p,d,q)=tsARIMA.get_ARIMA()
        bAlreadyCreated =1
        if (p==-1 and d==-1 and q==-1):   # model should be created
            bAlreadyCreated = 0


        if (self.nameModel == 'control_best_arima'):

            #initial parameters from Control object settings
            _, _, _, _, _, _, self.period, self.n_predict,  self.discret,  self.ts_data = self.param
            start_p_=start_d_=start_q_=0
            max_p_ = max_q_ = 3
            max_d_ = 2
            if bAlreadyCreated:  # ARIMA was created,initial parameters are being got from tsARIMA class static members/
                start_p_=0
                start_d_=0
                start_q_=0
                max_p_  =p+1
                max_d_  =d
                max_q_  =0 #q+1


            ################################################################################3

            # model = pm.auto_arima(self.ts_data, exogenous=None, start_p=start_p_, d=start_d_, start_q=start_q_, \
            #                       max_p=max_p_, max_d=max_d_, max_q=max_q_, max_order=16,seasonal=False, trace=True, \
            #                       error_action='ignore', suppress_warnings=True, stepwise=True)

            # model = pm.auto_arima(self.ts_data, exogenous=None, d=2, seasonal=False, trace=True, \
            #                       error_action='ignore', suppress_warnings=True, stepwise=True)

            model = pm.auto_arima(self.ts_data, exogenous=None, d=0, max_d=2, start_q=0, max_q=0, seasonal=False, trace=True, \
                                  error_action='ignore', suppress_warnings=True, stepwise=True)

            self.model = model
            model.summary()
            predict0 = model.predict(self.n_predict, exogenous=None)
            if not bAlreadyCreated:
                tsARIMA.set_ARIMA(model.order)
                self.predict = model.predict(  self.n_predict, exogenous=None)
                arima_dict = model.to_dict()
                msg2log(self.control_arima.__name__,"\n\nlogDictArima\n\n", self.f)
                logDictArima(arima_dict, 0, self.f)
            title = 'ARIMA: {} predict values'.format(self.nameModel)
            vector_logging(title, predict0, 16, self.f)

        (p, d, q, P, D, Q) = tsARIMA.get_SARIMA()
        bAlreadyCreated =1
        if (P==-1 and D==-1 and Q==-1 and p==-1 and d==-1 and q==-1):   # model should be created
            bAlreadyCreated = 0

        if (self.nameModel == 'control_seasonal_arima'):
            # initial parameters from Control object settings
            start_p_, start_d_, start_q_, max_p_, max_d_, max_q_, self.period,self.n_predict, self.discret, \
            self.ts_data = self.param
            # for seasonal part of the model the initial values are added hehre
            start_P_=0
            start_D_=0
            start_Q_=0
            max_P_  =2
            max_D_  =0
            max_Q_  =0

            if bAlreadyCreated: # tsARIMA was created ,initial parameters are being got from tsARIMA class static members.
                start_p_=p
                start_d_=d
                start_q_=q
                max_p_  =p
                max_d_  =d
                max_q_  =q
                start_P_=P
                start_D_=D
                start_Q_=Q
                max_P_  =P
                max_D_  =D
                max_Q_  =Q

            #self.ts_data=self.ts_data[:-1]

            # model = pm.auto_arima(self.ts_data, exogenous=None, start_p=start_p_, d=start_d_, start_q=start_q_,
            #                       max_p=max_p_, max_d=max_d_, max_q=max_q_, start_P=start_P_, D=start_D_, start_Q=start_Q_,
            #                       max_P=max_P_, max_D=max_D_, max_Q=max_Q_, max_order=16,     seasonal=True, m=self.period,
            #                       trace=True,   error_action='ignore', suppress_warnings=True,stepwise=True)
            model = pm.auto_arima(self.ts_data, start_p=0,max_p=1,start_q=0, max_q=1,d=0,max_d=1, D=0, max_D=0,start_P=0, max_P=1,start_Q=0,max_Q=0,
                                  exogenous=None, seasonal=True, m=self.period,
                                  trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
            self.model = model
            model.summary()
            predict1 = model.predict(self.n_predict, exogenous=None)

            if not bAlreadyCreated:
                (p,d,q) =model.order
                (P,D,Q,SS)= model.seasonal_order
                val =(p,d,q,P,D,Q)
                tsARIMA.set_SARIMA(val)
                arima_seas_dict=model.to_dict()
                msg2log(self.control_arima.__name__, "\n\nlogDictArima\n\n", self.f)
                logDictArima(arima_seas_dict,0,self.f)
            title = 'ARIMA: {} predict values'.format(self.nameModel)
            vector_logging(title, predict1, 16, self.f)

        return None

    def fitted_model_logging(self):
        pass
        title = self.__str__()
        # for charting
        x = np.zeros((len(self.ts_data) + self.n_predict))
        for i in range(len(x)):
            x[i] = i

        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')
        fig, ax = plt.subplots()

        ax.plot(x[:len(x) - self.n_predict],   self.ts_data, marker='', color=palette(0), label="Time series values")
        ax.plot(x[(len(x) - self.n_predict):], self.predict, marker='', color=palette(1), label="Predict values")
        plt.legend(loc=2, ncol=2)
        ax.set_title(title)

        # plt.title = title
        # plt.plot(x[:len(x) - self.n_predict], self.ts_data, c='blue')
        # plt.plot(x[(len(x) - self.n_predict):], self.predict, c='red')
        #plt.show(block=False)


        filePng=Path(PlotPrintManager.get_PredictLoggingFolder())/ ("Predict_{}_{}.png".format(self.nameModel, self.timeseries_name))

        plt.savefig(filePng)
        if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")

        if self.f is not None:
            message=f"""
                    Name Model    : {self.nameModel}
                    Accurace Model: {self.model.__str__()}
                      df_model :   {self.model.df_model()}
                      aic      :   {self.model.aic()} 
                      aicc     :   {self.model.aicc()}
            """
            msg2log(self.fitted_model_logging.__name__, message, self.f)

            prm_names = self.model.arima_res_.param_names
            prms = self.model.arima_res_.params
            self.f.write("ARIMA parameters\n")
            for i in range(len(prms)):
                self.f.write("{} {} = {}\n".format(i, prm_names[i], prms[i]))
            self.f.write('\nPredict\n')
            for i in range(len(self.predict)):
                self.f.write('{} {}\n'.format(i, self.predict[i]))

        return

    @exec_time
    def ts_analysis(self,NFFT: int)->(np.array,np.array,np.array,np.array):
        """

        :param NFFT: -segment size for FFT belongs to {16,32,64,  ..., 2^M}, M<12
        :return:
        """
        if NFFT>2048:
            NFFT=2048

        message = ""
        max_d = 5
        max_D = 2
        try:
            d = pm.arima.ndiffs(self.ts_data, 0.05, 'kpss', max_d)
            print("d={}".format(d))
            # p_order[1]=d
        except:
            pass
        try:
            D = pm.arima.nsdiffs(self.ts_data, self.period, max_D, 'ocsb')
            print("D={}".format(D))
        except ValueError:
            message = f"""
                        Oops!! That was no valid value..
                        Error : {sys.exc_info()[0]}
            """

        except:
            message = f"""
                                    Oops!! Unexpected error...
                                    Error : {sys.exc_info()[0]}
                        """

        finally:
            msg2log(self.ts_analysis.__name__, message, self.f)


        message =""
        delta = self.discret*60  # in sec
        N=len(self.ts_data)

        Fs=1.0/delta
        maxFreq= 1.0/(2*delta)
        stepFreqPSD = 1.0/(NFFT*delta)
        stepFreq = 1.0 / (N * delta)
        try:
            mean=np.mean(self.ts_data)
            std = np.std(self.ts_data)
        except ValueError:
            message = f"""
                            Oops!! That was no valid value..
                            Error : {sys.exc_info()[0]}
            """

        except:
            message = f"""
                            Oops!! Unexpected error...
                            Error : {sys.exc_info()[0]}
            """

        finally:
            msg2log(self.ts_analysis.__name__, message, self.f)


        message = f"""
                    Time series length     : {N}
                    FFT wnow length        : {NFFT}
                    Discretization, sec    : {self.discret*60}
                    Max.Frequency, Hz      : {maxFreq}
                    Freq. delta for PSD, Hz: {stepFreqPSD}
                    Mean value             : {mean}
                    Std. value             : {std}
            """
        msg2log(self.ts_analysis.__name__, message, self.f)

        for i in range(len(self.ts_data)):
            self.ts_data[i]=(self.ts_data[i]-mean)/std

        plt.subplot(311)
        t=np.arange(0, len(self.ts_data),1)
        plt.plot(t, self.ts_data)
        plt.subplot(312)
        message=""
        try:
            Pxx,freqs, line = plt.psd(self.ts_data,NFFT, Fs, return_line=True)
        except ValueError:
            message = f"""
                            Oops!! That was no valid value..
                            Error : {sys.exc_info()[0]}
            """

        except:
            message = f"""
                            Oops!! Unexpected error...
                            Error : {sys.exc_info()[0]}
            """

        finally:
            msg2log(self.ts_analysis.__name__, message, self.f)

        plt.subplot(313)
        message = ""
        try:
            maxlags=len(self.ts_data)/4
            if maxlags>250:
                maxlags=250


            alags, acorr, line,b = plt.acorr(self.ts_data, maxlags=maxlags,normed=True)
        except ValueError:
            message = f"""
                                    Oops!! That was no valid value..
                                    Error : {sys.exc_info()[0]}
                    """

        except:
            message = f"""
                                    Oops!! Unexpected error...
                                    Error : {sys.exc_info()[0]}
                    """

        finally:
            msg2log(self.ts_analysis.__name__, message, self.f)
        #plt.show(block=False)


        filePng = Path(PlotPrintManager.get_ControlLoggingFolder()) / (
            "{}_TS_SPD_AutoCorr_{}.png".format(self.nameModel, self.timeseries_name))

        plt.savefig(filePng)
        if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")
        psd_logging('{}_Power Spectral Density{}'.format(self.nameModel,  self.timeseries_name), freqs, Pxx)
        psd_logging('{}_Autocorrelation two-side{}'.format(self.nameModel,self.timeseries_name), alags, acorr)
        return (Pxx,freqs,acorr,alags)

def predict_sarima(ds,cp, n_predict):

    rcpower = ds.df[cp.rcpower_dset].values
    model = pm.auto_arima(rcpower, start_p=1,d=1, start_q=1,max_p=3,max_q=3, max_d=1, seasonal=True, start_P=1, \
                          D=1,start_Q=1,max_P=2,max_D=1, max_order=5, m=144, trace=True,stepwise=True)

    predict = model.fit_predict(rcpower, None, n_predict)
    # for charting
    x=np.zeros((len(rcpower) + n_predict))
    for i in range(len(x)):
        x[i]=i


    plt.plot(x[:len(x) - n_predict],rcpower, c='blue')
    plt.plot(x[(len(x) - n_predict):], predict, c='red')
    #plt.show(block=False)
    if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")


    return model

def predict_arima(ds,cp, n_predict):
    rcpower = ds.df[cp.rcpower_dset].values
    model = pm.auto_arima(rcpower, start_p=1,start_q=1,max_p=3,max_q=3, seasonal=False, d=1, trace=True,stepwise=True)
    predict = model.fit_predict(rcpower, None, n_predict)

    x=np.zeros((len(rcpower) + n_predict))
    for i in range(len(x)):
        x[i]=i


    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    fig, ax = plt.subplots()

    ax.plot(x[:len(x) - n_predict],   rcpower, marker='',color=palette(0), label="Time series values")
    ax.plot(x[(len(x) - n_predict):], predict, marker='',color=palette(0), label="Predict values")
    plt.legend(loc=2, ncol=2)


    plt.plot(x[:len(x) - n_predict],rcpower, c='blue')
    plt.plot(x[(len(x) - n_predict):], predict, c='red')
    #plt.show(block=False)
    if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")

    filePng = Path(PlotPrintManager.get_PredictLoggingFolder()) / (
        "Predict_{}.png".format(cp.rcpower_dset))
    plt.savefig(filePng)

    with open('arima_param.log','w') as fpar:
        fpar.write('df_model = {} aic = {} aicc = {}'.format( model.df_model(),model.aic(), model.aicc() ))
        prm_names=model.arima_res_.param_names
        prms = model.arima_res_.params
        fpar.write("ARIMA parameters\n")
        for i in range(len(prms)):
            fpar.write("{} {} = {}".format(i, prm_names[i], prms[i]))
        fpar.write('\nPredict\n')
        for i in range(len(predict)):
            fpar.write('{} {}'.format(i, predict[i]))
    return model






def test_arima(ds,cp):
    pass

    train, test = train_test_split(ds.rcpower,train_size=len(ds.rcpower) - 32)

    #fit model
    model = pm.auto_arima(train, start_p=1,start_q=0, max_p=5, max_q=1, seasonal=False,d=1, \
                          trace=True,error_action='ignore', suppress_warnings=True,stepwise=True)

    model.summary()
    with open("arima.log",'w') as fa:
      model.summary()
      fa.write(str(model.arima_res_) )

    #make forecast
    forecast = model.predict(test.shape[0])
    predict = model.fit_predict(ds.rcpower,None,10)
    #visualization of forecast

    x=np.arange(ds.rcpower.shape[0])
    if PlotPrintManager.isNeedDestroyOpenPlots():
        plt.close("all")
    plt.plot(x[:len(x)-32], train, c='blue')
    plt.plot(x[(len(x) - 32):], forecast, c='green')
    y=x
    for i in range(10):
        y=y.append(predict[i])
    plt.plot(y[len(x):], predict, c='red')
    #plt.show(block=False)
    if PlotPrintManager.isNeedDestroyOpenPlots():
        plt.close("all")
    return
