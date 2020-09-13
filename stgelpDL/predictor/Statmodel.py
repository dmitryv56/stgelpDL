#!/usr/bin/python3

from predictor.predictor import Predictor
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import copy
from pickle import dump, load
from pathlib import Path
import sys
from predictor.utility import exec_time, msg2log, PlotPrintManager,psd_logging

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
        history =[]
        pass
        if self.nameModel=='seasonal_arima':
            self.predict = self.model.fit_predict(self.ts_data, None, self.n_predict)
        elif self.nameModel == 'best_arima':

            start_p_, start_d_, start_q_, max_p_, max_d_, max_q_, self.n_predict, \
            self.discret, self.ts_data = self.param

            model = pm.auto_arima(self.ts_data, exogenous=None, start_p=start_p_, d=start_d_, start_q=start_q_, \
                                  max_p=max_p_, max_d=max_d_, max_q=max_q_, seasonal=False, trace=True, \
                                  error_action='ignore', suppress_warnings=True, stepwise=True)
            self.predict = model.predict(self.n_predict, exogenous=None)
            self.model = model

        msg="\n\n{} was sucessfully fitted\n".format(self.nameModel)
        msg2log(self.fit_model.__name__, msg, self.f)

        return history

    @exec_time
    def predict_n_steps(self, n_predict_number):
        y=self.model.predict(n_predict_number,exogenous=None)

        return y

    def load_model_wrapper(self):
        self.load_model()
        return True

    def load_model(self):

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

        model_folder = Path(self.path2modelrepository) / self.timeseries_name / self.typeModel / self.nameModel
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        model_saving_file = Path(model_folder) / "arima.pkl"
        with open(model_saving_file, 'wb') as pkl:
            dump(self.model, pkl)
        msg ="Model saved in {} ".format(model_folder)
        msg2log(self.save_model.__name__, msg, self.f)

        return

class tsARIMA(Statmodel):
    pass
    _ar_order  = 0
    _ma_order  = 0
    _d_order   = 0
    _AR_order  = 0
    _MA_order  = 0
    _D_order   = 0
    _period    = 0
    _n_predict = 1
    _discret   = 600 # sec
    _ts_data   = None
    _param     = None

    def __init__(self,nameM, typeM, n_steps, n_epochs, f=None):
        self.predict=None
        super().__init__(nameM, typeM, n_steps, n_epochs, f)
        pass

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
        return "{} TimeSeries\nARIMA ({},{},{}) Seasonal ARM ({}{},{}) with period ={}".format( \
            self.timeseries_name, self.ar_order,self.d_order, self._ma_order, self.AR_order, self.D_order, \
            self.MA_order, self.period)

    @exec_time
    def seasonal_arima(self): #:
        title=self.__str__()
        # rcpower = ds.df[cp.rcpower_dset].values

        self.ar_order,self.d_order,self.ma_order, self.P_order, self.D_order, self.MA_order, self.period, \
        self.n_predict, self.discret, self.ts_data,  = self.param


        p_order=np.array([self.ar_order,self.d_order,self.ma_order])
        P_order = np.array([self.AR_order, self.D_order, self.MA_order, self.period])
        model = pm.arima.ARIMA( p_order,P_order)
        self.model = model


        return model

    @exec_time
    def best_arima(self): #:
        title=self.__str__()
        # rcpower = ds.df[cp.rcpower_dset].values

        start_p_, start_d_, start_q_, max_p_,  max_d_, max_q_, self.n_predict ,\
            self.discret, self.ts_data = self.param

        return None

    @exec_time
    def control_arima(self): #:
        title=self.__str__()
        # rcpower = ds.df[cp.rcpower_dset].values
        self.nameModel = 'control_arima'
        start_p_, start_d_, start_q_, max_p_,  max_d_, max_q_, self.n_predict ,\
            self.period, self.discret, self.ts_data = self.param
        ################################################################################3

        # model = pm.auto_arima(self.ts_data, exogenous=None, start_p=start_p_, d=start_d_, start_q=start_q_, \
        #                       max_p=max_p_, max_d=max_d_, max_q=max_q_, seasonal=False, trace=True, \
        #                       error_action='ignore', suppress_warnings=True, stepwise=True)
        #
        # self.model = model
        # model.summary()
        # self.predict = model.predict(  self.n_predict, exogenous=None)
        #
        # self.save_model()

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
        plt.show(block=False)
        if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")

        filePng=Path(PlotPrintManager.get_PredictLoggingFolder())/ ("Predict_{}_{}.png".format(self.nameModel, self.timeseries_name))

        plt.savefig(filePng)


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
    def ts_analysis(self):

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
            return


        delta = self.discret*60  # in sec
        N=len(self.ts_data)
        NFFT = 256
        Fs=1/(self.discret *60)
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
            return

        message = f"""
                    Time series length     : {N}
                    Discretization, sec    : {self.discret}
                    Max.Frequency, Hz      : {maxFreq}
                    Freq. delta for PSD, Hz: {stepFreqPSD}
                    Mean value             : {mean}
                    Std. value             : {std}
            """
        msg2log(self.ts_analysis.__name__, message, self.f)

        for i in range(len(self.ts_data)):
            self.ts_data[i]=(self.ts_data[i]-mean)/std

        plt.subplot(211)
        t=np.arange(0, len(self.ts_data),1)
        plt.plot(t, self.ts_data)
        plt.subplot(212)
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
            return

        plt.show(block=False)


        filePng = Path(PlotPrintManager.get_ControlLoggingFolder()) / (
            "SpectralDensity_{}.png".format(self.timeseries_name))

        plt.savefig(filePng)
        if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")
        psd_logging('Power Spectral Density', freqs, Pxx)
        return

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
    plt.show(block=False)
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
    plt.show(block=False)
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
    plt.show(block=False)
    if PlotPrintManager.isNeedDestroyOpenPlots():
        plt.close("all")
    return
