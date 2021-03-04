#!/usr/bin/python3

import copy
from pathlib import Path
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,metrics
from tensorflow.keras.layers.experimental import preprocessing

from predictor.utility import msg2log
from clustgelDL.auxcfg  import D_LOGS,log2All,exec_time
from canbus.api import TR_DICT,tr_names,T__, T_0, T_1, T0_, T00, T01, T1_, T10, T11
from offlinepred.api import logMatrix

STATUS_INIT =0
STATUS_SET  =1
STATUS_COMP =2
STATUS_FIT  =3
STATUS_EVAL =4
STATUS_SAVE =5
STATUS_LOAD =6

D_STATUSES={STATUS_INIT:"empty model",STATUS_SET:"model was set",STATUS_COMP:"compiled model",
            STATUS_LOAD:"loaded model",STATUS_FIT:"fitted model",STATUS_EVAL:"evaluated model",
            STATUS_SAVE:"saved model"}

class DigitalTwin(object):

    def __init__(self,name:str = "DigitalTwin",model_repository:str = "ckpt", f:object = None):
        self.model=tf.keras.Sequential()
        self.name=name
        self.model_repository=model_repository
        self.f=f
        self.status=STATUS_INIT
        self.fitting_counter=0
        self.validation_split =0.2
        self.loaded_model=None



    def loadModel(self):
        if Path(Path( self.model_repository)/Path("assets")).exists() and \
                Path(Path(self.model_repository) / Path("variables")).exists() and \
                Path(Path( self.model_repository) / Path("saved_model.pb")).exists():
            del self.loaded_model
            self.loaded_model=tf.keras.models.load_model(self.model_repository)
            self.status = STATUS_LOAD
            msg2log(None, "Model is successfully loaded from {}".format(self.model_repository), self.f)
            self.loaded_model.summary()
            if self.f is not None:
                self.loaded_model.summary(print_fn=lambda x: self.f.write(x + '\n'))
            self.status=STATUS_LOAD
        else:
            msg2log(None, "Model repository {} not found.".format(self.model_repository), self.f)
        return

    def saveModel(self):
        pass
        if self.status<STATUS_FIT:
            msg2log(None,"{} model is not fitted. The saving terminated.".format(self.name), self.f)
            return
        self.model.save(self.model_repository)
        msg2log(None, "{} model saved in {} ".format(self.name, self.model_repository), self.f)
        self.status = STATUS_SAVE

    def compile(self):
        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[metrics.SparseCategoricalAccuracy(), metrics.MeanSquaredError(),
                           metrics.KLDivergence()])
        self.status = STATUS_COMP
        self.model.summary()
        if self.f is not None:
            self.model.summary(print_fn=lambda x: self.f.write(x + '\n'))

    @exec_time
    def fit(self,x_train,y_train,batch_size:int = 64,epochs:int = 10):
        history = self.model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,
                                 validation_split = self.validation_split)
        self.status = STATUS_FIT
        self.fitting_counter=+1
        msg="\n\nTraining history for model {} after {}th stage of fitting.\n{}".format(self.name,self.fitting_counter,
                                                                                        history.history)
        msg2log(None, history.history, self.f)

        return history

    @exec_time
    def evaluate(self,x_test,y_test, batch_size:int = 64):
        results=self.model.evaluate(x_test,y_test,batch_size=batch_size)
        msg2log(None,"test loss,test_acc:\n{}".format(results))
        self.status = STATUS_EVAL

    @exec_time
    def predict(self, x_to_predict:np.array):
        if self.status==STATUS_FIT or self.status==STATUS_EVAL or self.status==STATUS_SAVE:
            msg2log(None, "Predicting by {} model".format(self.name), D_LOGS['predict'])
            predictions = self.model.predict(x_to_predict)
        elif self.status==STATUS_LOAD:
            msg2log(None, "Predicting by loaded model", D_LOGS['predict'])
            predictions = self.loaded_model.predict(x_to_predict)
        else:
            msg2log(None, "No models for predicting",self.f)

        # Generate arg maxes for predictions
        logMatrix(predictions, title="Predictions: belonging to the pattern of waveform transitions",
                  f=D_LOGS['predict'])
        classes = np.argmax(predictions, axis=1)
        n,=classes.shape
        classes=classes.reshape((n,1))
        logMatrix(classes, title="Predictions: argmax solution", f=D_LOGS['predict'])
        print(classes)
        return classes


class DigitalTwinMLP(DigitalTwin):

    def __init__(self,name:str = "DigitalTwin",model_repository:str = "ckpt", f:object = None):
        super().__init__(name=name,model_repository=model_repository, f=f)

    def setModel(self, n_input:int,normalize:preprocessing.Normalization,output:dict, hidden_layers:list):
        self.model=tf.keras.Sequential()
        self.model.add(keras.Input(shape=(n_input,)))
        if normalize is not None:
            self.model.add(normalize)
        for hl in hidden_layers:
            (unit,activation),=hl.items()
            # self.model.add(layers.Dense(unit,activation=activation))
            if activation is None or activation=="linear" or activation=="linear":
                self.model.add(layers.Dense(unit))
            else:
                self.model.add(layers.Dense(unit, activation=activation))
        (n_classes,activation), =output.items()
        if activation is None or activation == "linear" or activation == "linear":
            self.model.add(layers.Dense(n_classes))
        else:
            self.model.add(layers.Dense(n_classes,activation=activation))
        self.status = STATUS_SET

""" API for Digital Twins  """

def get_normalize(x:np.array)->preprocessing.Normalization:
    normalize = preprocessing.Normalization()
    normalize.adapt(x)
    return normalize


def dumpChunk(x: list = None, y: list = None, file_name: str = 'train.csv',  f: object = None) -> (str,int):
    """

    :param x: list of list [[...],...[]]
    :param y: list [...]
    :param name:
    :param chunk_number:
    :param f:
    :return:
    """

    xx = np.array(x)
    (n, input_size) = xx.shape
    yy = np.array(y)
    (n,) = yy.shape
    yy = yy.reshape((n, 1))
    xx = np.hstack((xx, yy))

    columns = ['x' + str(i) for i in range(input_size)]
    columns.append('y')
    pd.DataFrame(xx, columns=columns).to_csv(file_name, index=False)
    return file_name, input_size


def readChunk(filename: str = None,norm:bool = False,f:object = None) -> (np.array, np.array,
                                                                          preprocessing.Normalization):
    df = pd.read_csv(filename)
    df.head(n=10)
    print(df)
    features = df.copy()
    labels = features.pop('y')
    print(hex(id(features)))
    features = np.array(features)
    print(hex(id(features)))
    print(features)
    normalize=None
    if norm:
        normalize = get_normalize(features)

    return features, labels, normalize

""" Generation of the waveforma according to the bit stream. The waveforms form the Supervised Learning Data for 
classification neuron net. The desired data is a type of bit.
"""
def wfstreamGenSLD(mode: str = 'train', transition_stream: list = None, source:str = 'hist', fsample: float = 16e+6,
                   bitrate: float = 125000.0, slope: float = 0.1,  snr: float = 20, trwf_d: dict = TR_DICT,
                   title: str = "", dump_chunk_size:int = 256, repository:str="", start_chunk:int=0,
                   f: object = None)->(list,int) :

    chunk_list=[]
    packet_in_stream = -1
    anomaly_d = {}
    loggedSignal = np.array([])
    loggedHist = []
    numberLoggedBit = 16
    subtitle = "Fsample={} MHz Bitrate={} Kbit/sec SNR={} Db".format(round(fsample / 10e+6, 3),
                                                                     round(bitrate / 10e3, 2),
                                                                     round(snr, 0))
    """ random number for logged packet in the stream """
    loggedPacket = random.randrange(0, len(transition_stream))

    xSLD=[]
    ySLD=[]
    n_count=0
    chunk_number=start_chunk
    for packet in transition_stream:
        packet_in_stream += 1
        # here accumulated histogram per transition in following structue {transit:list}
        tr_hist = {T__: [], T_0: [], T0_: [], T_1: [], T1_: [], T00: [], T01: [], T10: [], T11: []}
        n_match_train = 0
        no_match_train = 0
        n_match_test = 0
        no_match_test = 0

        startLoggedBit = -1
        endLoggedBit = -1
        """ logged bits in the packet """
        if packet_in_stream == loggedPacket:
            startLoggedBit = random.randrange(0, len(packet))
            endLoggedBit = startLoggedBit + numberLoggedBit

        bit_in_packet = -1


        for transit in packet:
            bit_in_packet += 1
            cb = trwf_d[transit](fsample=fsample, bitrate=bitrate, slope=slope, SNR=snr, f=f)
            cb.genWF()
            arow=None
            desired=transit
            if source == 'hist':
                cb.histogram()
                arow =copy.copy(cb.hist)
            elif source=='wf':
                arow=copy.copy(cb.signal)
            else:
                arow = copy.copy(cb.signal)

            xSLD.append(arow.tolist())
            ySLD.append(desired)
            n_count+=1

            if n_count == dump_chunk_size:
                file_name=str(Path(repository)/Path("train_{}".format(chunk_number)).with_suffix(".csv"))
                file_name, m =dumpChunk(xSLD,ySLD,file_name=file_name,f=None)
                if chunk_number==start_chunk:
                    input_size=m
                chunk_list.append(file_name)
                xSLD = []
                ySLD = []
                n_count = 0
                chunk_number+=1

            """ select signals for charting """
            if bit_in_packet >= startLoggedBit and bit_in_packet < endLoggedBit:
                loggedSignal = np.concatenate((loggedSignal, cb.signal))
                loggedHist.append(cb.hist)

            if mode == 'train':
                tr_hist[transit].append(cb.hist)
                continue
    if chunk_number==start_chunk:
        file_name = str(Path(repository) / Path("train_{}".format(chunk_number)).with_suffix(".csv"))
        file_name, input_size=dumpChunk(xSLD, ySLD, file_name=file_name, f=None)
        chunk_list.append(file_name)
    log2All()
    return chunk_list, input_size
"""
hyperprm = {
        'normalize':True,
        'batch_size':32,
        'epochs':10,
        'input_size': {input_size: None},
        'n_classes': {n_classes: None},
        'hidden_layers': [{32: 'relu'}, {32: 'relu'}]
    }
"""
@exec_time
def digitalTwinTrain(model:object=DigitalTwinMLP, name:str="DigitalTwinMLP", input_size:int=128,hyperprm:dict=None,
                     chunk_list:list=None, repository:str="", f:object=None):
    pass
    digital_twin=model(name = "DigitalTwin",model_repository=repository, f=f)
    # hyperprm= {
    #     input:{input_size:None},
    #     output:{n_classes:None},
    #     hidden_layers:[{32:'relu'},{32:'relu'}]
    #           }

    """ parse hyper parameters"""
    d_input_size=hyperprm['input_size']
    (input_size,_),=d_input_size.items()
    n_classes = hyperprm['n_classes']
    hidden_layers=hyperprm['hidden_layers']
    normalize=None
    if hyperprm['normalize']:
        x,y,normalize = readChunk(filename=chunk_list[0], norm=True, f=f)

    digital_twin.setModel(input_size,normalize,n_classes,hidden_layers)
    digital_twin.compile()

    """ fit process """
    if len(chunk_list)>1:
        for item in chunk_list[:-1]:
            x,y,_ = readChunk(item,f=f)
            digital_twin.fit(x,y,batch_size=hyperprm['batch_size'],epochs=hyperprm['epochs'])
            log2All()
    else:
        x, y, _ = readChunk(chunk_list[0], f=f)
        digital_twin.fit(x, y, batch_size=hyperprm['batch_size'], epochs=hyperprm['epochs'])

    x, y, _ = readChunk(chunk_list[-1:][0], f=f)
    digital_twin.evaluate(x,y)
    digital_twin.saveModel()
    log2All()
    return

@exec_time
def digitalTwinPredict(model:object=DigitalTwinMLP, name:str="DigitalTwinMLP",
                     chunk_list:list=None, repository:str="", f:object=None):
    pass
    digital_twin=model(name = "DigitalTwin",model_repository=repository, f=f)
    digital_twin.loadModel()

    """ fit process """
    i=0
    if len(chunk_list)>1:
        for item in chunk_list[:-1]:
            x,y,_ = readChunk(item,f=f)
            predicted_transitions =digital_twin.predict(x)
            title="Bit Transitions Anomaly State in {} th chunk".format(i)
            plotTransitionStates(predicted_transitions=predicted_transitions, title=title)
            i+=1
            log2All()
    else:
        x, y, _ = readChunk(chunk_list[0], f=f)
        predicted_transitions=digital_twin.predict(x)
        title = "Bit Transitions Anomaly State in the 0 chunk"
        plotTransitionStates(predicted_transitions=predicted_transitions, title=title)
        log2All()
    x, y, _ = readChunk(chunk_list[-1:][0], f=f)
    predicted_transitions=digital_twin.predict(x)
    title = "Bit Transitions Anomaly State in the last chunk"
    plotTransitionStates(predicted_transitions=predicted_transitions, title=title)
    log2All()
    return

def plotTransitionStates(predicted_transitions:np.array=None, fsample:float=1.0, packetNumber:int=0,
               title:str="",subtitle:str=""):
    pass
    suffics = '.png'
    pred_transitions_png = Path(D_LOGS['plot'] / Path(title)).with_suffix(suffics)

    delta=1.0/fsample
    t=np.arange(0.0, len(predicted_transitions)*delta, delta)
    fig, ax = plt.subplots(figsize=(18, 5))
    high_bound=np.array([T11 for i in range(len(predicted_transitions))])
    ax.plot(t,predicted_transitions, color='r')
    ax.plot(t, high_bound, color='g')
    ax.set_xlabel('Bit number')
    ax.set_ylabel('Bit Transitions')
    ax.set_title(title)
    ax.grid(True)
    plt.savefig(pred_transitions_png)
    plt.close("all")
    return

if __name__ == "__main__":
    # normalize = preprocessing.Normalization()
    # dtw = DigitalTwin(24,normalize,10,[{32:'relu'},{16:'alu'}])
    pass
