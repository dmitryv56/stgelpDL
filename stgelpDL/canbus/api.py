#!/usr/bin/python3

import sys
import copy
from pathlib import Path
from datetime import datetime,timedelta
import re
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pandas as pd
import subprocess
from pickle import dump,load


from predictor.utility import msg2log
from clustgelDL.auxcfg  import D_LOGS,log2All
from canbus.BF import BF

DB_NAME="canbas"
""" DB in repository.
DB fields are following:  
"""
DT="Date Time"
DUMP="Dump"
MATCH_KEY="Match_key"
METHOD="Method"
PKL="PKL"
REPOSITORY="Repository"
MISC="Misc"
DB_COLS=[DT, DUMP, MATCH_KEY, METHOD, PKL, REPOSITORY, MISC]

# number of randomly generated 'no signal' bits in bit stream
INSERTED_NO_SIGNAL=5
# phy layer state
SIG_  = 0
SIG_0 = 1
SIG_1 = 2

#transtions
T__ = 0   # no signal to no signal SIG_ -> SIG_
T_0 = 1   # SIG_ -> SIG_0
T0_ = 2   # SIG_0 -> SIG_
T_1 = 3   # SIG_ -> SIG_1
T1_ = 4   # SIG_1 _> SIG_
T00 = 5   # SIG_0 -> SIG_0
T01 = 6   # SIG_0 -> SIG_1
T10 = 7   # SIG_1 -> SIG_0
T11 = 8   # SIG_1 -> SIG_1
TAN = 9
tr_names={T__:"no signal waveform",
          T_0 :"transition to zero",
          T0_ : "transition from zero",
          T_1 : "transition to one",
          T1_ : "transition from one",
          T00 : "transition zero-zero",
          T01 : "transition zero-one",
          T10 : "transition one-zero",
          T11 : "transition one-one",
          TAN : "possible anomaly"
          }
tr_labels={T__:"**", T_0:"*0",T0_:"0*",T_1:"*1",T1_:"1*",T00:"00",T01:"01",T10:"10",T11:"11",TAN:"XX"}



""" Linear interpolator for 'slope' part of waveform."""
def interpSlopeWF(fsample:float=16e+06,bitrate:float=125000.0,slope:float=0.1,left_y:float=0.0, right_y:float=1.0,
                  f:object=None)->np.array:
    """

    :param fsample:
    :param bitrate:
    :param slope:
    :param left_y:
    :param right_y:
    :param f:
    :return:
    """
    n0 = int(slope * (fsample / bitrate))
    x = [0, n0]
    y = [left_y, right_y]
    xnew = np.arange(0, n0, 1)
    yinterp = np.interp(xnew, x, y)
    pure = np.array([yinterp[i] for i in range(n0)] + [right_y for i in range(n0, int(fsample / bitrate))])
    return pure

def transitionsPng(fsample:float=16e+06,bitrate:float=125000.0,snr:float=30.0,slope:float=0.2,f:object=None):
    transition_list=[T__WF,T_0WF,T_1WF,T0_WF,T00WF,T01WF,T1_WF,T10WF,T11WF]
    # fsample = 16e+06
    # bitrate = 125000.0
    # slope = 0.3
    SNR = 20
    f = None
    x = np.arange(0,int(fsample / bitrate))
    suffics = '.png'
    name="simulated_waveforms"
    waveform_png = Path(D_LOGS['plot'] / Path(name)).with_suffix(suffics)
    title="Transition Waveform( SNR={} DB, slope ={}, Fsample={} MHz, Bitrate ={} K/sec)".format( SNR, slope,
                                                                fsample/10e+6, bitrate/1e+3)
    fig,ax_array =plt.subplots(nrows=3,ncols=3,figsize = (18,5),sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.suptitle(title,fontsize=16)
    i=0
    for ax in np.ravel(ax_array):

        tobj=transition_list[i](fsample=fsample, bitrate=bitrate, slope=slope, SNR=SNR, f=f)
        tobj.genWF()
        auxTransitionsPng(ax, tobj, x)
        i=i+1

    plt.savefig(waveform_png)
    plt.close("all")
    return

def auxTransitionsPng(ax,tobj, x):
    # ln,=ax.plot(x, tobj.pure, x, tobj.signal)
    ln, = ax.plot(x, tobj.pure)
    ln, = ax.plot(x, tobj.signal)
    # ax[i, j].set_xlim(0, len(x) * 1 / fsample)
    ax.set_xlabel('time')
    ax.set_ylabel('Signal')
    ax.set_title(tobj.title)
    ax.grid(True)

    return ln

class canbusWF():
    """
    canbus
    """

    def __init__(self,fsample:float=16e+06,bitrate:float=125000.0,slope:float=0.1,SNR:int=3, f:object=None):
        self.fsample=fsample
        self.bitrate=bitrate
        self.slope=slope
        self.vcan_lD=1.5
        self.vcan_lR=2.5
        self.vcan_hR=2.5
        self.vcan_hD=3.5
        self.SNR=SNR    # 10*math.log(Vsignal/Vnoise)
        self.signal=None
        self.pure = None
        self.title=""
        self.f =f
        #hist
        self.h_min=self.vcan_lD-0.7
        self.h_max=self.vcan_hD+0.7
        self.h_step=0.5
        self.bins=[float(w/10) for w in range( int(self.h_min*10),  int((self.h_max+self.h_step)*10),
                                               int(self.h_step *10))]

        self.hist = None
        self.density = None
        pass

    """ Additive white Gaussian noise  (awgn)"""
    def awgn(self,signal:np.array=None):

        sigpower = sum([math.pow(abs(signal[i]),2) for i in range (len(signal))])
        sigpower=sigpower/len(signal)
        noisepower = sigpower/(math.pow(10,self.SNR/10))
        noise=math.sqrt(noisepower)*(np.random.uniform(-1,1,size=len(signal)))
        return noise

    def histogram(self):
        self.hist,_ = np.histogram(self.signal, self.bins, density=False)
        self.density, _ = np.histogram(self.signal, self.bins, density=True)
        return

    """ Random  signal waveform shift along t-axisto simulate the random latency in  bit stream.
    Max. shift is 10% from bit waveform period. shift_n -the number of signal samples by which the shift occurs is randomly 
    generated. shift_direction - the direction of the shift forward or back is randomized too.
    """
    def rndshift(self):
        if self.signal is None:
            return

        n,=self.signal.shape
        n_dist=int(n*0.1)
        shift_n=np.random.randint(n_dist,size=1)
        shift_direction = np.random.randint(3, size=1)
        signal_list=self.signal.tolist()
        if shift_direction ==0: # left shift, append
            for i in range(shift_n):
                signal_list.pop(0)
                signal_list.append(signal_list[-1])
        elif shift_direction==1: #right shift, insert at 0
            for i in range(shift_n):
                signal_list.pop(-1)
                signal_list.insert(0,signal_list[0])
        elif shift_direction == 2:
            for i in range(shift_n):
                signal_list.pop(-1)
                signal_list.insert(0, self.vcan_lR )

        del self.signal
        self.signal=np.array(signal_list)
        return



class T__WF(canbusWF):

    def __init__(self,fsample:float=16e+06,bitrate:float=125000.0,slope:float=0.1,SNR:int=3, f:object=None):
        super().__init__(fsample=fsample,bitrate=bitrate,slope=0.0,SNR=SNR, f=f)
        self.title="Transition _->_"

    def genWF(self):
        self.pure=np.array([self.vcan_hR for i in range(int(self.fsample/self.bitrate))])
        self.signal=np.add(self.pure,self.awgn(self.pure))

class T_0WF(canbusWF):

    def __init__(self, fsample: float = 16e+06, bitrate: float = 125000.0, slope: float = 0.1, SNR: int = 3,
                 f: object = None):
        super().__init__(fsample=fsample, bitrate=bitrate, slope=slope, SNR=SNR, f=f)
        self.title = "Transition _->'0'"

    def genWF(self):
        self.pure = interpSlopeWF(fsample=self.fsample, bitrate=self.bitrate, slope=self.slope,
                                  left_y=self.vcan_lD, right_y=self.vcan_hD, f=self.f)
        self.signal = np.add(self.pure, self.awgn(self.pure))

class T_1WF(canbusWF):

    def __init__(self, fsample: float = 16e+06, bitrate: float = 125000.0, slope: float = 0.1, SNR: int = 3,
                 f: object = None):
        super().__init__(fsample=fsample, bitrate=bitrate, slope=slope, SNR=SNR, f=f)
        self.title = "Transition _->'1'"

    def genWF(self):
        self.pure = interpSlopeWF(fsample=self.fsample, bitrate=self.bitrate, slope=self.slope,
                                  left_y=self.vcan_lR, right_y=self.vcan_lD, f=self.f)
        self.signal = np.add(self.pure, self.awgn(self.pure))

class T0_WF(canbusWF):

    def __init__(self, fsample: float = 16e+06, bitrate: float = 125000.0, slope: float = 0.1, SNR: int = 3,
                 f: object = None):
        super().__init__(fsample=fsample, bitrate=bitrate, slope=slope, SNR=SNR, f=f)
        self.title = "Transition '0'->_"

    def genWF(self):
        self.pure = interpSlopeWF(fsample=self.fsample, bitrate=self.bitrate, slope=self.slope,
                                  left_y=self.vcan_hD, right_y=self.vcan_lD, f=self.f)
        self.signal = np.add(self.pure, self.awgn(self.pure))

class T1_WF(canbusWF):

    def __init__(self, fsample: float = 16e+06, bitrate: float = 125000.0, slope: float = 0.1, SNR: int = 3,
                 f: object = None):
        super().__init__(fsample=fsample, bitrate=bitrate, slope=slope, SNR=SNR, f=f)
        self.title = "Transition '1'->_"

    def genWF(self):

        self.pure= interpSlopeWF(fsample=self.fsample, bitrate=self.bitrate, slope=self.slope,
                                 left_y=self.vcan_lD, right_y=self.vcan_lR, f=self.f)
        self.signal=np.add(self.pure,self.awgn(self.pure))

class T00WF(canbusWF):

    def __init__(self, fsample: float = 16e+06, bitrate: float = 125000.0, slope: float = 0.1, SNR: int = 3,
                 f: object = None):
        super().__init__(fsample=fsample, bitrate=bitrate, slope=slope, SNR=SNR, f=f)
        self.title = "Transition '0'->'0'"

    def genWF(self):
        self.pure=np.array([self.vcan_hD for i in range(int(self.fsample/self.bitrate))])
        self.signal=np.add(self.pure,self.awgn(self.pure))

class T11WF(canbusWF):

    def __init__(self, fsample: float = 16e+06, bitrate: float = 125000.0, slope: float = 0.1, SNR: int = 3,
                 f: object = None):
        super().__init__(fsample=fsample, bitrate=bitrate, slope=slope, SNR=SNR, f=f)
        self.title = "Transition '1'->'1'"

    def genWF(self):
        self.pure=np.array([self.vcan_lD for i in range(int(self.fsample/self.bitrate))])
        self.signal=np.add(self.pure,self.awgn(self.pure))

class T10WF(canbusWF):

    def __init__(self, fsample: float = 16e+06, bitrate: float = 125000.0, slope: float = 0.1, SNR: int = 3,
                 f: object = None):
        super().__init__(fsample=fsample, bitrate=bitrate, slope=slope, SNR=SNR, f=f)
        self.title = "Transition '1'->'0'"

    def genWF(self):

        self.pure= interpSlopeWF(fsample=self.fsample, bitrate=self.bitrate, slope=self.slope,
                                 left_y=self.vcan_lD, right_y=self.vcan_hD, f=self.f)
        self.signal=np.add(self.pure,self.awgn(self.pure))

class T01WF(canbusWF):

    def __init__(self, fsample: float = 16e+06, bitrate: float = 125000.0, slope: float = 0.1, SNR: int = 3,
                 f: object = None):
        super().__init__(fsample=fsample, bitrate=bitrate, slope=slope, SNR=SNR, f=f)
        self.title = "Transition '0'->'1'"

    def genWF(self):

        self.pure= interpSlopeWF(fsample=self.fsample, bitrate=self.bitrate, slope=self.slope,
                                 left_y=self.vcan_hD, right_y=self.vcan_lD, f=self.f)
        self.signal=np.add(self.pure,self.awgn(self.pure))

""" Waveform per transition  dictionary """
TR_DICT={T__: T__WF,
         T_0: T_0WF,
         T_1: T_1WF,
         T0_: T0_WF,
         T00: T00WF,
         T01: T01WF,
         T1_: T1_WF,
         T10: T10WF,
         T11: T11WF}

""" Return list of  following dict 
{'DateTime':<Date Time>,
 'IF':<interface>>, 
 'ID':<canbus packet ID>,
 'Data':<canbus packet data>,
 'Packet':<ID | data> in hexa,
 'bitstr_list':<list of bits>
 'bit_str':<string of bits>
}
"""
def readChunkFromCanBusDump(offset_line:int =0, chunk_size:int=128,canbusdump:str=None, f:object=None)->list:

    parsed_list=[]
    if canbusdump is None or canbusdump=="" or not Path(canbusdump).exists():
        return parsed_list
    line_count=0
    last_line=offset_line + chunk_size
    with open(canbusdump,'r') as fcanbus:
        while line_count<offset_line:
            line = fcanbus.readline()
            if not line:
                return parsed_list
            line_count+=1
        while line_count<last_line:
            line = fcanbus.readline()
            if not line:
                return parsed_list
            line_count+=1
            parsed_list.append(parseCanBusLine(line))

    return parsed_list

"""This function parses string to 'DateTime', 'interface', 'packet ID' and 'packet Data'.
The concatenation of two elements 'ID' and 'Data' forms an additional return element 'packet'.
The packet string converts to list bit strings. Every two symbols are converted to the bit string.
All return items are merged into a dictionary. 
"""
def parseCanBusLine(line:str=None, f:object=None)->dict:
    if line is None:
        return {}
    aitems=line.split(' ')
    itemDateTime=re.search(r'\((.*?)\)',line).group(1)
    itemData=re.search('(?<=#)\w+',line).group(0)
    aitemID=aitems[2].split('#')
    itemID=aitemID[0]
    itemIF=aitems[1]
    if len(itemID)%2 != 0:
        itemID='0'+itemID
    if len(itemData)%2 !=0:
        itemData='0'+itemData
    itemPacket=itemID +itemData

    bitstr_list =packet2bits(packet=itemPacket,f=f)
    bit_str=''.join(bitstr_list)
    """ random generation 0-INSERTED_NO_SIGNAL 'no signal' bits marked as *"""
    nrnd=random.randrange(0,INSERTED_NO_SIGNAL+1)
    insnosigb=''.join(["*" for i in range(nrnd+1)])
    if len(insnosigb)>0:
        bit_str=bit_str+insnosigb

    return {'DateTime':itemDateTime,'IF':itemIF, 'ID':itemID,'Data':itemData,'Packet':itemPacket,
            'bitstr_list':bitstr_list,'bit_str':bit_str}

""" This function forms a list of bits string from a packet data
Every two symbols (two nibbles  or byte) is hex number which is converted to bit array.
The function returns the list of bit strings.
For example, packet is '6B6B00FF'
'6B'=107 =>'1101011'
'6B'=107 =>'1101011'
'00'=0 =>'00000000' 
'FF'=255 => '11111111'
The result list contains ['1101011','1101011','00000000' ,'11111111']
"""

def packet2bits(packet:str=None,f:object=None)->list:

    start=0
    step=2
    bits_list=[]
    for i in range(start,len(packet),step):

        bss="{0:b}".format(int( packet[start:start+step], 16)).zfill(8)
        bits_list.append(bss)
        start=start +step
    return bits_list

""" Transform bit to  the state, the type of waveform being be generated, according by current bit and previous state
                           st=R(bit, prev_st).
The set of states is {T__,T_0,T_1,T0_.T1_,T00,T01,T10,T11}, the current bit belongs to { '0' , '1', '*'-no signal}. 
"""
def transitionRules(prev_state:int, current_bit:str)->(int, int):
    """

    :param prev_state:
    :param current_bit:
    :return:
    """
    if prev_state==SIG_:
        if current_bit=='0':
            transition=T_0
        elif current_bit=='1':
            transition=T_1
        elif current_bit=='*':
            transition=T__
        else:
            transition=T__

    elif prev_state==SIG_0:
        if current_bit == '0':
            transition = T00
        elif current_bit == '1':
            transition = T01
        elif current_bit == '*':
            transition = T0_
        else:
            transition = T0_
    elif prev_state==SIG_1:
        if current_bit == '0':
            transition = T10
        elif current_bit == '1':
            transition = T11
        elif current_bit == '*':
            transition = T1_
        else:
            transition = T1_
    if current_bit=='0':
        new_state=SIG_0
    elif current_bit=='1':
        new_state=SIG_1
    elif current_bit == '*':
        new_state=SIG_
    else:
        new_state=SIG_
    return transition, new_state

""" Transform bit to transition according by rules 
transition=R(bit,prev_state), 
where states are { SIG_-no signal, SIG_0- zero signal, SIG_1- one signal} and
transition belongs to {T__, T_0, T_1, T0_ , T00, T01, T1_, T10, T11 }.
Return list of transition and new prev_state for next packet."""
def genTransition(prev_st:int=SIG_, bit_str:str=None, f:object=None)->(list,int):

    """ transition array generation"""
    transition=[]
    st=prev_st
    for i in range(len(bit_str)):
        tr,st=transitionRules(st, bit_str[i])
        transition.append(tr)
    prev_st=SIG_
    return transition,prev_st

def logPackets(ld:list,offset_line:int=0,chunk_size:int=16):

    msg = "\nChunk start: {} Chunk size: {}\n".format(offset_line,chunk_size)
    msg2log(None,msg,D_LOGS['block'])
    msg = "{:<30s} {:<9s} {:^8s} {:^8s} {:<16s} ".format('Date Time','Interface','ID', 'Data','Packet')
    for dct in ld:
        msg="{:<30s} {:<9s} {:<8s} {:<8s} {:<16s} ".format(dct['DateTime'], dct['IF'], dct['ID'], dct['Data'],
                                                           dct['Packet'])
        msg2log(None,msg,D_LOGS['block'])
    return

""" For chunk generate list of trasitions."""
def trstreamGen(canbusdump:str="", offset_line:int=0, chunk_size:int=16, prev_state:int=SIG_, f:object=None)->list:
    # offset_line = offset_line
    # chunk_size = chunk_size
    # canbusdump = canbusdump
    transition_stream = []
    ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=chunk_size, canbusdump=canbusdump, f=f)
    if not ld:
        return transition_stream
    logPackets(ld=ld,offset_line=offset_line,chunk_size=chunk_size)

    for dct in ld:
        transition, prev_state = genTransition(prev_st=prev_state, bit_str=dct['bit_str'], f=f)
        transition_stream.append(transition)

    return transition_stream

""" Generation of the waveforma according to the bit stream. 
A statistical estimate of the histogram is calculated for each waveform.
At the training stage  within one packet (frame), histograms are averaged over the type of bit transitions.
The resulting histogram concatenated with type of the bit is added to Blooom Filter. (T.B.D. - add to DB too).
At the test stage no averaging. The histogram concatenated with the type of the bit is checked with BF. If no matc there
is an anomaly symptom. 

"""
def wfstreamGen(mode:str='train',transition_stream:list=[],fsample:float=16e+6,bitrate:float=125000.0, slope:float=0.1,
                snr:float=20, trwf_d:dict=TR_DICT,bf:BF=None, title:str="", repository:str="", f:object=None)->dict:
    """

    :param mode:
    :param transition_stream:
    :param fsample:
    :param bitrate:
    :param slope:
    :param snr:
    :param trwf_d:
    :param bf:
    :param title:
    :param f:
    :return:
    """
    packet_in_stream = -1
    anomaly_d={}
    loggedSignal = np.array([])
    loggedHist   = []
    numberLoggedBit = 16
    subtitle="Fsample={} MHz Bitrate={} Kbit/sec SNR={} Db".format(round(fsample/10e+6,3), round(bitrate/10e3,2),
                                                                   round(snr,0))

    """ random number for logged packet in the stream """
    loggedPacket=random.randrange(0,len(transition_stream))
    sum_match_train  = 0
    sum_no_match_train = 0
    sum_match_test   = 0
    sum_no_match_test  = 0
    for packet in transition_stream:
        packet_in_stream+=1
        # here accumulated histogram per transition in following structue {transit:list}
        tr_hist ={T__: [],  T_0: [],T0_: [],T_1: [], T1_: [],T00: [],T01: [],T10: [], T11: []}
        n_match_train=0
        no_match_train=0
        n_match_test = 0
        no_match_test = 0

        startLoggedBit=-1
        endLoggedBit = -1
        """ logged bits in the packet """
        if packet_in_stream == loggedPacket:
            startLoggedBit=random.randrange(0,len(packet))
            endLoggedBit  =startLoggedBit + numberLoggedBit


        bit_in_packet=-1

        for transit in packet:
            bit_in_packet+=1
            cb=trwf_d[transit](fsample=fsample,bitrate=bitrate, slope=slope, SNR=snr, f=f)
            cb.genWF()
            cb.histogram()
            """ select signals for charting """
            if bit_in_packet>=startLoggedBit and bit_in_packet<endLoggedBit:
                loggedSignal=np.concatenate((loggedSignal,cb.signal))
                loggedHist.append(cb.hist)

            if mode=='train':
                tr_hist[transit].append(cb.hist)
                continue
            """ hist to word """
            if bf is None:
                continue
            word=hex(transit).lstrip("0x")+"_"
            word = word + ''.join([hex(vv).lstrip("0x").rstrip("L") for vv in cb.hist.tolist()])

            if not bf.check_item(word):
                msg="no match in DB for {} transition  in {} packet".format(transit, packet_in_stream)
                msg2log("Warning!",msg,D_LOGS['predict'])
                msg2log("Warning!", msg, f)
                anomaly_d[packet_in_stream]={transit:tr_names[transit]}
                no_match_test += 1
            else:
                msg2log(None, "Match for {} transition  in {} packet".format(transit, packet_in_stream), D_LOGS['predict'])
                n_match_test += 1

        if mode=='test':
            msg2log(None, "\nTest\nmatch: {} no match: {}".format(n_match_test,no_match_test), D_LOGS['predict'])

        if mode=='train':
            """ histogram averaging """
            for key,val in tr_hist.items():
                if not val:
                    continue

                allhists=np.array(val)
                avehist=np.average(allhists,axis=0)
                if bf is None:
                    continue
                word = hex(key).lstrip("0x") + "_"
                word = word + ''.join([hex(int(vv)).lstrip("0x").rstrip("L") for vv in avehist.tolist()])
                if bf.check_item(word):
                    msg2log(None,"Match for {} transition  in {} packet".format(key,packet_in_stream),D_LOGS['train'])
                    n_match_train+=1
                else:
                    bf.add_item(word)
                    no_match_train+=1

            msg2log(None,"\nTrain\nmatch: {} no match:{}".format(n_match_train,no_match_train),D_LOGS['train'])

        sum_match_train +=n_match_train
        sum_no_match_train +=no_match_train
        sum_match_test +=n_match_test
        sum_no_match_test +=no_match_test


    if mode=="train":
        msg2log(None, "\nTrain summary for SNR={} DB\nmatch: {} no match:{}".format(snr, sum_match_train, sum_no_match_train),
                D_LOGS['train'])
        bf.save(repository)
    if mode=="test":
        msg2log(None, "\nTest summary for SNR = {} DB\nmatch: {} no match:{}".format(snr,sum_match_test, sum_no_match_test),
                    D_LOGS['predict'])
    log2All()

    if len(loggedSignal)>0:
        plotSignal(mode=mode, signal=loggedSignal, packetNumber=0, fsample=fsample, startBit=startLoggedBit,
                       title=title, subtitle=subtitle)

    return anomaly_d






def plotSignal(mode:str="train", signal:np.array=None, fsample:float=1.0, packetNumber:int=0, startBit:int=0,
               title:str="",subtitle:str=""):
    pass
    suffics = '.png'
    signal_png = Path(D_LOGS['plot'] / Path(title)).with_suffix(suffics)

    delta=1.0/fsample
    t=np.arange(0.0, (len(signal)-1)*delta, delta)

    n=min(len(t),len(signal))
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(t[:n],signal[:n], color='r')
    ax.set_xlabel('time')
    ax.set_ylabel('Signal wavefors')
    ax.set_title(title)
    ax.grid(True)
    plt.savefig(signal_png)
    plt.close("all")
    return

""" Get number of lines in dump file.
This function is executed ib the subprocess"""
def file_len(fname)->int:

    n=-1
    if Path(fname).exists():
        if sys.platform.startswith('linux'):
            try:
                p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                                          stderr=subprocess.PIPE)
                result, err = p.communicate()
                if p.returncode != 0:
                    #    we will not raise any exception , raise IOError(err)
                    n= -2
                n= int(result.strip().split()[0])
            except:
                pass
            finally:
                pass
        elif sys.platform.startswith('win'):
            fr=open(fname,'r')
            n=0
            while 1:
                line=fr.readline()
                if line is None:
                    break
                n+=1
            fr.close()
        else:
            n=-3

    return n


def dict2csv(d:dict=None, folder:str="", title:str="", dset_name:str=None, match_key:str='ID', f:object=None):
    if d is None:
        msg2log(None,"No dictionary {} {} for saving".format(title,match_key),f)
        return
    if dset_name is None or len(dset_name)<1 or ".csv" not in dset_name:
        msg2log(None,"Dtaset name is not set correctly {}".format(dset_name),f)
        return

    df=pd.DataFrame(d)
    df.to_csv(dset_name)
    msg2log(None,"{} dictionary for {} saved in {}".format(title,match_key,dset_name),f)

    return

def dict2pkl(d:dict=None, folder:str="", title:str="", match_key:str='ID', f:object=None)->(str,str):
    if d is None:
        msg2log(None,"No dictionary {} {} for saving".format(title,match_key),f)
        return

    file_pkl=Path(Path(folder)/Path("{}_{}".format(title,match_key))).with_suffix(".pkl")
    f_pkl=open(str(file_pkl),"wb")
    dump(d,f_pkl)

    msg2log(None,"{} dictionary for {} saved in {}".format(title,match_key,str(file_pkl)),f)
    pkl_stem=file_pkl.stem
    return pkl_stem, str(file_pkl)


def pkl2dict( folder: str = "", title: str = "", match_key: str = 'ID', pkl_stem:str="", f: object = None):



    file_pkl = Path(Path(folder) / Path("{}".format(pkl_stem))).with_suffix(".pkl")
    if not file_pkl.exists():
        msg="Serialized dictionary {} for {} -match key was not found in {} repository".format(pkl_stem,
                                                                                               match_key,folder)
        msg2log(None,msg,f)
        return None
    f_pkl = open(str(file_pkl), "rb")
    d=load(f_pkl)

    msg2log(None, "{} dictionary for {} loaded from {}".format(title, match_key, str(file_pkl)), f)

    return d

"""" statistical estimation for observed data"""

def mleexp(target_dict:dict=None, mleexp_dict:dict=None, n_min:int=5, title:str="Train path", f:object=None):

    msg="{}\n,Rare packets, no maximum likelihood estimation for exponential distribution of time gaps between " +\
        " packets appearing.".format(title)
    msg2log(None, msg, D_LOGS['cluster'])
    for key,vlist in target_dict.items():
        if len(vlist)<n_min:
            msg=f"""Packet with matched mey: {key}  is rare event: {len(vlist)} appearings"""
            msg2log(None,msg,D_LOGS['cluster'])
            continue
        l_duration=[vlist[i]-vlist[i-1] for i in range(1,len(vlist))]
        n=len(l_duration)
        sum_items=float(sum(l_duration))/1e06   # in seconds
        mle_lambda=float(n)/sum_items
        mle_var_lambda=(mle_lambda*mle_lambda)/float(n)
        mleexp_dict[key]={"n":n,"mle":mle_lambda,"var":mle_var_lambda,"sample":l_duration}
    return

def KL_decision(train_mleexp:dict=None, test_mleexp:dict=None, title:str="Anomaly packet",f:object=None)->list:

    trainSet=set(train_mleexp)
    testSet=set(test_mleexp)
    anomaly_list=[]
    chi2_1_05=3.84
    for key in trainSet.intersection(testSet):
        anomaly_counter=0
        train_val=train_mleexp[key]
        test_val=test_mleexp[key]
        lst_val=train_val['sample']+test_val['sample']
        xmean=np.array(lst_val).mean()
        xtrain=np.array(train_val['sample']).mean()
        xtest = np.array(test_val['sample']).mean()
        ntrain=train_val['n']
        ntest = test_val['n']
        KL2I12=ntrain*(xtrain-xmean)*(xtrain-xmean)/xmean + ntest*(xtest-xmean)*(xtest-xmean)/xmean
        KLJ12 =0.5*KL2I12 + 0.5 *( ntrain * (xtrain - xmean) * (xtrain - xmean) / xtrain + ntest * (xtest - xmean) * (
                    xtest - xmean) / xtest)

        if KL2I12>chi2_1_05 or KLJ12 > chi2_1_05:
            anomaly_counter+=1
            anomaly_list.append({'matched_key':key,"2I(1:2)":KL2I12,"J(1,2)":KLJ12, "chi2(1,0.05)":chi2_1_05,
                                 "train":train_val,"test":test_val,})

    return anomaly_list

def manageDB(repository:str=None, db:str=None,op:str='select',d_query:dict={}, f:object=None)->dict:
    file_db=Path(Path(repository)/Path(db)).with_suffix(".csv")
    if not file_db.exists():
        createDB(file_db=file_db, f=f)
    if op=='select':
        d_res = selectDB(file_db=file_db, d_query=d_query,f=f)
    elif op=='insert':
        d_res = insertDB(file_db= file_db, d_query = d_query, f = f)
        pass
    elif op=='update':
        pass
    elif op=='log':
        pass
    else:
        pass

    return d_res

def createDB(file_db:str=None, f:object=None):
    df=pd.DataFrame(columns=DB_COLS)
    df.to_csv(file_db,index=False)
    msg2log(None,"DB created {}".format(file_db),)
    return

def selectDB(file_db:str=None, d_query:dict=None,f:object=None)->dict:
    if file_db is None or not Path(file_db).exists() or d_query is None or len(d_query)==0:
        return None


    d_res={}
    l_res=[]   #list of dict
    df=pd.read_csv(file_db)
    for index,row in df.iterrows():
        keys=list(row.keys())
        if dictIndict(row,d_query,f=f):
            l_res.append(row)

    if len(l_res)>0:
        msg=f"""
Query: {d_query}
Selected: {l_res}
"""
        msg2log(None,msg,f)
        d_res=dict(l_res[-1] ) # select last item in list. The item gas Series -type and so it should be casted to dict.
    return d_res

def insertDB(file_db:str=None, d_query:dict=None,f:object=None)->dict:
    if file_db is None or not Path(file_db).exists() or d_query is None or len(d_query)==0:
        return None

    keys=list(d_query.keys())
    d_insert={item:"" for item in DB_COLS}
    for item in keys:
        d_insert[item]=d_query[item]
    d_insert[DT]=pd.Timestamp.now()
    df1=pd.DataFrame(d_insert,index=[0])


    df = pd.read_csv(file_db)
    df=df.append(df1,ignore_index=True)
    df.to_csv(file_db)
    msg2log(None,"Insert {}.\nAdded {} to DB in {} file".format(d_query,d_insert,file_db),f)

    return None

def fillQuery(dt:object="", dump_log:str="", match_key:str="ID", method:str="", pkl:str="", repository:str="",
              misc:str="", f:object=None)->dict:

    d_query={DUMP:dump_log,MATCH_KEY:match_key,METHOD:method,PKL:pkl,REPOSITORY:repository,MISC:misc}
    d_query[DT]="" if dt is None else dt

    msg2log(None, "Filling query {}".format(d_query),f)
    return d_query

def fillQuery2Test( match_key:str="ID", method:str="",repository:str="", f:object=None)->dict:

    d_query={MATCH_KEY:match_key,METHOD:method,REPOSITORY:repository}


    msg2log(None, "Filling query for 'Test stage' : {}".format(d_query),f)
    return d_query

def getSerializedModelData(d:dict=None, f:object=None)->dict:

    return pkl2dict(folder=d[REPOSITORY], title=d[METHOD], match_key= d[MATCH_KEY], pkl_stem= d[PKL], f=f)


def dictIndict(row:dict,d_query:dict,f:object=None)->bool:
    keys=list(d_query.keys())
    match=False
    for item in keys:
        if row[item]==d_query[item]:
            match=True
        else:
            match=False
            break
    return match


if __name__=="__main__":


    transitionsPng()
    pass
    cb=T10WF(fsample= 16e+06, bitrate = 125000.0, slope= 0.1, SNR = 20, f= None)
    cb.genWF()
    x=np.arange(int(cb.fsample/cb.bitrate))
    plt.plot(x, cb.signal)
    plt.plot(x, cb.pure)
    plt.xlabel("time (s)")
    plt.ylabel("signal (arb.u.)")
    plt.show()

    time = np.linspace(0, 200, num=2000)
    pure = 20 * np.sin(time / (2 * np.pi))
    noise=cb.awgn(pure)
    x = np.add(pure,noise)
    plt.plot(time, x)
    plt.plot(time, pure)
    plt.xlabel("time (s)")
    plt.ylabel("signal (arb.u.)")
    plt.show()

    offset_line=4
    chunk_size=2
    canbusdump="/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
    ft = open("log.log", 'w+')
    ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=chunk_size, canbusdump=canbusdump, f=ft)
    for item in ld:
        msg2log(None,item,ft)
        transition,prev_st =genTransition(prev_st = SIG_, bit_str=item['bit_str'], f=ft)
        plt.plot(transition)
        plt.show()

    ft.close()


