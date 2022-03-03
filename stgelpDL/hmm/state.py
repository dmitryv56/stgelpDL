#!/usr/bin/python3


import copy
from pathlib import Path
import logging

import numpy as np
import pandas as pd

# from hmm.api_acppgeg import plotStates
from predictor.utility import msg2log,tsBoundaries2log

logger = logging.getLogger(__name__)

#OFF = 0
D__ = 0
W__ = 1
DW_ = 2
#T__ = 4
DT_ = 3
WT_ = 4
DWT = 5
#P__ = 8
#DP_ = 9
WP_ = 6
DWP = 7
OTH = 8



_Dxxx = 0
iDxxx = _Dxxx + 1
#dDxxx = 2
xxxW_ = iDxxx + 1
xxiW_ = xxxW_ + 1
xxdW_ = xxiW_ + 1
_D_W_ = xxdW_ + 1
_DiW_ = _D_W_ + 1
_DdW_ = _DiW_ + 1
iD_W_ = _DdW_ + 1
# iDiW_ = 10
iDdW_ = iD_W_ + 1
dD_W_ = iDdW_ + 1
dDiW_ = dD_W_ + 1
# dDdW_ = 14
_DxxT = dDiW_ + 1
iDxxT = _DxxT + 1
dDxxT = iDxxT + 1
xxxWT = dDxxT + 1
xxiWT = xxxWT + 1
xxdWT = xxiWT + 1
_D_WT = xxdWT + 1
_DiWT = _D_WT + 1
_DdWT = _DiWT + 1
iD_WT = _DdWT + 1
# iDiWT = 25
iDdWT = iD_WT + 1
dD_WT = iDdWT + 1
dDiWT = dD_WT + 1
# dDdWT = 29
xxxWP = dDiWT + 1
xxiWP = xxxWP + 1
xxdWP = xxiWP + 1
_D_WP = xxdWP + 1
_DiWP = _D_WP + 1
_DdWP = _DiWP + 1
iD_WP = _DdWP + 1
iDiWP = iD_WP + 1
iDdWP = iDiWP + 1
dD_WP = iDdWP + 1
dDiWP = dD_WP + 1
# dDdWP = dDiWP + 1

DATA_COL  ="Imbalance"
DATE_COL ="Date Time"
DIESEL = 'Diesel_Power'
WIND   = 'WindTurbine_Power'
HYDRO  = 'Hydrawlic'
CSV_FILE = "~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"


class state():

    def __init__(self,states_dict,f=None):
        self._log = logging.getLogger(__class__.__name__)
        self.states_dict=copy.copy(states_dict)
        self.states=[]
        self.d_cnt ={}
        self.d_m_std ={}
        self.f=f
        self.printListState()


    def printListState(self):
        msg = "{:^20s} {:^20s} {:^40s}".format("Index State", "Short State Name", "Full State Name")
        msg2log(None, "\n\n{}".format(msg), self.f)
        self._log.info(msg)
        for i in range(len(self.states_dict)):
            shName, Name = self.getStateNamesByIndex(i)
            msg = "{:>20d} {:^20s} {:^40s}".format(i, shName, Name)
            msg2log(None, msg, self.f)
            self._log.info(msg)
        msg2log(None, "\n\n", self.f)
        return

    def getStateNamesByIndex(self,indx_state):
        shortStateName = "UNDF"
        fullStateName = "UNDF"
        for key, val in self.states_dict.items():
            if key == indx_state:
                (shortStateName, fullStateName) = list(val.items())[0]

                break
        return shortStateName, fullStateName

    def stateMeanStd(self,ds,data_col_name):
        for key,_ in self.states_dict.items():
            self.d_m_std[key]=[0.0, 0.0]
        for i in range(len(self.states_dict)):
            aux_ar=np.array([ds[data_col_name].values[k] for k in range(len(ds[data_col_name])) if self.states[k]==i])
            (n,)=aux_ar.shape
            if n == 1:
                self.d_m_std[i]=[aux_ar[0],1e-06]
            else:
                self.d_m_std[i]=[np.mean(aux_ar), np.std(aux_ar)]
            del aux_ar

        self._log.info(self.d_m_std)
        return

    def logMeanStd(self):

        msg="{:<5s} {:^10s} {:^10s}".format("State","mean", "s.t.d.")
        msg2log(None,msg, self.f)
        self._log.info(msg)
        for i in range(len(self.d_m_std)):
            short_state_name, _ = self.getStateNamesByIndex(i)
            mstd_list =self.d_m_std[i]
            msg="{:<5s} {:>10.4g} {:>10.4g}".format(short_state_name, mstd_list[0], mstd_list[1])
            msg2log(None,msg,self.f)
            self._log.info(msg)
        return

    def readDataset(self, csv_file: str, data_col_name: str, dt_col_name: str, other_cols: list) -> (
        pd.DataFrame, list):

        ds = pd.read_csv(csv_file)
        tsBoundaries2log(data_col_name, ds, dt_col_name, data_col_name, self.f)
        for item in other_cols:
            tsBoundaries2log(item, ds, dt_col_name, item, self.f)
        diesel_col_name = DIESEL
        wind_col_name = WIND
        hydro_col_name = HYDRO
        self.createStates(ds, data_col_name, diesel_col_name, wind_col_name, hydro_col_name)

        ds["hidden_states"] = self.states

        return ds, self.states



class state5(state):

    def __init__(self, f=None):


        states_dict =self.setStateDict()
        super().__init__(states_dict,f)

    def setStateDict(self):

        STATES_DICT = {0: {'BLNC': 'Balance of consumer power and generated power'},
                       1: {'LACK': 'Lack electrical power'},
                       2: {'SIPC': 'Sharp increase in power consumption'},
                       3: {'EXCS': 'Excess electrical power'},
                       4: {'SIGP': 'Sharp increase in generation power'}
                       }

        return STATES_DICT

    """ Linear extrapolation for sharp increase in the power detection.
        It gets 3 consecutive  time series values [Y(k-1),Y(k),Y(k+1)]. First, for the median point, a decision is made 
        about its belonging to one of the primary states of states according to the following rules:
        {y ~ 0 => balance}, {y >> 0 => dominance consumers power},{y << 0 => dominance of generation power}.
        Second , for median point Y(k) , is cheсked an  occurrence  of a sharp increase in generation (state 'SIGP', index 4)
        or consumption (state 'SIPC', index 2)  with using a linear extrapolation Yextrp(k+1) =F(Y(k),Y(k-1)).
        If Y(k)>Y(k-1), i.e. the function grows, and Y(k+1) > Yextrp(K+1) then Y(k) belongs to 'SIPC' state (the sharp i
        ncrease in power consumption).
        If Y(k)<Y(k-1), i.e. decrease function, and Y(k+1)<Yextrp(k+1) then Y(k) is in 'SIGP' state (Sharp increase in 
        generation power )/    

    """

    def getLnExtrapState(self,yk: list) -> int:
        if len(yk) != 3:
            return None
        if yk[1] < -1e-03:
            ret = 3  # Excess electrical power
        elif yk[1] > 1e-03:
            ret = 1  # Lack electrical power
        else:
            ret = 0  # balance of consumer power and generated power

        if yk[1] > yk[0] and yk[2] > 2.0 * yk[1] - yk[0]: ret = 2  # Sharp increase in customer power
        if yk[1] < yk[0] and yk[2] < 2.0 * yk[1] - yk[0]:  ret = 4  # Sharp increase in generation power

        return ret

    def getBoundState(self, yk: float) -> int:
        if yk < -1e-03:
            ret = 3  # dominance of generation power
        elif yk > 1e-03:
            ret = 1  # dominance of consumer power
        else:
            ret = 0  # balance of consumer power and generated power
        return ret

    def readDataset(self,csv_file: str, data_col_name: str, dt_col_name: str, other_cols:list) -> (pd.DataFrame, list):

        ds = pd.read_csv(csv_file)
        tsBoundaries2log(data_col_name, ds, dt_col_name, data_col_name, self.f)

        self.createStates(ds,data_col_name,DIESEL, WIND, HYDRO)

        ds["hidden_states"] = self.states

        return ds, self.states

    def createStates(self, ds,data_col_name,diesel_col_name, wind_col_name, hydro_col_name):

        for i in range(len(self.states_dict)):
            self.d_cnt[i]=0
        self.states.append(0)  # first sample
        self.d_cnt[0] += 1
        for i in range(1,len(ds[data_col_name])-1):
            yk =[ds[data_col_name].values[i-1],ds[data_col_name].values[i], ds[data_col_name].values[i+1]]
            ret = self.getLnExtrapState(yk)
            if ret is None:
                msg = "Invalid state for {} sample: {} ".format(i, yk)
                self._log.error(msg)
                msg = ''
                self.states.append(0)   # assign to balance state
                self.d_cnt[0] += 1
            else:
                self.states.append(ret)
                self.d_cnt[ret] += 1
        self.states.append(0)  # last sample
        self.d_cnt[0] += 1
        return



class state8(state):

    def __init__(self,f=None):
        states_dict = self.setStateDict()
        super().__init__(states_dict,f)

    def setStateDict(self):

        STATES_DICT = {#0:  {'OFF': 'Diesels Off, WindTurbines Off, Hydrawlic Off'},
                       D__:  {'D__': 'Diesels, WindTurbine Off, Hydrawlic Off'},
                       W__:  {'W__': 'Diesels Off,WindTurbine, Hydrawlic Off'},
                       DW_:  {'DW_': 'Diesels, WindTurbine, Hydrawlic Off'},
                       #3:  {'T__': 'Diesels Off, WindTurbines Off, Hydrawlic Turbine'},
                       DT_:  {'DT_': 'Diesels, WindTurbines Off, Hydrawlic Turbine'},
                       WT_:  {'WT_': 'Diesels Off, WindTurbines, Hydrawlic Turbine'},
                       DWT:  {'DWT': 'Diesels , WindTurbines, Hydrawlic Turbine'},
                       #8:  {'P__': 'Diesels Off, WindTurbines Off, Hydrawlic Pump'},
                       #9:  {'DP_': 'Diesels, WindTurbines Off, Hydrawlic Pump'},
                       WP_: {'WP_': 'Diesels Off, WindTurbines, Hydrawlic Pump'},
                       DWP: {'DWP': 'Diesels , WindTurbines, Hydrawlic Pump'}

                       }

        return STATES_DICT

    def readDataset(self,csv_file: str, data_col_name: str, dt_col_name: str, other_cols:list) -> (pd.DataFrame, list):

        ds = pd.read_csv(csv_file)
        tsBoundaries2log(data_col_name, ds, dt_col_name, data_col_name, self.f)
        for item in other_cols:
            tsBoundaries2log(item, ds, dt_col_name, item, self.f)
        diesel_col_name = DIESEL
        wind_col_name   = WIND
        hydro_col_name  = HYDRO
        self.createStates(ds,data_col_name,diesel_col_name, wind_col_name, hydro_col_name)

        ds["hidden_states"] = self.states

        return ds, self.states

    def createStates(self, ds,data_col_name,diesel_col_name, wind_col_name, hydro_col_name):

        for i in range(len(self.states_dict)):
            self.d_cnt[i]=0
        for i in range(len(ds[data_col_name])):
            d = ds[diesel_col_name].values[i]
            w = ds[wind_col_name].values[i]
            h = ds[hydro_col_name].values[i]
            # if d == 0 and w == 0 and h == 0:
            #     self.states.append(OFF)
            #     self.d_cnt[OFF]+=1
            if d > 0 and w == 0 and h == 0:
                self.states.append(D__)
                self.d_cnt[D__]+=1
            elif d == 0 and w > 0 and h == 0:
                self.states.append(W__)
                self.d_cnt[W__]+=1
            elif d > 0 and w > 0 and h == 0:
                self.states.append(DW_)
                self.d_cnt[DW_]+=1
            # elif d == 0 and w == 0 and h > 0:
            #     self.states.append(T__)
            #     self.d_cnt[T__]+=1
            elif d > 0 and w == 0 and h > 0:
                self.states.append(DT_)
                self.d_cnt[DT_] += 1
            elif d == 0 and w > 0 and h > 0:
                self.states.append(WT_)
                self.d_cnt[WT_] += 1
            elif d > 0 and w > 0 and h > 0:
                self.states.append(DWT)
                self.d_cnt[DWT] += 1
            # elif d == 0 and w == 0 and h < 0:
            #     self.states.append(P__)
            #     self.d_cnt[P__]+=1
            # elif d > 0 and w == 0 and h < 0:
            #     self.states.append(DP_)
            #     self.d_cnt[DP_]+=1
            elif d == 0 and w > 0 and h < 0:
                self.states.append(WP_)
                self.d_cnt[WP_]+=1
            elif d > 0 and w > 0 and h < 0:
                self.states.append(DWP)
                self.d_cnt[DWP]+=1
            else:
                msg ="Invalid state for {} sample: {} {} {}".format(i, d,w,h)
                self._log.error(msg)
                msg = ''
        return




code_val=lambda d,incd,decd,w,incw,decw,h,t,p: d + (incd<<1) + (decd<<2) + (w<<3)+ (incw<<4) + (decw<<5) + (h<<6) \
                                               + (t<<7) + (p<<8)
class state36(state):

    def __init__(self,f=None):
        states_dict = self.setStateDict()
        self.tngY = 0.577
        super().__init__(states_dict,f)

    def setStateDict(self, ):

        STATES_DICT = {
                       _Dxxx:  {'_Dxxx': 'Diesels, WindTurbine Off, Hydrawlic Off'},
                       iDxxx:  {'iDxxx': 'IncPower Diesels, WindTurbine Off, Hydrawlic Off'},
                       # dDxxx:  {'dDxxx': 'DecPower Diesels, WindTurbine Off, Hydrawlic Off'},
                       xxxW_:  {'xxxW_': 'Diesels Off, WindTurbine, Hydrawlic Off'},
                       xxiW_:  {'xxiW_': 'Diesels Off, IncPower WindTurbine, Hydrawlic Off'},
                       xxdW_:  {'xxdW_': 'Diesels Off, DecPower WindTurbine, Hydrawlic Off'},
                       _D_W_:  {'_D_W_': 'Diesels, WindTurbine, Hydrawlic Off'},
                       _DiW_:  {'_DiW_': 'Diesels, IncPower WindTurbine, Hydrawlic Off'},
                       _DdW_:  {'_DdW_': 'Diesels, DecPower WindTurbine, Hydrawlic Off'},
                       iD_W_:  {'iD_W_': 'IncPower Diesels, WindTurbine, Hydrawlic Off'},
                       # iDiW_:  {'iDiW_': 'IncPower Diesels, IncPower WindTurbine, Hydrawlic Off'},
                       iDdW_:  {'iDdW_': 'IncPower Diesels, DecPower WindTurbine, Hydrawlic Off'},
                       dD_W_:  {'dD_W_': 'DecPower Diesels, WindTurbine, Hydrawlic Off'},
                       dDiW_:  {'iDiW_': 'DecPower Diesels, IncPower WindTurbine, Hydrawlic Off'},
                       # dDdW_:  {'dDdW_': 'DecPower Diesels, DecPower WindTurbine, Hydrawlic Off'},
                       _DxxT:  {'_DxxT': 'Diesels, WindTurbine Off, Hydrawlic Turbine'},
                       iDxxT:  {'iDxxT': 'IncPower Diesels, WindTurbine Off,Hydrawlic Turbine'},
                       dDxxT:  {'dDxxT': 'DecPower Diesels, WindTurbine Off, Hydrawlic Turbine'},
                       xxxWT:  {'xxxWT': 'Diesels Off, WindTurbine, Hydrawlic Turbine'},
                       xxiWT:  {'xxiWT': 'Diesels Off, IncPower WindTurbine, Hydrawlic Turbine'},
                       xxdWT:  {'xxdWT': 'Diesels Off, DecPower WindTurbine, Hydrawlic Turbine'},
                       _D_WT:  {'_D_WT': 'Diesels, WindTurbine, Hydrawlic Turbine'},
                       _DiWT:  {'_DiWT': 'Diesels, IncPower WindTurbine, Hydrawlic Turbine'},
                       _DdWT:  {'_DdWT': 'Diesels, DecPower WindTurbine, Hydrawlic Turbine'},
                       iD_WT:  {'iD_WT': 'IncPower Diesels, WindTurbine, Hydrawlic Turbine'},
                       # iDiWT:  {'iDiWT': 'IncPower Diesels, IncPower WindTurbine, Hydrawlic Turbine'},
                       iDdWT:  {'iDdWT': 'IncPower Diesels, DecPower WindTurbine, Hydrawlic Turbine'},
                       dD_WT:  {'dD_WT': 'DecPower Diesels, WindTurbine, Hydrawlic Turbine'},
                       dDiWT:  {'dDiWT': 'DecPower Diesels, IncPower WindTurbine, Hydrawlic Turbine'},
                       # dDdWT:  {'dDdWT': 'DecPower Diesels, DecPower WindTurbine, Hydrawlic Turbine'},
                       xxxWP:  {'xxxWP': 'Diesels Off, WindTurbine, Hydrawlic Pump'},
                       xxiWP:  {'xxiWP': 'Diesels Off, IncPower WindTurbine, Hydrawlic Pump'},
                       xxdWP:  {'dWxxP': 'Diesels Off, DecPower WindTurbine, Hydrawlic Pump'},
                       _D_WP:  {'_D_WP': 'Diesels, WindTurbine, Hydrawlic Pump'},
                       _DiWP:  {'_DiWP': 'Diesels, IncPower WindTurbine, Hydrawlic Pump'},
                       _DdWP:  {'_DdWP': 'Diesels, DecPower WindTurbine, Hydrawlic Pump'},
                       iD_WP:  {'iD_WP': 'IncPower Diesels, WindTurbine, Hydrawlic Pump'},
                       iDiWP:  {'iDiWP': 'IncPower Diesels, IncPower WindTurbine, Hydrawlic Pump'},
                       iDdWP:  {'iDdWP': 'IncPower Diesels, DecPower WindTurbine, Hydrawlic Pump'},
                       dD_WP:  {'dD_WP': 'DecPower Diesels, WindTurbine, Hydrawlic Pump'},
                       dDiWP:  {'dDiWP': 'DecPower Diesels, IncPower WindTurbine, Hydrawlic Pump'},
                       # dDdWP:  {'dDdWP': 'DecPower Diesels, DecPower WindTurbine, Hydrawlic Pump'}
                      }

        return STATES_DICT

    def createStates(self, ds,data_col_name,diesel_col_name, wind_col_name, hydro_col_name):


        for i in range(len(self.states_dict)):
            self.d_cnt[i]=0
        d = 1 if ds[diesel_col_name].values[0]>0.0 else 0
        w = 1 if ds[wind_col_name].values[0]>0.0 else 1
        t = 1 if ds[hydro_col_name].values[0]>0.0 else 1
        p = 1 if ds[hydro_col_name].values[0] < 0.0 else 0
        h =  0
        cval = code_val(d, 0, 0, w, 0, 0, h, t, p)

        if        d and not w and not t and not p: self.states.append(_Dxxx); self.d_cnt[_Dxxx] += 1
        elif  not d and     w and not t and not p: self.states.append(xxxW_); self.d_cnt[xxxW_] += 1
        elif      d and     w and not t and not p: self.states.append(xxxW_); self.d_cnt[xxxW_] += 1
        elif  not d and not w and     t and not p: self.states.append(_DxxT); self.d_cnt[_DxxT] += 1
        elif  not d and     w and     t and not p: self.states.append(xxxWT); self.d_cnt[xxxWT] += 1
        elif      d and     w and     t and not p: self.states.append(_D_WT); self.d_cnt[_D_WT] += 1
        elif  not d and     w and not t and     p: self.states.append(xxxWP); self.d_cnt[xxxWP] += 1
        elif      d and     w and not t and p:     self.states.append(_D_WP); self.d_cnt[_D_WP] += 1
        else:
            msg="d={} w={} h={} t={} p={}".format(d,w,h,t,p)
            msg2log("i=0",msg,self.f)





        for i in range(1, len(ds[data_col_name])):


            incd = 1 if ds[diesel_col_name].values[i]-ds[diesel_col_name].values[i-1]>self.tngY else 0
            decd = 1 if ds[diesel_col_name].values[i] - ds[diesel_col_name].values[i - 1] < -self.tngY else 0
            d = 1 if ds[diesel_col_name].values[i] > 0.0 and incd==0 and decd==0 else 0



            incw = 1 if ds[wind_col_name].values[i]-ds[wind_col_name].values[i-1] > self.tngY else 0
            decw = 1 if ds[wind_col_name].values[i] - ds[wind_col_name].values[i - 1] < -self.tngY else 0
            w = 1 if ds[wind_col_name].values[i] > 0.0 and incw==0 and decw==0 else 0

            t = 1 if ds[hydro_col_name].values[i]>0.0 else 0
            p = 1 if ds[hydro_col_name].values[i] <0.0 else 0
            h = 0
            cval=code_val(d,incd,decd,w,incw,decw,h,t,p)




            if cval == 1:  self.states.append(_Dxxx); self.d_cnt[_Dxxx] += 1
            elif cval == 2:  self.states.append(iDxxx); self.d_cnt[iDxxx] += 1
            # elif cval == 4:  self.states.append(dDxxx); self.d_cnt[dDxxx] += 1
            elif cval == 8:  self.states.append(xxxW_); self.d_cnt[xxxW_] += 1
            elif cval == 16: self.states.append(xxiW_); self.d_cnt[xxiW_] += 1
            elif cval == 32: self.states.append(xxdW_); self.d_cnt[xxdW_] += 1
            elif cval == 9:  self.states.append(_D_W_); self.d_cnt[_D_W_] += 1
            elif cval == 17:  self.states.append(_DiW_); self.d_cnt[_DiW_] += 1
            elif cval == 33:  self.states.append(_DdW_); self.d_cnt[_DiW_] += 1
            elif cval == 10:  self.states.append(iD_W_); self.d_cnt[iD_W_] += 1
            # elif cval == 18:  self.states.append(iDiW_); self.d_cnt[iDiW_] += 1
            elif cval == 34:  self.states.append(iDdW_); self.d_cnt[iDdW_] += 1
            elif cval == 12:  self.states.append(dD_W_); self.d_cnt[dD_W_] += 1
            elif cval == 20:  self.states.append(dDiW_); self.d_cnt[dDiW_] += 1
            # elif cval == 36:  self.states.append(dDdW_); self.d_cnt[dDdW_] += 1
            elif cval == 129: self.states.append(_DxxT); self.d_cnt[_DxxT] += 1
            elif cval == 130: self.states.append(iDxxT); self.d_cnt[iDxxT] += 1
            elif cval == 132: self.states.append(dDxxT); self.d_cnt[dDxxT] += 1
            elif cval == 136: self.states.append(xxxWT); self.d_cnt[xxxWT] += 1
            elif cval == 144: self.states.append(xxiWT); self.d_cnt[xxiWT] += 1
            elif cval == 160: self.states.append(xxdWT); self.d_cnt[xxdWT] += 1

            elif cval == 137: self.states.append(_D_WT); self.d_cnt[_D_WT] += 1
            elif cval == 145: self.states.append(_DiWT); self.d_cnt[_DiWT] += 1
            elif cval == 161: self.states.append(_DdWT); self.d_cnt[_DdWT] += 1

            elif cval == 138: self.states.append(iD_WT); self.d_cnt[iD_WT] += 1
            # elif cval == 146: self.states.append(iDiWT); self.d_cnt[iDiWT] += 1
            elif cval == 162: self.states.append(iDdWT); self.d_cnt[iDdWT] += 1

            elif cval == 140: self.states.append(dD_WT); self.d_cnt[dD_WT] += 1
            elif cval == 148: self.states.append(dDiWT); self.d_cnt[dDiWT] += 1
            # elif cval == 164: self.states.append(dDdWT); self.d_cnt[dDdWT] += 1

            elif cval == 264: self.states.append(xxxWP); self.d_cnt[xxxWP] += 1
            elif cval == 272: self.states.append(xxiWP); self.d_cnt[xxiWP] += 1
            elif cval == 288: self.states.append(xxdWP); self.d_cnt[xxdWP] += 1
            elif cval == 265: self.states.append(_D_WP); self.d_cnt[_D_WP] += 1
            elif cval == 273: self.states.append(_DiWP); self.d_cnt[_DiWP] += 1
            elif cval == 289: self.states.append(_DdWP); self.d_cnt[_DdWP] += 1

            elif cval == 266: self.states.append(iD_WP); self.d_cnt[iD_WP] += 1
            elif cval == 274: self.states.append(iDiWP); self.d_cnt[iDiWP] += 1
            elif cval == 290: self.states.append(iDdWP); self.d_cnt[iDdWP] += 1

            elif cval == 268: self.states.append(dD_WP); self.d_cnt[dD_WP] += 1
            elif cval == 276: self.states.append(dDiWP); self.d_cnt[dDiWP] += 1
            # elif cval == 292: self.states.append(dDdWP); self.d_cnt[dDdWP] += 1
            else:
                msg="cval={} d={} incd={} decd={} w={} incw={} decw={} h={} t={} p={}".format(cval,d,incd,decd,w,incw,decw,h,t,p)
                msg2log("i={} ".format(i),msg,self.f)
                self._log.info(msg)
        self.printListState()
        return

class stateX(state36):
    def __init__(self,f=None):

        super().__init__(f)
        self.tngY = 1.1
        pass


if __name__ == "__main__":

    with open("Logs/states.log",'w') as fl:

        # st=state8(fl)
        # st = state36(fl)
        st = stateX(fl)
        ds,states=st.readDataset(CSV_FILE,DATA_COL,DATE_COL,[DIESEL,WIND,HYDRO])
        ds.to_csv("Imbalance_ElHierro_hiddenStates.csv")
        st.printListState()
        st.stateMeanStd(ds,DATA_COL)
        st.logMeanStd()

        period=512
        for start_index in range(0, len(ds), period):
            end_index=start_index+period
            if end_index>=len(ds) : end_index =  len(ds)-1
            title="{}_States_{}_from_{}_till_{}".format(DATA_COL,len(st.states_dict),start_index,end_index)
            path_to_file= Path("Logs/")
            # plotStates(ds, DATA_COL, st.states, title, path_to_file, start_index=start_index,end_index=end_index, f=fl)

    fl.close()
