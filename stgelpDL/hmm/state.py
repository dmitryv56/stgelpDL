#!/usr/bin/python3


import copy
from pathlib import Path

import numpy as np
import pandas as pd

# from hmm.api_acppgeg import plotStates
from predictor.utility import msg2log,tsBoundaries2log

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

DATA_COL  ="Imbalance"
DATE_COL ="Date Time"
DIESEL = 'Diesel_Power'
WIND   = 'WindTurbine_Power'
HYDRO  = 'Hydrawlic'
CSV_FILE = "~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"


class state():

    def __init__(self,states_dict,f=None):
        self.states_dict=copy.copy(states_dict)
        self.states=[]
        self.d_cnt ={}
        self.d_m_std ={}
        self.f=f

    def printListState(self):
        msg = "{:^20s} {:^20s} {:^40s}".format("Index State", "Short State Name", "Full State Name")
        msg2log(None, "\n\n{}".format(msg), self.f)
        for i in range(len(self.states_dict)):
            shName, Name = self.getStateNamesByIndex(i)
            msg = "{:>20d} {:^20s} {:^40s}".format(i, shName, Name)
            msg2log(None, msg, self.f)
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
            aux_ar=np.array([ds[data_col_name].values[k] for k in range(len(ds[data_col_name]))    if self.states[k]==i])
            self.d_m_std[i]=[np.mean(aux_ar), np.std(aux_ar)]
            del aux_ar
        return

    def logMeanStd(self):

        msg="{:<5s} {:^10s} {:^10s}".format("State","mean", "s.t.d.")
        msg2log(None,msg, self.f)
        for i in range(len(self.d_m_std)):
            short_state_name, _ = self.getStateNamesByIndex(i)
            mstd_list =self.d_m_std[i]
            msg="{:<5s} {:>10.4f} {:>10.4f}".format(short_state_name, mstd_list[0], mstd_list[1])
            msg2log(None,msg,self.f)
        return



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
        Second , for median point Y(k) , is cheked an  occurrence  of a sharp increase in generation (state 'SIGP', index 4)
        or consumption (state 'SIPC', index 2)  with using a linear extrapolation Yextrp(k+1) =F(Y(k),Y(k-1)).
        If Y(k)>Y(k-1), i.e. the function grows, and Y(k+1) > Yextrp(K+1) then Y(k) belongs to 'SIPC' state (the sharp i
        ncrease in power consumption).
        If Y(k)<Y9k-1), i.e. decrease function, and Y(k+1)<Yextrp(k+1) then Y(k) is in 'SIGP' state (Sharp increase in 
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
                pass
        return


if __name__ == "__main__":

    with open("Logs/states.log",'w') as fl:

        st=state8(fl)
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
