#!/usr/bin/python3

""" Generation 'programmed power' data for a DER power  time series.

"""

import os
import sys

import pandas as pd
from pathlib import Path
import numpy as np
import copy

from dsetanalyse.simpleHMM import simpleHMM,poissonHMM

# constants
DIESEL_PWR  = "Diesel_Power"
REAL_DEMAND = "Real_demand"
WIND_PWR    = "WindGen_Power"
HYDRTRB_PWR = "HydroTurbine_Power"
TURBINE_PWR = "Turbine_Power"
PUMP_PWR    = "Pump_Power"

D_EUIP_PWR ={DIESEL_PWR:1.02, REAL_DEMAND:1.0, WIND_PWR:1.0,TURBINE_PWR:1.0,PUMP_PWR:0.5}


def main(src_csv:str=None,repository:Path=None):
    data_col_names = [DIESEL_PWR, REAL_DEMAND,  WIND_PWR, TURBINE_PWR,PUMP_PWR]
    discret = 10



    test_size = 144


    dt_col_name = "Date Time"
    data_col_name = "Diesel_Power"
    state_col_name = "State_{}".format(data_col_name)
    # src_csv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_Diesel.csv"
    dst_csv_name=str(Path(src_csv).stem)
    dst_csv="{}_programmed.csv".format(dst_csv_name)
    df = pd.read_csv(src_csv)
    n=len(df)
    train_size=n-test_size

    # split HYDRTRB_PWR to TURBINE_PWR and PUMP_PWR
    v=df[HYDRTRB_PWR]
    v1=[df[HYDRTRB_PWR][i] if df[HYDRTRB_PWR][i]>=0.0 else 0.0 for i in range(n) ]
    v2 = [-df[HYDRTRB_PWR][i] if df[HYDRTRB_PWR][i] < 0.0 else 0.0 for i in range(n)]
    df[TURBINE_PWR]= v1
    df[PUMP_PWR]   = v2

    with open("../dev/Programmed.log", 'w') as ff:
        for item in data_col_names:
            data_col_name=item
            programmed_item="Programmed_{}".format(item)
            programmed_item_3pnts = "Programmed_3pnts_{}".format(item)
            programmed_item_hmm = "Programmed_hmm_{}".format(item)
            equip_pwr=D_EUIP_PWR[item]
            equip_pwrKwt = D_EUIP_PWR[item]*1000.0

            if (item==HYDRTRB_PWR):
                v=[]
                for i in range(n):
                    vv=df[item].values[i]
                    if vv>0:
                        equip=equip_pwr
                    else:
                        equip=0.5

                    v.append(int(round(int(round(vv/equip)) * equip)))


            else:
                v=[ int(round(int(round(df[item].values[i]/equip_pwr)) * equip_pwr))  for i in range(n)]

                # v = [ round((int(round((df[item].values[i]*1000.0) / equip_pwrKwt)) * equip_pwrKwt)/1000,0) for i in range(n)]
            df[programmed_item]=v

            v_3pnts=threePointsSmooth(v_src=v, f=ff)
            df[programmed_item_3pnts] = v_3pnts
            v_3pnts=[]

            if (item != HYDRTRB_PWR):

                # v_hmm=hmmSmooth(v_src=v, v_observations=df[item].values, item=item, f=ff)

                v_hmm = phmmSmooth(v_src=v, v_observations=df[item].values, item=item, f=ff)

                df[programmed_item_hmm] = v_hmm

    df1=mergeHydroTurbine(df=df, f=ff)

    df1.to_csv(dst_csv)

    return

def mergeHydroTurbine(df:pd.DataFrame=None,item:str=HYDRTRB_PWR, item1:str=TURBINE_PWR,item2:str=PUMP_PWR,
                      tmpl_list:list =["Programmed_","Programmed_3pnts_","Programmed_hmm_"],
                      f:object=None)->pd.DataFrame:

    n=len(df)
    for tmpl in tmpl_list:
        dst  ="{}_{}".format(tmpl,item)
        src1 ="{}_{}".format(tmpl,item1)
        src2 = "{}_{}".format(tmpl, item2)

        v=[]
        for i in range(n):
            if df[item][i]>=0.0:
                v.append(df[src1][i])
            else:
                v.append(-df[src2][i])

        df[dst]=v

    return df

def threePointsSmooth(v_src:list=None, f:object=None)->list:

    v=copy.copy(v_src)
    n=len(v)
    for i in range(n-2):
        if v[i]==v[i+2]:
            v[i+1]=v[i]

    return v

def hmmSmooth(v_src:list=None,v_observations:list=None, item:str="pwr",f:object=None)->list:


    vmin, vmax, n, dn, states,v, vv_observations= fillMissedStates(v_src=v_src, v_observations = v_observations, f=f)

    shmm = simpleHMM(name="{}_HMM".format(item), states=states, num_steps=n, f=f)

    shmm.setInit(train_states_seq=v)
    shmm.setTransition(train_states_seq=v)
    shmm.setEmission(train_states_seq=v, train_observations=vv_observations)

    shmm.createModel()
    shmm.fitModel(observations=vv_observations)
    posterior_mode = shmm.viterbiPath(observations=vv_observations)
    vv=posterior_mode.numpy()
    if vmin!=0:
        for i in range(n):
            vv[i]+=vmin
    if dn!=0:
        for k in range(dn):
            vv=np.delete(vv,-1)
    return vv.tolist()

def phmmSmooth(v_src:list=None,v_observations:list=None, item:str="pwr",f:object=None)->list:

    lambdas=[]
    vmin, vmax, n, dn, states,v, vv_observations= fillMissedStates(v_src=v_src, v_observations = v_observations, f=f)

    #coding observations
    for i in range(n):

        vv_observations[i]=round(vv_observations[i]*10,0)
    vs, vc = np.unique(np.array(v), return_counts=True)
    lambdas=[vc[i]/len(v) for i in vs]
    shmm = poissonHMM(name="{}_HMM".format(item), states=states, lambdas=lambdas,num_steps=n, f=f)

    shmm.setInit(train_states_seq=v)
    shmm.setTransition(train_states_seq=v)
    shmm.setEmission(train_states_seq=v, train_observations=vv_observations)

    shmm.createModel()
    shmm.fitModel(observations=vv_observations)
    posterior_mode = shmm.viterbiPath(observations=vv_observations)
    vv=posterior_mode.numpy()
    if vmin!=0:
        for i in range(n):
            vv[i]+=vmin
    if dn!=0:
        for k in range(dn):
            vv=np.delete(vv,-1)
    return vv.tolist()


def fillMissedStates(v_src:list=None, v_observations:list=None, f:object=None)->(int, int, int, int, np.array,
                                                                                 np.array,np.array):
    v = np.array(v_src).astype("int")
    n, = v.shape
    vmin = v.min()
    vmax = v.max()
    if vmin!=0:
        for i in range(n):
            v[i] = v[i] - vmin

    states, count_states = np.unique(v, return_counts=True)
    l_states=states.tolist()

    if l_states[-1] == len(l_states)-1:
        return vmin,vmax,n,0,states,v,v_observations
    l_full_states=[i for i in range(l_states[-1]+1)]
    for item in l_full_states:
        if item not in l_states:
            v_src.append(item)
            v_observations=np.append(v_observations, 0.0)
    v = np.array(v_src).astype("int")
    n1, = v.shape
    vmin = v.min()
    vmax = v.max()
    if vmin != 0:
        for i in range(n1):
            v[i] = v[i] - vmin

    states, count_states = np.unique(v, return_counts=True)
    return vmin, vmax, n1,n1-n, states,v, v_observations



if __name__== "__main__":
    pass

    dir_path = os.path.dirname(os.path.realpath(__file__))
    repository = Path(dir_path) / Path("Repository")
    file_csv = "updatedMwtMin.csv"
    src_csv = str(Path(repository) / Path(file_csv))

    main(src_csv=src_csv, repository=repository)