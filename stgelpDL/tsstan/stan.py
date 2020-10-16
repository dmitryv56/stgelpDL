#!/usr/bin/python3
""" This module is aimed for statistical analysis time series"""

import os
import sys
import copy

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

from predictor.Statmodel import tsARIMA
from predictor.control import ControlPlane
from predictor.utility import msg2log, PlotPrintManager, cSFMT, logDictArima


diff0        = lambda x,d: [ x[i] for i in range(len(x))]
diff1        = lambda x,d: [ x[i]-x[i+1] for i in range(len(x)-1)]
ddiff1       = lambda x,d: [ x[i]-2*x[i+1] +x[i+2] for i in range(len(x)-2)]
seas         = lambda x,d: [ x[i]-x[i+d] for i in range(len(x)-d)]
diffseas     = lambda x,d: [ x[i]-x[i+1] -x[i+d] +x[i+d+1] for i in range(len(x)-d-1)]
ddiffseas     = lambda x,d: [ x[i]-2*x[i+1] +x[i+2]-x[i+d] +2*x[i+d+1] -x[i+d+2] for i in range(len(x)-d-2)]
diff1seas1d  = lambda x,d: [ x[i]-x[i+1] -x[i+72] +x[i+72+1] for i in range(len(x)-d-1)]
ddiff1seas1d = lambda x,d: [ x[i]-2*x[i+1] +x[i+2]-x[i+72] +2*x[i+72+1]-x[i+72+2] for i in range(len(x)-d-1)]
seas288      = lambda x,d: [ x[i] - x[i+24] - x[i+2*24] + x[i+4*24] + x[i+5*24] + x[i+7*24] + x[i+8*24] - x[i+10*24] \
                                - x[i+11*24] + x[i +12*24]  for i in range(len(x)-d)]


l_seas_slplant=[
    {0:  {"PowerGen_":                diff0}},
    {1:  {"PowerGenDiff":             diff1}},
    {2:  {"PowerGenDiffDiff":         ddiff1}},
    {24: {"PowerGenSeas4h":           seas}},
    {48: {"PowerGenSeas8h":           seas}},
    {72: {"PowerGenSeas12h":          seas}},
    {144:{"PowerGenSeas1day":         seas}},
    {288:{"PowerGenSeas4h8h12h1d":    seas288}},
    {73: {"PowerGenDiffSeas1day":     diff1seas1d}},
    {74: {"PowerGenDiffDiffSeas1day": ddiff1seas1d}},

]

l_seas_prhouse=[

    {0: {"Real_demand_":              diff0}},
    {1: {"Real_demandDiff":           diff1}},
    {2: {"Real_demandDiffDiff":       ddiff1}},
    {4: {"Real_demandSeas4h":         seas}},
    {6: {"Real_demandSeas6h":         seas}},
    {8: {"Real_demandSeas8h":         seas}},
    {12:{"Real_demandSeas12h":        seas}},
    {24:{"Real_demandSeas1d":         seas}},
    {24:{"Real_demandDiffSeas1d":     diffseas}},
    {24:{"Real_demandDiffDiffSeas1d": ddiffseas}},
    {168:{"Real_demandSeas1w":        seas}},

]


def seasFilter(name: str,l_seas:dict, df: pd.DataFrame,main_col_name:str, discret:int, NFFT:int, f:object)->None:

    if len(l_seas) ==0:
        return

    d_item =l_seas.pop(0)
    [(seasonaly_period,dd_item)]=d_item.items()
    [(data_col_name, lmbd)] = dd_item.items()

    tsdiff = []
    arobj = tsARIMA("statanalysis", "tsARIMA", 32, 100, f)
    arobj.param = (0, 0, 0, 0, 0, 0, seasonaly_period, 10, discret, df[main_col_name].values)
    arobj.timeseries_name = data_col_name
    arobj.discret = discret
    arobj.nameModel=name
    tsdiff = lmbd(df[main_col_name].values,seasonaly_period)
    # if seasonaly_period==0:
    #     arobj.ts_data = df[data_col_name].values
    #
    # else:
    #
    #     for i in range(prev_len - seasonaly_period):
    #         tsdiff.append(df[prev_col_name].values[i] - df[prev_col_name].values[i + seasonaly_period])
    #     prev_len=prev_len-seasonaly_period
    arobj.ts_data=copy.copy(tsdiff)
    (Psd,freq,aCorr,aLags) = arobj.ts_analysis(NFFT)

    while len(tsdiff) < len(df):
        tsdiff.append(None)
    df[data_col_name] = copy.copy(tsdiff)
    del arobj
    return seasFilter(name, l_seas, df,main_col_name,  discret,NFFT, f)


if __name__=="__main__":

    pass

    l_seas=copy.copy(l_seas_slplant)
    name = "SolarPlant"
    csv_source="~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020.csv"
    csv_dest  = "~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020_seas.csv"
    data_col_name = "PowerGen"
    discret = 10  # min

    # l_seas = copy.copy(l_seas_prhouse)
    # name = "PrivateHouse"
    # csv_source = "~/LaLaguna/stgelpDL/dataLaLaguna/PrivateHouseElectricityConsumption_21012020.csv"
    # csv_dest = "~/LaLaguna/stgelpDL/dataLaLaguna/PrivateHouseElectricityConsumption_21012020_seas.csv"
    # data_col_name = "Real_demand"
    # discret = 60  # min

    main_log = "{}.txt".format(name)
    dt_col_name = 'Date Time'
    aux_col_name = "Programmed_demand"

    PlotPrintManager.set_Logfolders('Logs/control','Logs/predict')
    with open(main_log, 'w') as f:
        # df = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020.csv")
        df = pd.read_csv(csv_source)
        dt_col_name = 'Date Time'
        aux_col_name = "Programmed_demand"

        for i in range(len(df)):
            if np.isnan(df[data_col_name].values[i]):
                df[data_col_name][i]=0
        NFFT=2048

        seasFilter(name,l_seas, df, data_col_name,discret, NFFT, f)
        pass
    # df.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020_seas.csv")
    df.to_csv(csv_dest)
    pass




    f.close()
    pass