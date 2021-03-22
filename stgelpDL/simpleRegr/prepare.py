#!/usr/bin/python3

import os
import sys
from datetime import datetime
import math

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from predictor.utility import msg2log
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
LOW=0
MID=1
HIGH=2

# csv_file="/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
def prepare(csv_file:str="",dt_col_name:str="Date Time", co2_col_name:str="CO2",dsl_col_name:str="Diesel_Power",
            f:object=None)->(dict,dict):
    pass


    f_name=Path(csv_file).stem
    f_parent =Path(csv_file).parent

    df=pd.read_csv(csv_file)
    n=len(df)
    co2=np.array(df[co2_col_name])
    dsl=np.array(df[dsl_col_name])
    datetime=np.array(df[dt_col_name])

    densityCO2 = np.zeros((n), dtype=float)

    point2interval = np.zeros((n), dtype=int)

    for i in range(n):
        if abs(dsl[i])<0.000001:
            point2interval[i]=-1
            continue
        densityCO2[i]=co2[i]/dsl[i]
    densityCO2= np.round(densityCO2,4)

    ar_un,ar_cnt=np.unique(point2interval,return_counts=True)
    ddensityCO2  = np.zeros((n),dtype=float)
    dddensityCO2 = np.zeros((n),dtype=float)
    ddsl         = np.zeros((n),dtype=float)
    dddsl        = np.zeros((n),dtype=float)

    msg2log(None,"Sample size :{}  Diesel Power zero and non zero values{} ".format(n,ar_cnt),f)
    ddensityCO2[1]=densityCO2[1]-densityCO2[0]
    ddsl[1]=dsl[1]-dsl[0]
    for i in range(2,n):
        ddensityCO2[i]=densityCO2[i]-densityCO2[i-1]
        dddensityCO2[i] = densityCO2[i] - 2.0* densityCO2[i - 1] + densityCO2[i-2]
        ddsl[i] = dsl[i] - dsl[i-1]
        dddsl[i] = dsl[i] - 2*dsl[i - 1]+dsl[i-2]
    ddensityCO2=np.round(ddensityCO2,4)
    dddensityCO2=np.round(dddensityCO2,4)
    ddsl=np.round(ddsl,4)
    dddsl=np.round(dddsl,4)
    df['densCO2']  = densityCO2.tolist()
    df['ddensCO2']  = ddensityCO2.tolist()
    df['dddensCO2'] = dddensityCO2.tolist()
    df['flagCO2']=point2interval.tolist()
    df['ddsl'] = ddsl.tolist()
    df['dddsl'] = dddsl.tolist()


    df.to_csv( Path( Path(f_parent )/Path("{}_CO2diffs_Dsldiffs".format(f_name)) ).with_suffix(".csv"))
    title="original and differences"
    labels=[co2_col_name,'diff({})'.format(co2_col_name),'diff(diff({}))'.format(co2_col_name)]
    # plotTS(densityCO2, y_diff=ddensityCO2, y_diffdiff=dddensityCO2, endogen_col_name=co2_col_name, title=title,
    #        labels=labels, wwidth=256,   folder=D_LOGS["plot"], f=f)
    plotTS(dsl, y_diff=ddsl, y_diffdiff=dddsl, endogen_col_name=dsl_col_name, title=title,
           labels=labels, wwidth=256, folder=D_LOGS["plot"], f=f)
    max_co2=0.0
    min_co2=10000.0
    for i in range(n):
        if point2interval[i]==-1:
            continue
        if densityCO2[i]>max_co2:
           max_co2=densityCO2[i]
        if densityCO2[i]<min_co2:
           min_co2=densityCO2[i]


    delta_co2=round((max_co2-min_co2)/3.0,4)

    d_CO2={LOW:{},MID:{},HIGH:{}}

    low_CO2=[]
    diesel_low_CO2=[]
    ddiesel_low_CO2 = []
    dddiesel_low_CO2 = []
    dt_low_CO2=[]
    mid_CO2=[]
    diesel_mid_CO2=[]
    ddiesel_mid_CO2 = []
    dddiesel_mid_CO2 = []
    dt_mid_CO2=[]
    high_CO2=[]
    diesel_high_CO2=[]
    ddiesel_high_CO2 = []
    dddiesel_high_CO2 = []
    dt_high_CO2=[]
    msg2log(None, "Min density CO2:{} Max density CO2:{} delta {}".format(min_co2,max_co2,delta_co2),f)
    msg2log(None, "Range of density CO2: {} - {}".format(min_co2,max_co2),f)



    for i in range(n):
        if densityCO2[i]>=min_co2 and densityCO2[i]<=min_co2+delta_co2 and point2interval[i]==0:
            d_CO2[LOW][datetime[i]]=co2[i]   #densityCO2[i]
            low_CO2.append(co2[i])  # densityCO2[i])
            diesel_low_CO2.append(dsl[i])
            ddiesel_low_CO2.append(ddsl[i])
            dddiesel_low_CO2.append(dddsl[i])
            dt_low_CO2.append(datetime[i])
        elif densityCO2[i]>min_co2+delta_co2 and densityCO2[i]<=min_co2+2*delta_co2 and point2interval[i]==0:
            d_CO2[MID][datetime[i]] = co2[i]   #densityCO2[i]
            mid_CO2.append(co2[i])                   #densityCO2[i])
            diesel_mid_CO2.append(dsl[i])
            ddiesel_mid_CO2.append(ddsl[i])
            dddiesel_mid_CO2.append(dddsl[i])
            dt_mid_CO2.append(datetime[i])
        elif densityCO2[i] > min_co2 + 2* delta_co2 and densityCO2[i] <= max_co2 and point2interval[i]==0:
            d_CO2[HIGH][datetime[i]] = co2[i] #densityCO2[i]
            high_CO2.append(co2[i])                 #densityCO2[i])
            diesel_high_CO2.append(dsl[i])
            ddiesel_high_CO2.append(ddsl[i])
            dddiesel_high_CO2.append(dddsl[i])
            dt_high_CO2.append(datetime[i])
        else:
            pass

    message = f"""Density CO2 Intervals
Low:  {min_co2} <= X < {min_co2 + delta_co2}                 Samples: {len(low_CO2)}
Mid:  {min_co2 + delta_co2} < X <= {min_co2 + 2 * delta_co2} Samples: {len(mid_CO2)}
High: {min_co2 + 2 * delta_co2} < X <= {max_co2}             Samples: {len(high_CO2)}
The density CO2 is used for build the interval only. The original CO2 data should be in regeression models.
"""

    msg2log(None,message,f)

    low_csv_file  = Path( Path(f_parent )/Path("low_co2") ).with_suffix(".csv")
    mid_csv_file  = Path( Path(f_parent )/Path("mid_co2") ).with_suffix(".csv")
    high_csv_file = Path( Path(f_parent )/Path("high_co2") ).with_suffix(".csv")

    dfLowCO2=pd.DataFrame()
    dfLowCO2[dt_col_name]  = dt_low_CO2
    dfLowCO2[co2_col_name] = low_CO2
    dfLowCO2[dsl_col_name] = diesel_low_CO2
    dfLowCO2["d_{}".format(dsl_col_name)]  = ddiesel_low_CO2
    dfLowCO2["dd_{}".format(dsl_col_name)] = dddiesel_low_CO2

    dfLowCO2.to_csv(low_csv_file)

    dfMidCO2 = pd.DataFrame()
    dfMidCO2[dt_col_name]  = dt_mid_CO2
    dfMidCO2[co2_col_name] = mid_CO2
    dfMidCO2[dsl_col_name] = diesel_mid_CO2
    dfMidCO2["d_{}".format(dsl_col_name)]  = ddiesel_mid_CO2
    dfMidCO2["dd_{}".format(dsl_col_name)] = dddiesel_mid_CO2

    dfMidCO2.to_csv(mid_csv_file)

    dfHighCO2 = pd.DataFrame()
    dfHighCO2[dt_col_name]  = dt_high_CO2
    dfHighCO2[co2_col_name] = high_CO2
    dfHighCO2[dsl_col_name] = diesel_high_CO2
    dfHighCO2["d_{}".format(dsl_col_name)]  = ddiesel_high_CO2
    dfHighCO2["dd_{}".format(dsl_col_name)] = dddiesel_high_CO2

    dfHighCO2.to_csv(high_csv_file)

    title = "Low density CO2 interval"
    labels = ["CO2","Diesel Pwr",""]
    plotTS(np.array(low_CO2),  y_diff=np.array(diesel_low_CO2), endogen_col_name="lowCO2", title=title, labels=labels,
           wwidth=256,folder=D_LOGS["plot"], f=f)
    title = "Mid density CO2 interval"
    labels = ["CO2","Diesel Pwr",""]
    plotTS(np.array(mid_CO2), y_diff=np.array(diesel_mid_CO2), endogen_col_name="midCO2", title=title, labels=labels,
           wwidth=256, folder=D_LOGS["plot"], f=f)
    title = "High density CO2 interval"
    labels = ["CO2","Diesel Pwr",""]
    plotTS(np.array(high_CO2), y_diff=np.array(diesel_high_CO2), endogen_col_name="highCO2", title=title, labels=labels,
           wwidth=256, folder=D_LOGS["plot"], f=f)
    # check datasrts

    lowDf=pd.read_csv(low_csv_file)
    midDf = pd.read_csv(mid_csv_file)
    highDf = pd.read_csv(high_csv_file)

    d_df={LOW:lowDf,MID:midDf,HIGH:highDf}

    message=f"""
Low CO2 interval dataset:  {low_csv_file}
Mid CO2 interval dataset:  {mid_csv_file}
High CO2 interval dataset: {high_csv_file}

Each dataset contains a following columns: {lowDf.columns}
Note: difference values of the 'Diesel Power' are sparse samples for entire 'Diesel Power' time series ! 
"""

    msg2log(None,message,f)
    return d_CO2, d_df


def plotTS( y: np.array, y_diff: np.array=None, y_diffdiff: np.array=None, endogen_col_name: str="CO2",  title:str="",
            labels:list=["","",""], wwidth: int = 512,  folder: str = None, f: object = None):
    # Plot outputs
    if folder is not None:
        pathFolder = Path(folder)
    else:
        pathFolder = ""

    (n,) = y.shape


    for n_first in range(0, n, wwidth):
        n_last = n_first + wwidth if n_first + wwidth <= n else n
        x = np.arange(n_first, n_last)
        plt.plot(x, y[n_first:n_last], color='blue', label=labels[0])
        if y_diff is not None:
            plt.plot(x, y_diff[n_first:n_last], color='orange', label=labels[1])
        if y_diffdiff is not None:
            plt.plot(x, y_diffdiff[n_first:n_last], color='green', label=labels[2])
        plt.xlabel("Sample number")
        plt.ylabel(endogen_col_name)
        plt.title("{}-{} (from {} to {})".format(endogen_col_name, title, n_first, n_last))
        plt.axis('tight')
        filepng = Path(Path(pathFolder) / Path("{}_{}_{}.png".format(endogen_col_name, n_first, n_last))).with_suffix(
            ".png")
        plt.legend()
        plt.savefig(filepng)

        plt.close("all")

    return
def main():

    csv_file = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
    dt_col_name: str = "Date Time"
    co2_col_name:str = "CO2"
    dsl_col_name:str = "Diesel_Power"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(co2_col_name, date_time)

    listLogSet(str(folder_for_logging))  # A logs are creating
    prepare(csv_file=csv_file, dt_col_name=dt_col_name, co2_col_name= co2_col_name, dsl_col_name=dsl_col_name,
            f=D_LOGS["control"])

    closeLogs()

if __name__ == "__main__":
    main()
