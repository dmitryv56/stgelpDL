import pandas as pd
import copy
from pathlib import Path

import pandas as pd
import numpy as np
import math
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

from predictor.utility import msg2log

def Forcast_imbalance_edit():
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv")
    src_col_name = "Forcasting"
    src1_col_name = "Real_demand"
    dst_col_name = "FrcImbalance"
    ds[dst_col_name] =[round(ds[src_col_name][i]-ds[src1_col_name][i],2)  for i in range(len(ds)) ]


    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020_CommonAnalyze.csv", index=False)
    return

def WindTurbine_edit():
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_20102020_WindGenPower.csv")
    aux_col_name = "Programmed_demand"
    dest_col_name="Real_demand"
    src_col_name="WindGen_Power_"
    ds[aux_col_name]=[ 0.0  for i in range(len(ds[aux_col_name]))]

    ds[dest_col_name]=ds[src_col_name]
    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/editedElHiero_24092020_20102020_WindGenPower.csv", index=False)
    return

def profivateHouse_edit():
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/__PrivateHouseElectricityConsumption_21012020.csv")

    col_name='lasts'
    dt_col_name='Date Time'
    aux_col_name="Programmed_demand"
    data_col_name="Demand"
    v=ds[col_name].values
    for i in range (len(ds[col_name].values)):
        a=v[i].split('-')
        v[i]='T'+a[0]+":00.000+02:00"

    ds[col_name]=copy.copy(v)
    for i in range(len(ds[col_name])):
        ds[dt_col_name][i] =ds[dt_col_name][i] + v[i]
    ds1 =ds.drop([col_name], axis=1)
    add_col=[]
    for i in range(len(ds[dt_col_name])):
        add_col.append(ds[data_col_name].values[i] * 2 )
    ds1[aux_col_name]=add_col
    ds1.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/PrivateHouseElectricityConsumption_21012020.csv", index=False)

    pass


def powerSolarPlant_edit():
    # ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/__PowerGenOfSolarPlant_21012020.csv")
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/__SolarPlantPowerGen_21012020.csv")
    # col_name = 'lasts'
    dt_col_name = 'Date Time'
    aux_col_name = "Programmed_demand"
    data_col_name = "PowerGen"
    v = ds[dt_col_name].values
    for i in range(len(ds[dt_col_name].values)):
        a = v[i].split(' ')
        b=a[0].split('.')
        if len(a[1])<5:
            a[1]='0'+a[1]
        v[i]='2020-'+b[1]+"-"+b[0]+'T'+a[1]+':00.000+02:00'


    ds[dt_col_name] = copy.copy(v)

    # ds1 = ds.drop([col_name], axis=1)
    add_col = []
    for i in range(len(ds[dt_col_name])):
        add_col.append(ds[data_col_name].values[i] * 2)
    ds[aux_col_name] = add_col
    # ds1.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/PowerGenOfSolarPlant_21012020.csv", index=False)
    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020_21012020.csv", index=False)
    pass

def powerSolarPlant_Imbalance():
    # ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/__PowerGenOfSolarPlant_21012020.csv")
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020.csv")
    # col_name = 'lasts'
    dt_col_name = 'Date Time'
    aux_col_name = "Programmed_demand"
    data_col_name = "PowerGen"
    ds["Imbalance"] = [ds[aux_col_name].values[i]-ds[data_col_name].values[i] for i in range(len(ds)) ]


    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020.csv", index=False)
    pass


def powerElHiero_edit():

    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/_ElHiero_24092020_20102020_additionalData.csv")

    dt_col_name = 'Date Time'
    aux_col_name = "Programmed_demand"
    data_col_name = "PowerGen"
    v = ds[dt_col_name].values
    for i in range(len(ds[dt_col_name].values)):
        a = v[i].split(' ')
        b = a[0].split('.')
        if len(a[1]) < 5:
            a[1] = '0' + a[1]
        v[i] = '2020-' + b[1] + "-" + b[0] + 'T' + a[1] + ':00.000+02:00'

    ds[dt_col_name] = copy.copy(v)


    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_20102020_additionalData.csv", index=False)
    pass

def datosElHiero_edit():
    # ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/__PowerGenOfSolarPlant_21012020.csv")
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/Datos_de_El_Hierro_2016.csv")
    # col_name = 'lasts'
    dt_col_name = 'Date Time'

    v = ds[dt_col_name].values
    for i in range(len(ds[dt_col_name].values)):
        a = v[i].split(' ') #29-12-2016 3:00:00
        b=a[0].split('-')   # dd mm year
        a0new="{}-{}-{}".format(b[2],b[1],b[0])
        c=a[1].split(':')  #h mm ss
        if len(c[0])<2:
            c[0]='0{}'.format(c[0])
        a1new='{}:{}:{}.000+00:00'.format(c[0],c[1],c[2])

        v[i]="{}T{}".format(a0new,a1new)

    ds[dt_col_name] = copy.copy(v)


    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.", index=False)
    pass
def datosElHiero_PerDay():
    src_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv"
    dst_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_Days.csv"
    ds = pd.read_csv(src_csv)
    # col_name = 'lasts'
    dt_col_name = 'Date Time'
    data_col_name = 'Real_demand'
    v=ds[data_col_name].values
    n_v=len(v)
    col_names=["{}:{}".format(int(i/6), (i%6)*10) for i in range(144)]
    v_dict = {col_names[i]:[] for i in range(144)}
    # d_dict = {"Date Time": [ds[dt_col_name][i] for i in range(0, n_v, 144)]}
    # d_dict={"Date": [pd.to_datetime(ds[dt_col_name][i],dayfirst=True).date() for i in range(0,n_v,144)]}
    d_dict = {"Date": [pd.to_datetime(ds[dt_col_name][i], dayfirst=True).date().strftime('%Y-%m-%d') for i in range(0, n_v, 144)]}

    for i in range(n_v):
        v_dict[col_names[i%144]].append(v[i])
    # last list 365-size, we add v[0]

    d={**d_dict,**v_dict}
    ds1=pd.DataFrame(d)
    ds1.to_csv(dst_csv)
    return

def datosElHiero_PerTimeStamps():
    src_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv"
    dst_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_TimeStamps.csv"
    ds = pd.read_csv(src_csv)
    # col_name = 'lasts'
    dt_col_name = 'Date Time'
    data_col_name = 'Real_demand'
    v=ds[data_col_name].values
    n_v=len(v)
    col_names= [pd.to_datetime(ds[dt_col_name][i], dayfirst=True).date().strftime('%Y-%m-%d') for i in
                       range(0, n_v, 144)]
    row_names=["{}:{}".format(int(i/6), (i%6)*10) for i in range(144)]
    v_dict={}
    # v_dict = {item:[] for item in col_names}
    # d_dict = {"Date Time": [ds[dt_col_name][i] for i in range(0, n_v, 144)]}
    # d_dict={"Date": [pd.to_datetime(ds[dt_col_name][i],dayfirst=True).date() for i in range(0,n_v,144)]}
    d_dict = {"Date": [pd.to_datetime(ds[dt_col_name][i], dayfirst=True).date().strftime('%Y-%m-%d') for i in range(0, n_v, 144)]}
    d_dict = {"Timestamp": row_names}
    n_start=0
    n_series=144  # obsefvation points
    n_features=366 # days
    for icol in col_names:
        v_dict[icol]=v[n_start:n_start+n_series].tolist()
        n_start=n_start+n_series
    # last list 365-size, we add v[0]

    d={**d_dict,**v_dict}
    ds1=pd.DataFrame(d)
    ds1.to_csv(dst_csv)
    return

def datosElHiero_PerTimeStamps():
    dst_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016-Est.csv"
    src_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_TimeStamps.csv"
    ds = pd.read_csv(src_csv)
    # col_name = 'lasts'
    dt_col_name = 'Timestamp'


    d_dst={}


    vts=ds['Timestamp'].values.tolist()
    vts.append('ave')
    vts.append('std')
    vts.append('min')
    vts.append('max')
    d_dst["RowName"]=vts
    for col in ds.columns:
        if 'Unnamed' in col or 'Timestamp' in col: continue
        v=ds[col].values.tolist()
        a=np.array(v)
        ave=a.mean()
        std=a.std()
        minv=a.min()
        maxv=a.max()
        v.append(round(ave,3))
        v.append(round(std,3))
        v.append(round(minv,2))
        v.append(round(maxv,2))
        d_dst[col]=v

    ds1 = pd.DataFrame(d_dst)
    n_ds1=len(ds1)

    ave = []
    std = []
    minv = []
    maxv = []
    for i in range(n_ds1):

        v=[]
        for col in ds1.columns[1:]:
            v.append(ds1[col][i])
        a=np.array(v)
        ave.append(round(a.mean(),3))
        std.append(round(a.std(),3))
        minv.append(round(a.min(),2))
        maxv.append(round(a.max(),2))
    ds1['ave']=ave
    ds1['std'] = std
    ds1['min'] = minv
    ds1['max'] = maxv


    ds1.to_csv(dst_csv)
    return

""" transform time series (feature in source dataset) matrix Nrows * Mcols.

The source dataset must comprise  a timestamp feature(column) named 'dt_col_name' or index column. The 'data_col_name'-
feature must be a time series (TS) are ordered by  index ot timestamp feature('dt_col_name') , that is, the observation 
must be equidistant and without gaps. Additional, the beginning timestamp must be 00:00:00 (or 0 for index).
The 'period' and 'direction' determine the formation of the matrix. 
If the 'direction' is along the 'X'- axis, then the segments of the TS corresponding to the 'period' are the 
rows of the matrix. Column names are derived from  observation times within a period, for example, "00: 00,00: 10, ..., 
23:50" to 10 minutes discretization and a period of 1 day. 
If the direction is along 'Y'-axis, then the segment of  the TS corresponding to the 'period' are the column of matrix.
The column names are derived from the period in the timestamp, i.g. data string if the period is 1 day like as 
'2016-03-27', and row names are derived from observations within period, for example, "00:00,...,23:50".
 

"""
def ts2matrix(source_csv:str=None, dest_csv:str=None, dt_col_name:str="Date Time",
                      data_col_name:str='Real_demand', outRowIndex="rowNames", discret:int=10,period:object=None,
                      direction:str='x', title:str="", f:object=None):

    if source_csv is None or not Path(source_csv).exists() :
        msg="The source dataset path is invalid:  {}".format(source_csv)
        msg2log(None,msg,f)
        return None

    ds=pd.read_csv(source_csv)
    if dt_col_name not in ds.columns or data_col_name not in ds.columns:
        msg = "{} or {} not found in the dataset {}".format(dt_col_name,data_col_name, source_csv)
        msg2log(None, msg, f)
        return None
    folder_csv=Path(source_csv).parent
    dest_csv = Path(folder_csv / "{}_{}".format(title,data_col_name)).with_suffix(".csv")
    dest_se_csv = Path(folder_csv / "{}_{}_StatEst".format(title, data_col_name)).with_suffix(".csv")

    if direction=="X" or direction == "x":
        dsMatrix = ts2matrix_X(ds=ds, dt_col_name=dt_col_name, data_col_name=data_col_name,outRowIndex=outRowIndex,
                               discret=discret, period=period, f=f)
    elif direction=="Y" or direction == "y":
        dsMatrix = ts2matrix_Y(ds=ds, dt_col_name=dt_col_name, data_col_name=data_col_name, outRowIndex=outRowIndex,
                               discret=discret, period=period, f=f)
    else:
        msg = "{} invalid direction ".format(direction)
        msg2log(None, msg, f)
        return None
    dsMatrix.to_csv(dest_csv)

    dsMatrixStatEst=ts2matrix_statest(ds=dsMatrix, rowIndexName=outRowIndex, f=f)
    dsMatrixStatEst.to_csv(dest_se_csv)
    return


def ts2matrix_X(ds:pd.DataFrame=None, dt_col_name:str='Date Time',data_col_name:str='Real_demand',
                outRowIndex:str="rowNames", discret:int=10,period:int=144, f:object=None)->pd.DataFrame:
    # src_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv"
    # dst_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_Days.csv"
    # ds = pd.read_csv(src_csv)
    # # col_name = 'lasts'
    # dt_col_name = 'Date Time'
    # data_col_name = 'Real_demand'
    h_period_sr=60/discret  # hour in sample resolution, 6
    d_period_sr = period    # day period in sample resolution, 144
    v=ds[data_col_name].values
    n_v=len(v)
    if n_v%d_period_sr!=0:
        msg="Time series size is {} and it is not multiply of {} period".format(n_v,d_period_sr)
        msg2log(None,msg,f)
        return None

    col_names = ["{}:{}".format(int(i / h_period_sr), int((i % h_period_sr) * 10)) for i in range(d_period_sr)]
    v_dict = {col_names[i]:[] for i in range(d_period_sr)}

    d_dict = {outRowIndex: [pd.to_datetime(ds[dt_col_name][i], dayfirst=True).date().strftime('%Y-%m-%d') \
                       for i in range(0, n_v, d_period_sr)]}

    for i in range(n_v):
        v_dict[col_names[i%144]].append(v[i])

    d={**d_dict,**v_dict}
    ds1=pd.DataFrame(d)
    # ds1.to_csv(dst_csv)
    return ds1

def ts2matrix_Y(ds:pd.DataFrame=None, dt_col_name:str='Date Time',data_col_name:str='Real_demand',
                outRowIndex:str="rowNames", discret:int=10,period:int=144, f:object=None)->pd.DataFrame:
    # src_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv"
    # dst_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_TimeStamps.csv"
    # ds = pd.read_csv(src_csv)
    # # col_name = 'lasts'
    # dt_col_name = 'Date Time'
    # data_col_name = 'Real_demand'
    #
    #

    v=ds[data_col_name].values
    n_v=len(v)

    h_period_sr = 60 / discret  # hour in sample resolution, 6
    d_period_sr = period  # day period in sample resolution, 144
    v = ds[data_col_name].values
    n_v = len(v)
    if n_v % d_period_sr != 0:
        msg = "Time series size is {} and it is not multiply of {} period".format(n_v, d_period_sr)
        msg2log(None, msg, f)
        return None

    col_names= [pd.to_datetime(ds[dt_col_name][i], dayfirst=True).date().strftime('%Y-%m-%d') for i in
                       range(0, n_v, d_period_sr)]
    row_names=["{}:{}".format(int(i/h_period_sr), int((i%h_period_sr)*10)) for i in range(d_period_sr)]
    v_dict={}

    d_dict = {outRowIndex: row_names}
    n_start=0
    n_series=d_period_sr  # obsefvation points
    n_features=366 # days
    for icol in col_names:
        v_dict[icol]=v[n_start:n_start+n_series].tolist()
        n_start=n_start+n_series


    d={**d_dict,**v_dict}
    ds1=pd.DataFrame(d)
    # ds1.to_csv(dst_csv)
    return ds1

def ts2matrix_statest(ds:pd.DataFrame=None,rowIndexName:str="rowNames",f:object=None):
    # dst_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016-Est.csv"
    # src_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_TimeStamps.csv"
    # ds = pd.read_csv(src_csv)
    # col_name = 'lasts'
    dt_col_name = rowIndexName


    d_dst={}


    vts=ds[rowIndexName].values.tolist()
    vts.append('ave')
    vts.append('std')
    vts.append('min')
    vts.append('max')
    d_dst[rowIndexName]=vts
    for col in ds.columns:
        if 'Unnamed' in col or rowIndexName in col: continue
        v=ds[col].values.tolist()
        a=np.array(v)
        ave=a.mean()
        std=a.std()
        minv=a.min()
        maxv=a.max()
        v.append(round(ave,3))
        v.append(round(std,3))
        v.append(round(minv,2))
        v.append(round(maxv,2))
        d_dst[col]=v

    ds1 = pd.DataFrame(d_dst)
    n_ds1=len(ds1)

    ave = []
    std = []
    minv = []
    maxv = []
    for i in range(n_ds1):

        v=[]
        for col in ds1.columns[1:]:
            v.append(ds1[col][i])
        a=np.array(v)
        ave.append(round(a.mean(),3))
        std.append(round(a.std(),3))
        minv.append(round(a.min(),2))
        maxv.append(round(a.max(),2))
    ds1['ave']=ave
    ds1['std'] = std
    ds1['min'] = minv
    ds1['max'] = maxv


    # ds1.to_csv(dst_csv)
    return ds1







if __name__=="__main__":
    # privateHouse_edit()
    # powerSolarPlant_edit()
    # powerElHiero_edit()
    #WindTurbine_edit()
    # Forcast_imbalance_edit()
    # powerSolarPlant_Imbalance()
    # powerSolarPlant_analysis()
    # datosElHiero_edit()
    # datosElHiero_PerDay()
    # datosElHiero_PerTimeStamps()
    # datosElHiero_PerTimeStamps()
    src_csv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv"

    with open("loglog.log",'w+') as ff:
        ts2matrix(source_csv=src_csv,  dt_col_name = "Date Time", data_col_name= 'Real_demand',
                  outRowIndex = "rowNames",  discret = 10, period=144, direction = 'x', title = "X_direction", f=ff)
        ts2matrix(source_csv=src_csv, dt_col_name="Date Time", data_col_name='Real_demand',
                  outRowIndex="rowNames", discret=10, period=144, direction='y', title="Y_direction", f=ff)
    pass
