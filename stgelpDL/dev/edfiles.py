import pandas as pd
import copy

import pandas as pd
import numpy as np
import math
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

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

def privateHouse_edit():
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
feature must be a time series are ordered by  index ot timestamp feature('dt_col_name') , that is, the observation must 
be equidistant and without gaps. Additional, the beginning timestamp must be 00:00:00 (or 0 for index).
The 'period' and 'direction' determine the formation of the matrix. If the 'direction' is along the X- axis, then the 
segments of the time series corresponding to the 'period' are the rows of the matrix. Column names are derived from 
observation times within a period, for example, "00: 00,00: 10, ..., 23:50" to 10 minutes discretization and a period 
of 1 day. 
 

"""
def timeseries2matrix(source_csv:str=None, dest_csv:str=None, dt_col_name:str="Date Time",
                      data_col_name:str='Real_demand', discret:int=10,period:object=None, direction:str='x',
                      row_name:str="rowNames",title:str="",f:object=None):

    pass






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
    datosElHiero_PerTimeStamps()
    pass
