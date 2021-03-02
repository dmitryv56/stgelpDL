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
    dest_col_name = "Real_demand"
    src_col_name = "WindGen_Power_"
    ds[aux_col_name] = [0.0 for i in range(len(ds[aux_col_name]))]

    ds[dest_col_name] = ds[src_col_name]
    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/editedElHiero_24092020_20102020_WindGenPower.csv", index=False)
    return


def privateHouse_edit():
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/__PrivateHouseElectricityConsumption_21012020.csv")

    col_name = 'lasts'
    dt_col_name = 'Date Time'
    aux_col_name = "Programmed_demand"
    data_col_name = "Demand"
    v = ds[col_name].values
    for i in range(len(ds[col_name].values)):
        a = v[i].split('-')
        v[i] = 'T' + a[0] + ":00.000+02:00"

    ds[col_name] = copy.copy(v)
    for i in range(len(ds[col_name])):
        ds[dt_col_name][i] = ds[dt_col_name][i] + v[i]
    ds1 = ds.drop([col_name], axis=1)
    add_col = []
    for i in range(len(ds[dt_col_name])):
        add_col.append(ds[data_col_name].values[i] * 2)
    ds1[aux_col_name] = add_col
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
        b = a[0].split('.')
        if len(a[1]) < 5:
            a[1] = '0' + a[1]
        v[i] = '2020-' + b[1] + "-" + b[0] + 'T' + a[1] + ':00.000+02:00'

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

def powerSolarPlant_Normalize():
    # ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/__PowerGenOfSolarPlant_21012020.csv")
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020.csv")
    # col_name = 'lasts'
    dt_col_name = 'Date Time'
    aux_col_name = "Programmed_demand"
    data_col_name = "PowerGen"
    ds["normPowerGen"] = [round((ds[data_col_name].values[i]-round(95.0/2.4,4)),4) for i in range(len(ds)) ]


    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/normSolarPlantPowerGen_21012020.csv", index=False)
    pass

def powerSolarPlant_log():
    # ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/__PowerGenOfSolarPlant_21012020.csv")
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/difnormSolarPlantPowerGen_21012020.csv")
    # col_name = 'lasts'
    dt_col_name = 'Date Time'
    aux_col_name = "lnPowerGen"
    data_col_name = "PowerGen"
    v=ds[data_col_name].values
    v1=[]
    for i in range(len(v)):
        v1.append(round(math.log(v[i]+1.0),4))

    ds[aux_col_name] = v1


    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/difnormSolarPlantPowerGen_21012020.csv", index=False)
    pass


def powerSolarGen_Aggregation():
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020.csv")
    # col_name = 'lasts'
    dt_col_name = 'Date Time'
    aux_col_name = "Programmed_demand"
    data_col_name = "PowerGen"
    ds["normPowerGen"] = [round((ds[data_col_name].values[i]-round(95.0/2.4,4)),4) for i in range(len(ds)) ]
    v = ds[data_col_name].values
    dt = ds[dt_col_name].values
    start_ind = 0
    delta_ind = 144
    vmin=[]
    vmax=[]
    vmean=[]
    vdt=[]
    while (start_ind+delta_ind<= len(v)):
        # vmin.append(min(v[start_ind:start_ind + delta_ind]))
        vvmax=max(v[start_ind:start_ind + delta_ind])
        vmax.append(vvmax)
        vmean.append(round(np.mean(v[start_ind:start_ind + delta_ind]),3))
        vdt.append(dt[start_ind])
        vvmin=vvmax
        for i in range(start_ind,start_ind + delta_ind):
            if v[i]>0.0001 and v[i]<vvmin:
                vvmin=v[i]
        vmin.append(vvmin)

        start_ind = start_ind + delta_ind
    file_csv="~/LaLaguna/stgelpDL/dataLaLaguna/aggSolarPlantPowerGen_21012020.csv"
    ds1=pd.DataFrame()
    ds1['min_power']=vmin
    ds1['max_power']=vmax
    ds1['mean_power']=vmean
    ds1['Date Time']=vdt
    ds1.to_csv(file_csv, index=False)
    pass
    ds1.to_csv(file_csv, index=False)

    df1 = pd.read_csv(file_csv, parse_dates=['Date Time'], index_col='Date Time')
    # series = df.loc[:, 'value'].values
    # df.plot(figsize=(14,8),legend=None,title='a10-Drug Sales Series')
    plot1 = df1.plot(figsize=(14, 8), legend=True, title='Solar Power Gen')
    hist = df1.hist(bins=10,density=True)


    return

def powerSolarPlant_analysis():
    # ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/__PowerGenOfSolarPlant_21012020.csv")
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/difnormSolarPlantPowerGen_21012020.csv")
    # col_name = 'lasts'
    dt_col_name = 'Date Time'
    aux_col_name = "lnPowerGen"
    data_col_name = "PowerGen"
    v=ds[data_col_name].values
    v1=[]
    start_ind=0
    delta_ind=144
    while (start_ind<len(v)):

        v1.append(v[start_ind:start_ind+delta_ind])
        start_ind=start_ind+delta_ind
    # ds[aux_col_name] = v1

    a_min=np.zeros((144),dtype=float)
    for i in range(144):
        a_min[i]=100000.0
    a_max = np.zeros((144), dtype=float)
    a_mean = np.zeros((144), dtype=float)

    n=0
    for i in range(len(v1)):
        (m,)=v1[i].shape
        if m<144: continue
        n=n+1
        for j in range(m):
            if a_min[j]>v1[i][j]:   a_min[j] = v1[i][j]
            if a_max[j] < v1[i][j]: a_max[j] = v1[i][j]
            a_mean[j] = a_mean[j]+v1[i][j]
    (m,)=a_mean.shape
    for j in range(m) :
        a_mean[j]=round(a_mean[j]/n,4)

    pass
    df1=pd.DataFrame()
    df1['min_pwr'] =a_min.tolist()
    df1['max_pwr'] =a_max.tolist()
    df1['mean_pwr']=a_mean.tolist()
    today = datetime(year=2020, month=1, day=1, hour=0, minute=0)
    date_list = [today + timedelta(minutes=10 * x) for x in range(144)]
    datetext = [x.strftime('%H:%M') for x in date_list]
    df1['Date Time'] = datetext
    file_csv="~/LaLaguna/stgelpDL/dataLaLaguna/SolarGenHourMinute.csv"
    df1.to_csv(file_csv, index=False)

    df = pd.read_csv(file_csv, parse_dates=['Date Time'], index_col='Date Time')
    # series = df.loc[:, 'value'].values
    # df.plot(figsize=(14,8),legend=None,title='a10-Drug Sales Series')
    plot1=df1.plot(figsize=(14, 8), legend=True, title='Solar Power Gen')
    hist=df1.hist(bins=10,density=True)
    pass

    for i in range(47,119):
        title="{}_PowerGen".format(datetext[i])
        file_save="Logs/{}.png".format(title)
        a=np.zeros((n),dtype=float)
        for k in range(n):
            a[k]=v1[k][i]
        n_out, bins, _ = plt.hist(a,density=True)
        plt.title(title)
        plt.legend(title)
        plt.xlabel("MWatt")
        plt.savefig(file_save)
        plt.close("all")

    return


if __name__ == "__main__":
    # privateHouse_edit()
    # powerSolarPlant_edit()
    # powerElHiero_edit()
    #WindTurbine_edit()
    # Forcast_imbalance_edit()
    # powerSolarPlant_Imbalance()
    # powerSolarPlant_analysis()
    powerSolarGen_Aggregation()
    pass
