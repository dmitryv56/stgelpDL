import pandas as pd
import copy


def ElHieroPlant_edit():
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/_ElHieroPowerPlantSummary2018_2020_DayIncrements.csv")


    dt_col_name='Date Time'
    aux_col_name="Programmed_demand"
    data_col_name="Demand"
    v=ds[dt_col_name].values
    for i in range (len(ds[dt_col_name].values)):
        a=v[i].split('T')
        b=a[0].split('-')
        Year=b[2]
        Month=b[1]
        if len(Month)<2: Month='0'+Month
        Day=b[0]
        if len(Day)<2: Day='0'+Day
        v[i]=Year +'-'+Month+'-'+Day+'T'+"00:00:00.000+02:00"

    ds[dt_col_name]=copy.copy(v)

    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/ElHieroPowerPlantSummary2018_2020_DayIncrements.csv", index=False)

    pass

def Imbalance_edit():
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_20102020.csv")
    second_col_name = "Programmed_demand"
    first_col_name="Real_demand"
    dest_col_name="Imbalance"
    ds[dest_col_name]=ds[first_col_name] -ds[second_col_name]

    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/Imbalance_ElHiero_24092020_20102020.csv", index=False)
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

def powerElHiero_edit():

    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/_ElHiero_24092020_20102020_additionalData.csv")

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


    ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_20102020_additionalData.csv", index=False)
    pass

if __name__=="__main__":
    # privateHouse_edit()
    #powerSolarPlant_edit()
    # powerElHiero_edit()
    #WindTurbine_edit()
    ElHieroPlant_edit()
    pass