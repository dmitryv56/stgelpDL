#!/usr/bin/python3
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
import dateutil
import matplotlib.pyplot as plt

from tsstan.pltStan import setPlot,plotAll
from predictor.utility import cSFMT,logDictArima,svld2log,msg2log

def isMidnight(ISO8601str: str)->bool:
    bRet = False
    t = dateutil.parser.parse(ISO8601str)
    if t.hour == 0 and t.minute == 0:
        bRet=True
    return bRet




def  crtDictProfile(df: pd.DataFrame, dt_col_name: str, data_col_name:str, typeProfile:str ='day', f: object =None)->dict:

    d_data={}
    data_list=[x for x in df[data_col_name].tolist() if np.isnan(x) == False]
    dt_list = [x for x in df[dt_col_name].tolist() ]
    k=None
    k_prev=None
    for i in range( len(data_list)):
        if isMidnight(dt_list[i]):
            k=i
            if k_prev is not None:
                d_data[dt_list[k_prev]]=list_day_value
            list_day_value=[]
            k_prev=k
        else:
            if k is None:
                continue
        list_day_value.append(data_list[i])


    pass
    return d_data

def dictLists2matrix(d_dict_lists: dict,f: object =None)->(np.array,list):
    n=len(d_dict_lists)
    labelX=[]
    # get first key in dictionary
    key_first=list(d_dict_lists.keys())[0]
    list_first=d_dict_lists[key_first]
    m=len(list_first)
    X=np.zeros((n,m), dtype=float)
    i=0
    meanX=np.zeros((m),dtype=float)
    for key,value in d_dict_lists.items():
        j=0
        labelX.append(key)
        for item in value:
            X[i][j]=item
            meanX[j]+=item
            j+=1
        i+=1
    #
    msg=" X matrix {} x {}\n\n".format(n,m)
    msg2log(dictLists2matrix.__name__,msg,f)
    svld2log(X, labelX, 4, f)
    meanX=meanX/n
    msg = " Mean values vector {} \n\n".format( m)
    msg2log(dictLists2matrix.__name__, msg, f)
    svld2log(meanX, [n ], 4, f)
    for i in range(n):
        for j in range(m):
            X[i][j]=X[i][j]-meanX[j]
    msg = " \n\n Centerd X matrix {} x {}\n\n".format(n, m)
    msg2log(dictLists2matrix.__name__, msg, f)
    svld2log(X, labelX, 4, f)

    return X,labelX

    pass
def pca(X: np.array,labelX: list, f: object = None)->(np.array, np.array):
    # Data matrix
    n,m =X.shape
    C=np.dot(X.T,X)/(n-1)
    # eigen decomposition
    eig_vals,eig_vecs= np.linalg.eig(C)
    msg = " \n\n Eigen values\n\n{}".format(eig_vals)
    msg2log(pca.__name__, msg, f)
    #project X ontoPC space
    X_pca=np.dot(X,eig_vecs)

    msg = " X maped onto PCA  matrix {} x {}\n\n".format(n, m)
    msg2log(pca.__name__, msg, f)
    svld2log(X, labelX, 4, f)

    return X_pca,eig_vals




if __name__ =="__main__":
    f = None
    setPlot()
    file_png_path="DaysPlot"
    csv_source = "~/LaLaguna/stgelpDL/dataLaLaguna/PrivateHouseElectricityConsumption_21012020_seas.csv"
    df = pd.read_csv(csv_source)
    name = "PrivateHouse"
    dt_col_name="Date Time"
    data_column_name = "Real_demandDiff"
    d_day_data= crtDictProfile(df, dt_col_name, data_column_name, typeProfile = 'day', f = None)
    pass
    with open("DayProfile.log","w") as fl:
        logDictArima(d_day_data, 4, fl)
        X, labelX = dictLists2matrix(d_day_data, fl)

        X_pca,eig_vals = pca(X,labelX,fl)

        plt.scatter(X_pca[:,0],X_pca[:,1])
        plt.savefig("TwoComponent.png")
        plt.close("all")
    if False:
        for data_date, data_list in d_day_data.items():
            plotAll(name, data_list, "{}_{}".format(data_column_name,data_date), file_png_path, f=None)
            pass