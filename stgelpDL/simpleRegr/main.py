#!/usr/bin/python3

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import math

import pandas as pd
from sklearn import linear_model as lm
from sklearn.covariance import EmpiricalCovariance
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from predictor.utility import msg2log

__version__ ="0.0.1"

def parsing():
    # command-line parser
    sDescriptor = 'Linear Regression (green) electricity load processes by using Deep Learning'
    sCSVhelp = "Absolute path to a source dataset (csv-file)."
    sEndogenTSnameHelp = "Endogenous Variable (TS,column name)  in the dataset."
    sExogenTSnameHelp = "Exogenous Variables. A string contains the comma-separated the column names of TS  in the dataset. "
    sTimeStampHelp = "Time Stamp (column) name in the dataset."
    sWWidthHelp = "Window width for endogenous variable plotting"
    parser = argparse.ArgumentParser(description=sDescriptor)


    parser.add_argument('-c', '--csv_dataset', dest='cl_dset', action='store', help=sCSVhelp)
    parser.add_argument('-e', '--endogen', dest='cl_endots', action='store', default='Imbalance',
                        help=sEndogenTSnameHelp)
    parser.add_argument('-x', '--exogen', dest='cl_exogts', action='store', default='',
                        help=sExogenTSnameHelp)
    parser.add_argument('--timestamp', dest='cl_timestamp', action='store', default='Date Time',
                        help=sTimeStampHelp)
    parser.add_argument('-w','--width_plot', dest='cl_wwidth', action='store', default='512',
                        help=sWWidthHelp)
    parser.add_argument('--verbose', '-v', dest='cl_verbose', action='count', default=0)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    # command-line parser
    arglist = sys.argv[0:]
    argstr = ""
    for arg in arglist:
        argstr += " " + arg
    message0 = f"""  Command-line arguments

    {argstr}

Dataset path                        : {args.cl_dset}
Endogenious Time Series in dataset  : {args.cl_endots}
Exogenious Time Series in dataset   : {args.cl_exogts}
Timestamp in dataset                : {args.cl_timestamp}
Window width for Endogenous plotting: {args.cl_wwidth} 
 """
    return args, message0




def main(argc,argv):
    args,message0 =parsing()

    dt_col_name: str      = args.cl_timestamp  # "Date Time"
    endogen_col_name: str = args.cl_endots  # "Imbalance"
    exogen_list           = list(args.cl_exogts.split(','))
    csv_file:str          = args.cl_dset
    wwidth:int            = int(args.cl_wwidth)
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(endogen_col_name, date_time)

    listLogSet(str(folder_for_logging))  # A logs are creating
    msg2log(None, message0, D_LOGS["clargs"])
    msg2log(None, message1, D_LOGS["timeexec"])

    f = D_LOGS["main"]
    folder_for_train_logging = str(Path(os.path.realpath(f.name)).parent)

    message = f"""  Common parameters set

Dataset path                        : {csv_file}
Endogenious Time Series in dataset  : {endogen_col_name}
Exogenious Time Series in dataset   : {exogen_list}
Timestamp in dataset                : {dt_col_name} 
Window width for Endogenous plotting: {wwidth} 
Folder for control logging          : {folder_for_logging}
Folder for plots                    : {D_LOGS["plot"]}

Logs                                :
    {logList()}
            """
    msg2log(None, message, D_LOGS["main"])

    df=pd.read_csv(csv_file)
    y=np.array(df[endogen_col_name])
    y_mean=y.mean()
    m=len(exogen_list)

    X=np.zeros((len(df),len(exogen_list)),dtype=float)
    for i in range(len(exogen_list)):
        X[:,i]=df[exogen_list[i]]
    X_mean=X.mean(0)
    lr = lm.LinearRegression(fit_intercept=True,normalize=False, copy_X=True)
    reg=lr.fit(X,y)
    r=reg.score(X,y)
    y_pred=reg.predict(X)
    aa,yy_pred = myregr(X, y)
    qe=0.0
    msa=0.0
    l1=0.0
    meane=0.0
    n=len(df)
    qr=0.0
    for i in range(n):
        delta=(y[i]-y_pred[i])
        qe=qe+delta*delta
        qr=qr +(y_pred[i]-y_mean) * (y_pred[i]-y_mean)
        l1=l1+abs(delta)
        meane=meane + (delta)
        if abs(delta)>msa: msa=abs(delta)
    sig2 = qe / (float(n-1))
    sigreg2=qe/(float(n-len(exogen_list)-1))
    l1=l1/float(n)
    meane=meane/float(n)
    nu=n-1
    nu1=len(exogen_list)
    nu2=n-len(exogen_list)-1
    T_resid = abs(meane)*math.sqrt(float(n))/math.sqrt(sig2)
    F_all_coef=qr*float(nu2) /(qe*float(nu1))
    T_095_4000=1.96
    F_005_1_400=3.84
    message=f"""
Output: {endogen_col_name}   Mean value:{y_mean}
Inputs: {exogen_list}        Mean values: {X_mean}
Linear regression: y=Intersect + Sum( A[i]*x[i] )
Intercept: {reg.intercept_}  Coefficients: {reg.coef_}
MultiCorrelation Coefficient: {r}  Singular: {reg.singular_} Input features: {reg.n_features_in_}
Mean Residuals:{meane} Variance Residuls: {sig2}  Regression Residuals: {sigreg2}  MSE:{l1}

    Significance Tests:
H0: Residuals are zero  T {T_resid}  < T-dist(0.05,nju ={nu}: {T_095_4000}
HO: Coefficients are zero F {F_all_coef} <F-dist(0.05, nju1={nu1},nju2={nu2}): {F_005_1_400}

    
"""

    msg2log(None,message,f)
    plotRegr(X, y, y_pred, endogen_col_name, exogen_list, wwidth=wwidth,folder=D_LOGS["plot"], f=f)

    message = "Time execution logging stoped at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    msg2log(None, message, D_LOGS["timeexec"])
    closeLogs()  # The loga are closing

    return

def plotRegr(X:np.array, y:np.array, y_pred:np.array, endogen_col_name:str, exogen_list:list, wwidth:int=512,folder:str=None,
             f:object=None):
    # Plot outputs
    if folder is not None:
        pathFolder=Path(folder)
    else:
        pathFolder=""

    (n,m)=X.shape
    for i in range(len(exogen_list)):
        plt.scatter(X[:, i], y, color='black')
        plt.plot(X[:, i], y_pred, color='blue', linewidth=1,label=endogen_col_name)
        plt.xlabel(exogen_list[i])
        plt.ylabel(endogen_col_name)
        plt.title("{} as linear regression approximation.X-axes maped on {}".format(endogen_col_name,exogen_list[i]))
        plt.axis('tight')
        filepng= Path( Path(pathFolder) /Path( "{}_{}.png".format(endogen_col_name,exogen_list[i]))).with_suffix(".png")
        plt.legend()
        plt.savefig(filepng)

        plt.close("all")
    
    for n_first in range(0,n,wwidth):

        n_last=n_first+wwidth if n_first+wwidth<=n else n
        x=np.arange(n_first,n_last)
        plt.plot(x,y[n_first:n_last],color='blue',label='y=f(t)')
        plt.plot(x,y_pred[n_first:n_last],color='orange',label='y_pred=f(t)')
        plt.xlabel("Sample number")
        plt.ylabel(endogen_col_name)
        plt.title("{}-original and linear regression data (from {} to {})".format(endogen_col_name,n_first,n_last))
        plt.axis('tight')
        filepng = Path(Path(pathFolder) / Path("{}_{}_{}.png".format(endogen_col_name,n_first,n_last))).with_suffix(".png")
        plt.legend()
        plt.savefig(filepng)

        plt.close("all")

    return

def myregr(Z:np.array,y:np.array)->(np.array,np.array):
    (n,m)=Z.shape
    X=np.ones((n,m+1),dtype=float)
    for j in range(m):
        X[:,j+1]=Z[:,j]
    XTX=X.T.dot(X)
    XTY=X.T.dot(y)
    d=np.linalg.det(XTX)
    XTXinv=np.linalg.inv(XTX)
    E=XTXinv.dot(XTX)
    A=XTXinv.dot(XTY)
    Y_pred=X.dot(A)
    Res=np.subtract(y,Y_pred)
    return A,Y_pred

if __name__ == "__main__":

    main(len(sys.argv),sys.argv)