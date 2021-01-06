#!/usr/bin/python3

import argparse
import os
import sys
import copy
from datetime import datetime

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f as Fdist

from stcgelDL.blddsddata import logMatrix
from predictor.utility import msg2log
from simTM.auxapi import dictIterate,listIterate
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from stcgelDL.cfg import GlobalConst
from stcgelDL.blddsddata import logMatrix

H1=1
H2=0


__version__ ="0.0.1"

def parsing():
    # command-line parser
    sDescriptor = 'Information Criterions for linear regression parameters estimation for several  samples ' + \
                  'of the (green) electricity load processes'
    sRpshelp    = "Absolute path to a  dataset repository (folder)."
    sDshelp     = "List of dataset names/A string contains the comma-separated file names in the dataset repository " +\
                  "without file extensions( .csv)"

    sEndogenTSnameHelp = "Endogenous Variable (column name)  in the datasets."
    sExogenTSnameHelp  = "All Exogenous Variables. A string contains the comma-separated the column names in the " + \
                         "datasets. "
    sExclExogenTSnameHelp = "The Exogenous Variables for which statistical hypothesa about zero parameters is tested."
    sTimeStampHelp     = "Time Stamp (column) name in the datasets."
    sWWidthHelp = "Window width for endogenous variable plotting."
    sAlfaHelp = "Significal level (alfa), a probability threshold below which the tested hypothesis will be rejected."
    parser = argparse.ArgumentParser(description=sDescriptor)


    parser.add_argument('-r', '--repository', dest='cl_rps', action='store', help=sRpshelp)
    parser.add_argument('-d', '--datasets',   dest='cl_ds',  action='store', default="low_co2,mid_co2, high_co2",
                        help=sDshelp)
    parser.add_argument('-e', '--endogen', dest='cl_endog', action='store', default='CO2',
                        help=sEndogenTSnameHelp)
    parser.add_argument('-x', '--exogen', dest='cl_exogs', action='store', default="Diesel_Power, Pump_Power",
                        help=sExogenTSnameHelp)
    parser.add_argument('-l', '--exclude', dest='cl_exclexogs', action='store', default="Pump_Power",
                        help=sExclExogenTSnameHelp)
    parser.add_argument('--timestamp', dest='cl_timestamp', action='store', default='Date Time',
                        help=sTimeStampHelp)
    parser.add_argument('-w','--width_plot', dest='cl_wwidth', action='store', default='512',
                        help=sWWidthHelp)
    parser.add_argument('-a', '--alfa', dest='cl_alfa', action='store', default='0.05',
                        help=sAlfaHelp)
    parser.add_argument('--verbose', '-v', dest='cl_verbose', action='count', default=0)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()
    GlobalConst.setVerbose(args.cl_verbose)
    # command-line parser
    arglist = sys.argv[0:]
    argstr = ""
    for arg in arglist:
        argstr += " " + arg
    message0 = f"""  Command-line arguments

    {argstr}

Repository path                     : {args.cl_rps}
Datasets                            : {args.cl_ds}
Endogenious factors  in datasets    : {args.cl_endog}
Exogenious  factors in datasets     : {args.cl_exogs}
Excluded Exogenious  factors        : {args.cl_exclexogs}
Timestamp in dataset                : {args.cl_timestamp}
Window width for Endogenous plotting: {args.cl_wwidth} 
Significant level                   : {args.cl_alfa} 
 """
    return args, message0





""" An input data lists contains a list of the  input matrices (observation matrices) and a list of the output vectors.  
All input matrices have the same   number of the features (column numbers) and different number of the rows. The number 
of rows in the each matrix should be equival to the size of according output vector.
The  checkInputDataValid()-function checks these requirements and returns a status and a tuple of calculated parameters.
The status is 0 for valid data, else -1.
The tuple contains (k,listN,p,n) , where k- is a number of categories (the number of the matrices and vectors),
listeN - is a list of the rows amountnumber of observations.
"""
def checkInputDataValid(lstX:list=None,lstY:list=None,f:object=None)->(int,tuple):
    """

    :param lstX:
    :param lstY:
    :param f:
    :return: int, (int,list, int,int)
    """
    ret=-1
    rettuple=(-1,[],-1,-1)
    if lstX is None or lstY is None:
        msg = "No input lists of arrays"
        msg2log(None, msg, f)
        return ret,rettuple

    if not lstX or not lstY:
        msg = "Empty input lists of arrays"
        msg2log(None, msg, f)
        return ret,rettuple

    k=len(lstX)
    k1=len(lstY)

    if (k1 != k):
        msg = "The input lists have a different naumber items: {} vs {}".format(k,k1)
        msg2log(None, msg, f)
        return ret,rettuple
    lstP=[]
    lstN=[]
    lstNy=[]
    for item in lstX:
        X:np.array=item
        (n,p)=X.shape
        lstP.append(p)
        lstN.append(n)
    for item in lstY:
        y:np.array=item
        (n,)=y.shape
        lstNy.append(n)
    p=lstP[0]
    for i in range(len(lstP)):
        if p!=lstP[i]:
            msg="The feature nimbers are different: {} vs {}".format(p,lstP[i])
            msg2log(None,msg,f)
            return ret,rettuple
    if lstN!=lstNy:
        msg="Different sample sizes:\n{}\n{}".format(lstN,lstNy)
        msg2log(None, msg, f)
        return ret,rettuple
    rettuple=(k,lstN,p,sum(lstN))
    ret=0

    return ret,rettuple



def invs(title:str=None, S:np.array=None,f:object=None):

    if S is None :
        msg="3D array not found"
        msg2log("invs", msg, f)
        return None

    (k,p,p)=S.shape
    Sinv = np.zeros((k, p, p), dtype=float)
    for m in range(k):
        msg=""
        try:
            d = np.linalg.det(S[m,:,:])
            message="Sample: {} Determinant: {}".format(m,d)
            msg2log(None,message,f)
            Sinv[m,:,:] = np.linalg.inv(S[m,:,:])
        except ValueError as e:
            msg = "O-o-ops! I got a ValueError - reason  {}\n{}".format(str(e), sys.exc_info())
        except:
            msg = "O-o-ops! I got an unexpected error - reason  {}\n".format(sys.exc_info())
        finally:
            if len(msg) > 0:
                # msg2log("invs", msg, D_LOGS["except"])
                # log2All()
                msg2log("invs", msg,f)
    return Sinv

def estb(title:str=None, Sinv:np.array=None, XtY:np.array=None, f:object=None)->np.array:
    if Sinv is None or XtY is None:
        msg="3D array not found"
        return None

    (k,p,p)=Sinv.shape
    b= np.zeros((k, p), dtype=float)
    for m in range(k):
        b[m,:]=Sinv[m,:,:].dot(XtY[m,:])
    return b


"""Kulback-Liebler divergence is a way to for statistical hypothesis about of parameters of the linear models testing.
 
"""
class kldivEst():
    def __init__(self,title:str=None, categories:list=None,k:int=2,p:int=2, n:int=16, lstX:list =None, lstY:list=None,f:object=None):
        self.k     = k    # number of categories (observation matrices and outputs )
        self.p     = p    # number of the features in observation matrices.
        self.n     = n    # n - number of observations, n=n0+n1+... +nk-1
        self.lstX  = lstX # list of observation matrices [0X(n0,p),1X(n1,p), ...,(k-1)X(nk-1,p)
        self.lstY  = lstY # list of outputs [y(n0),...,Y(nk-1)
        self.f     = f
        self.div   = 0.0
        self.SH1   = np.zeros((self.k, self.p, self.p), dtype=float)      # shape is (k,p,p)
        self.SH2   = np.zeros((1,      self.p, self.p), dtype=float)       # shape is (1,p,p)
        self.XtYH1 = np.zeros((self.k, self.p),         dtype=float)      # shape is (k,p)
        self.XtYH2 = np.zeros((1,      self.p),         dtype=float)      # shape is (1,p)
        self.bH1   = np.zeros((self.k, self.p),         dtype=float)
        self.bH2   = np.zeros((1,      self.p),         dtype=float)
        self.d_ANOVA = {}
        self.title = title
        if categories is not None:
            self.catecories = categories
        else:
            self.categories=[str(i) for i in range(self.k)]

        self.hist=history()

        pass
    def fit(self,lstX:list =None, lstY:list=None ):
        self.lstX = lstX  # list of observation matrices [0X(n0,p),1X(n1,p), ...,(k-1)X(nk-1,p)
        self.lstY = lstY  # list of outputs [y(n0),...,Y(nk-1)
        self.lineq()
        self.linestimation()
        self.anova()
        return

    def predict(self,x:np.array=None,hypothesis:int=H1,cat_ind:int=0)->np.array:
        if hypothesis==H1:
            y=x.dot(self.bH1[cat_ind,:])
        else:
            y = x.dot(self.bH2[0,:])
        return


    def lineq(self):

        for i in range(self.k):
            X: np.array = self.lstX[i]
            y: np.array = self.lstY[i]

            self.SH1[i, :, :] = X[:, :].T.dot(X[:, :])
            self.XtYH1[i, :] = X[:, :].T.dot(y[:])

        for m in range(self.k):
            for i in range(self.p):
                self.XtYH2[0, i] = self.XtYH2[0, i] + self.XtYH1[m, i]
                for j in range(self.p):
                    self.SH2[0, i, j] = self.SH2[0, i, j] + self.SH1[m, i, j]
        return  #S, XtY, SH2, XtYH2, N

    def linestimation(self):
        SinvH2 = invs(S=self.SH2, f=None)
        Sinv = invs(S=self.SH1, f=None)
        pass
        print(SinvH2)
        print(Sinv)
        E = np.zeros((1, self.p, self.p), dtype=float)
        E = SinvH2[0, :, :].dot(self.SH2[0, :, :])
        print(E)

        self.bH2 = estb(Sinv=SinvH2, XtY=self.XtYH2)
        self.bH1 = estb(Sinv=Sinv, XtY=self.XtYH1)

        print(np.round(self.bH2, 5))
        print(np.round(self.bH1, 5))
        return

    def anova(self):
        # H1
        ssH1 = 0.0
        ssH1df = self.p * self.k
        sst = 0.0
        sstdf = self.n
        for m in range(self.k):
            ssH1 += self.bH1[m, :].T.dot(self.SH1[m, :, :].dot(self.bH1[m, :]))
            y: np.array = self.lstY[m]
            sst += y.T.dot(y)
        sseH1 = sst - ssH1
        sseH1df = self.n - self.p * self.k
        sig2 = sseH1 / float(sseH1df)
        # H2
        ssH2 = self.bH2[0, :].T.dot(self.SH2[0, :, :].dot(self.bH2[0, :]))
        ssH2df = self.p
        ssbH2 = ssH1 - ssH2
        ssbH2df = self.p * (self.k - 1)
        mse: float = sseH1 / float(sseH1df)
        msb: float = ssbH2 / float(ssbH2df)

        F: float = msb / mse
        divKL = ssbH2 / sig2
        FdivKL = divKL / float(ssbH2df)
        crit = Fdist.ppf(q=1 - 0.05, dfn=ssbH2df, dfd=sseH1df)
        quantile = Fdist.ppf(FdivKL, dfn=ssbH2df, dfd=sseH1df)
        self.d_ANOVA = {"H1":    {"SS": ssH1, "SSdf": ssH1df, "SSE": sseH1, "SSEdf": sseH1df, "MSE": mse, "sig2": sig2},
                        "H2":    {"SS": ssH2, "SSdf": ssH2df, "SSB": ssbH2, "SSBdf": ssbH2df, "MSB": msb, "F": F},
                        "Total": {"SST": sst, "SSTdf": sstdf},
                        "divKL": {"J(H1,H2)": divKL, "F": FdivKL, 'dfn': ssbH2df, 'dfd': sseH1df, "CV": crit,
                                  "Q": quantile}}

        return   #d_ANOVA, F, ssbH2df, sseH1df

    def res2log(self):
        pass

class history():  #TODO
    def __init__(self):
        self.f = D_LOGS["train"]
        self.history={}


class linreg(): # TODO

    def __init__(self,name:str="", X:np.array=None,y:np.array=None,f:object=None):
        self.X=copy.copy(X)
        self.y=copy.copy(y)
        (self.n,self.p) = self.X.shape
        self.S=np.zeros((self.p,self.p),dtype=float)
        self.XtY=np.zeros((self.p),dtype=float)
        self.sumsig2:float = 0.0
        self.f =f

    def estimate(self):
        XTX = self.X.T.dot(self.X)
        XTY = self.X.T.dot(self.y)
        d = np.linalg.det(XTX)
        XTXinv = np.linalg.inv(XTX)
        E = XTXinv.dot(XTX)
        A = XTXinv.dot(XTY)
        Y_pred = self.X.dot(A)
        Res = np.subtract(self.y, Y_pred)

def readData(title:str="data",csv_file:str=None,endogen_col_name:str="CO2",exogen_list:list=["Diesel_Power"])->(np.array,np.array):
    df = pd.read_csv(csv_file)
    y = np.array(df[endogen_col_name])
    y_mean = y.mean()
    m = len(exogen_list)

    X = np.ones((len(df), len(exogen_list)+1), dtype=float)
    for i in range(len(exogen_list)):
        X[:, i+1] = df[exogen_list[i]]
    X_mean = X.mean(0)
    msg2log(None,"{} Output mean {} Input means {}".format(title,y_mean,X_mean, f=None))
    return y,X

def joinYX(y:np.array, X:np.array)->np.array:
    n,=y.shape
    (n,p)=X.shape
    Z=np.zeros((n,p+1),dtype=float)
    Z[:,0]=y
    Z[:,1:]=X
    return Z

def main(argc, argv):
    args,message0 =parsing()

    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))


    dsRepository=args.cl_rps
    ds_list=list(args.cl_ds.split(','))
    exogen_list = list(args.cl_exogs.split(','))
    exclexogen_list = list(args.cl_exclexogs.split(','))
    endogen_col_name = args.cl_endog
    dt_col_name =args.cl_timestamp
    wwidth=args.cl_wwidth
    alfa=float(args.cl_alfa)

    #sort exogenius list
    for item in exclexogen_list:
        exogen_list.remove(item)
        exogen_list.append(item)
    k=len(ds_list)   # number of samples
    output_numbers=1
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}_{}".format("KL_div",endogen_col_name, date_time)

    listLogSet(str(folder_for_logging))  # A logs are creating
    msg2log(None, message0, D_LOGS["clargs"])
    msg2log(None, message1, D_LOGS["timeexec"])
    fm=D_LOGS["main"]

    message=f"""
Repository path               : {dsRepository}
Datasets                      : {ds_list}
Endogenious factor in datasets: {endogen_col_name}
Exogenious factors in datasets: {exogen_list}
Excluded  Exogenious factors  : {exclexogen_list}
Timestamp in dataset          : {dt_col_name}
Window  width for Endogenous 
plotting                      : {wwidth}
Number of samples             : {k}
Number of outputs in the linear 
regression model              : {output_numbers}
Number of input factors in the 
linear regression model       : {len(exogen_list)}
Significant level             : {round(alfa*100,0)} % 
Log folder                    : {folder_for_logging} 
"""

    msg2log(None,message,fm)

    f=None
    # DSrepository="/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna"
    # lDSets = ["low_co2","mid_co2","high_co2"]
    lstX=[]
    lstY=[]
    msg = "\n\n{:<20s} {:<20s} {:<20s}".format("Dataset","Output mean", "Input factors means")
    msg2log(None,msg,fm)
    for item in ds_list:
        csv_file=Path(Path(dsRepository)/Path( item.strip())).with_suffix(".csv")
        y,X = readData(title=item, csv_file=csv_file)
        lstX.append(X)
        lstY.append(y)
        msg="{:<20s} {:<20.6f} {}".format(str(item),round(y.mean(),4), np.round(X.mean(),3))
        msg2log(None,msg,fm)
        if GlobalConst.getVerbose()>1:
            logMatrix(joinYX(y, X),title=item,f=D_LOGS["control"])
            msg2log(None, "\n\n", f=D_LOGS["control"])

    status, (k, lstN, p, N) = checkInputDataValid(lstX=lstX, lstY=lstY, f=f)
    if status != 0:
        print("exit due invalid input data")
        sys.exit(-1)

    print(k,lstN,p,N)
    kldiv = kldivEst(title=None,k=k,p=p,n=N,lstX=lstX,lstY=lstY,f=f)
    kldiv.lineq()
    kldiv.linestimation()
    kldiv.anova()
    msg = dictIterate(ddict=kldiv.d_ANOVA)
    msg2log(None, msg, f=None)

    message = "Time execution logging stoped at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    msg2log(None, message, D_LOGS["timeexec"])
    closeLogs()  # The logs are closing

    return


if __name__=="__main__":
    # X=np.ones((3,4,2),dtype=float)
    # (k,n,p)=X.shape
    # csv_file ="/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/low_co2.csv"
    # y_low, X_low = readData(title="low", csv_file=csv_file)
    # csv_file = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/mid_co2.csv"
    # y_mid, X_mid = readData(title="mid", csv_file=csv_file)
    # csv_file = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/high_co2.csv"
    # y_high, X_high = readData(title="high", csv_file=csv_file)
    # lstX =[X_low,X_mid,X_high]
    # lstY=[y_low,y_mid,y_high]
    #
    # k=len(lstY)
    # (N,p)=X_low.shape
    # S,XtY,SH2,XtYH2,N = sxty( title="test", lstX=lstX, lstY=lstY, f= None)
    # print(S)
    # print(SH2)
    # SinvH2 = invs(S=SH2, f=None)
    # Sinv = invs( S=S, f = None)
    # pass
    # print (SinvH2)
    # print(Sinv)
    # E=np.zeros((1,p,p),dtype=float)
    # E=SinvH2[0,:,:].dot(SH2[0,:,:])
    # print(E)
    # bH2=np.zeros((1,p),dtype=float)
    # b=np.zeros((k,p),dtype=float)
    # bH2 = estb( Sinv=SinvH2, XtY=XtYH2)
    # b = estb(Sinv=Sinv, XtY=XtY)
    #
    # print(np.round(bH2,5))
    # print(np.round(b,5))
    # d_anova,f,df1,df2 = anova( n=N, S=S, b=b, SH2=SH2, bH2=bH2, lstY=lstY,f = None)
    # print(d_anova)
    # msg = dictIterate(ddict=d_anova, max_width = 120, curr_width = 0)
    # msg2log(None,msg,f=None)
    # print(f,df1,df2)
    main(len(sys.argv),sys.argv)
    pass
