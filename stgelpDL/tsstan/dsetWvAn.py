#!/usr/bin/python3


import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

from predictor.control import ControlPlane
from predictor.utility import msg2log,vector_logging


def padTS(x:list, mode:str='smooth',f:object=None)->(np.array, tuple):
    n=len(x)
    (n1,n2)=(0,0)
    for i in range(2,16):
        if n== pow(2,i):
            break
        elif n>pow(2,i) and n<pow(2,(i+1)):
            dn=pow(2,(i+1)) - n
            if dn%2==0:
                (n1,n2)=(dn/2,dn/2)
            else:
                n2=(dn+1)/2
                n1=n2-1
            break
    if n1==0 and n2==0:
        y=np.array(x)
    else:
        y=pywt.pad(np.array(x),(n1,n2),mode)
    return y,(int(n1),int(n),int(n2))




def drive_tsWvAnalysis(ds, listTS,  cp):



    w = pywt.Wavelet('db3')
    filter_len=w.dec_len
    listTS.insert(0,"Imbalance")
    listTS.insert(1, "Real_demand")
    wvList=pywt.wavelist('db')
    wvList.insert(0,'haar')
    for item in listTS:
        pass

        y,(n1,n,n2) =padTS(ds[item], mode = 'smooth', f= cp.fp)
        (data_len,)=y.shape
        max_level = pywt.dwt_max_level(data_len,filter_len)
        coef_len=pywt.dwt_coeff_len(data_len,filter_len,mode='smooth')
        wvEnable=['db1','db2','db10','db20','db30','db38']
        for wvname in wvList:

            w = pywt.Wavelet(wvname)
            try:
                bbb=wvEnable.index(wvname)
            except ValueError:
                continue


            filter_len=w.dec_len
            msg2log("\n{} {} Timeseries length. {} wavelet. {} filter length, {} -mode\n".format(item, n, w.name, filter_len,'smooth'),
                    "Decomposition and Reconstruction for {} levels\n ".format(max_level),cp.fp)
            llevAD=pywt.wavedec(y,w,mode='smooth')
            cA=llevAD[0]
            level=0
            for i in range(len(llevAD)-1,0,-1):
                level = level + 1
                msg="\nDetail {} Level, {} coefficients\n".format(level, len(llevAD[i]))
                # msg2log(None,msg,cp.fp)
                coef = llevAD[i]
                vector_logging(msg, coef, 4, cp.fp)

                for j in range(len(coef)):
                    if abs(coef[j])<1e-01 : coef[j] =0
                vector_logging(msg, coef, 4, cp.fp)


            msg = "\nApproximation {} Level, {} coefficients\n".format(level,len(llevAD[0]))
            # msg2log(None, msg, cp.fp)
            vector_logging(msg, llevAD[0], 4, cp.fp)

            yrec=pywt.waverec(llevAD,w,mode='smooth')
            for i in range(len(llevAD) - 1, 0, -1):
                coef = llevAD[i]
                coef[:]=0.0
            yapp = pywt.waverec(llevAD, w, mode='smooth')
            if n1==0 or n2==0:

                plotDecRecTS("{}_Wavelet_Reconstruct_mode_{}".format(w.name, 'smooth'), w, item, y, yrec,yapp, cp)
            else:
                plotDecRecTS("{}_Wavelet_Reconstruct_mode_{}".format(w.name,'smooth'), w, item,
                             y[n1:-n2], yrec[n1:-n2],yapp[n1:-n2], cp)
            continue
        continue

    return

def plotDecRecTS(pref:str, w:pywt.Wavelet,item:str, TS:np.array, recTS:np.array,rectTSapprox:np.array, cp:ControlPlane):
    """

    :param df:
    :param cp:
    :return:
    """

    suffics = ".png"
    file_name = "{}_DecRec_{}".format(pref,item)

    file_png = file_name + ".png"
    path_png = Path(cp.folder_predict_log / file_png)

    try:
        # plt.plot(observations, viterbi_path, label="Viterbi path")
        # plt.plot(observations, hidden_sequence, label="Hidden path")
        plt.plot(TS, label=item)

        plt.plot(recTS, label="Reconstruct_".format(item))
        plt.plot(rectTSapprox, label="Approx_".format(item))
        plt.plot(TS - rectTSapprox, label="Detail_".format(item))

        numfig = plt.gcf().number
        fig = plt.figure(num=numfig)
        fig.set_size_inches(18.5, 10.5)

        fig.suptitle("{} reconstructed by {} - wavelet ".format( item, w.name), fontsize=24)
        plt.ylabel("Values", fontsize=18)
        plt.xlabel("Indexes", fontsize=18)
        plt.legend()
        plt.savefig(path_png)

    except:
        pass
    finally:
        plt.close("all")
    return



if __name__=="__main__":
    pass


