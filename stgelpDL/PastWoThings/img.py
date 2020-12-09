#!/usr/bin/python3

import argparse
import os
from datetime import datetime
import os
import sys
from pathlib import Path
from math import log

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv,det
import pywt
import pywt.data

from PastWoThings.pilim import rgba2rgb,png2jpg, kMeanCluster
from predictor.utility import msg2log

SOURCES_PATH = ""   #"/home/dmitryv/PastWithoutToughts"

APPROXIMATION ="Approximation"
DETAIL        = "Detail"
PAINTING_QUANTIFICATION = "Quantification"
PAINTING_FIGURES        = "figures"
__version__ = '0.0.1'

def parseCL():
    # command-line parser
    sDescriptor = 'Image Classifications'
    sAPhelp = "Absolute path to a folder contains an image files."
    sNClustHelp = "Number clusters to form from given set of images"
    sWVtypeHelp = "A type of used wavelet transformation"

    parser = argparse.ArgumentParser(description=sDescriptor)
    parser.add_argument('-f', '--src_folder', dest='cl_src', action='store', help=sAPhelp)
    parser.add_argument('-w', '--wv_type', dest='cl_wvtype', action='store', default='db2',help=sWVtypeHelp,
                        choices=['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10','db11','db12',
                                 'db13','db14','db15','db16','db17','db18','db19','db20','db21','db22','db23','db24',
                                 'db25','db26','db27','db28','db29','db30','db31','db32','db33','db34','db35','db36',
                                 'db37','db38'])

    parser.add_argument('-n', '--num_cluster', dest='cl_num_clusters', action='store', default=4,help=sNClustHelp)
    parser.add_argument('--verbose', '-v', dest='cl_verbose', action='count', default=0)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    # command-line parser
    arglist = sys.argv[0:]
    argstr = ""
    for arg in arglist:
        argstr += " " + arg
    message = f"""  Command-line arguments

             {argstr}


Image files are in                    : {args.cl_src}
Number clusters to form               : {args.cl_num_clusters}
Wavelet Tarnsformation (by I.Dobeshi) : {args.cl_wvtype}
             

     """
    with open("commandline_arguments.log", "w+") as fcl:
        msg2log(None, message, fcl)
    fcl.close()
    return args

def createApprox0(rgbArray:np.array,f:object=None, wavelet='db38',mode='symmetric')->(np.array):
    coeff_R, coeff_G, coeff_B = dec2D(rgbArray, f, wavelet, mode)
    arApprox = rec2D(coeff_R, coeff_G, coeff_B, f, wavelet, mode)
    return arApprox
def createApprox(rgbArray:np.array,f:object=None, wavelet='db38',mode='symmetric')->(np.array):
    coeff_R,coeff_G,coeff_B = dec2D(rgbArray, f, wavelet , mode )
    coeff_Rapp = setZeroDetail(coeff_R)
    coeff_Gapp = setZeroDetail(coeff_G)
    coeff_Bapp = setZeroDetail(coeff_B)
    arApprox = rec2D(coeff_Rapp, coeff_Gapp, coeff_Bapp, f, wavelet , mode )
    pass
    return arApprox

def createDetail(rgbArray:np.array,f:object=None, wavelet='db38',mode='symmetric')->(np.array):
    coeff_R,coeff_G,coeff_B = dec2D(rgbArray, f, wavelet , mode )
    coeff_Rdet = setZeroApprox(coeff_R)
    coeff_Gdet = setZeroApprox(coeff_G)
    coeff_Bdet = setZeroApprox(coeff_B)
    arDetail = rec2D(coeff_Rdet, coeff_Gdet, coeff_Bdet, f, wavelet , mode )
    pass
    return arDetail
def dec2D(rgbArray:np.array, f:object=None, wavelet='db38', mode='symmetric')->(list,list,list):
    ar, ag, ab = splitRGB2planes(rgbArray)
    # plt.imshow(ar)
    # plt.imshow(ag)
    # plt.imshow(ab)

    coeffs_R = pywt.wavedec2(ar, wavelet)
    coeffs_G = pywt.wavedec2(ag, wavelet)
    coeffs_B = pywt.wavedec2(ab, wavelet)
    return coeffs_R,coeffs_G,coeffs_B

def rec2D(coeffs_R:np.array, coeffs_G:np.array,coeffs_B:np.array, f:object=None, wavelet='db38', mode='symmetric')->(np.array):
    rArRec = pywt.waverec2(coeffs_R, wavelet, mode)
    gArRec = pywt.waverec2(coeffs_G, wavelet, mode)
    bArRec = pywt.waverec2(coeffs_B, wavelet, mode)
    recRGBarray=planes2RGD(rArRec, gArRec, bArRec, f, type = 'uint8')
    return recRGBarray

def setZeroDetail(coeffs:list)->(list):
    nlen=len(coeffs)
    for i in range(1,len(coeffs)):
        coeffs[-i] = tuple([np.zeros_like(v) for v in coeffs[-i]])
    return coeffs

def setZeroApprox(coeffs:list)->(list):
    coeffs[0] = tuple([np.zeros_like(v) for v in coeffs[0]])
    return coeffs

def splitRGB2planes(arRGB:np.array)->(np.array,np.array,np.array):
    rAr = np.array(arRGB[:, :, 0])
    gAr = np.array(arRGB[:, :, 1])
    bAr = np.array(arRGB[:, :, 2])

    return (rAr,gAr,bAr)

def planes2RGD(ar:np.array,ag:np.array,ab:np.array, f:object=None, type='float')->(np.array):
    if type=='uint8':
        argb = (np.dstack((ar, ag, ab))).astype(np.uint8)  # stacks 3 h x w arrays -> h x w x 3
    else:
        argb = np.dstack((ar, ag, ab))  # stacks 3 h x w arrays -> h x w x 3
    return argb
def plot_images(main_title:str, figures:dict, nrows:int=1, ncols:int=1, folder:str= "", f:object=None):
    """ Plot a dictionary of figures.

    :main_title: title for all dictionary
    :param figures: <title,figure> dictionary
    :param nrows: number of rows of subplots wanted in the figure
    :param ncols: number of columns of subplots wanted in the display
    :param folder: name of folder where plots will be saved
    :param f:  file handler for logging
    :return:
    """
    file_png=Path(Path(folder)/Path(main_title+".png"))
    fig,axelist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)),figures):
        axelist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        axelist.ravel()[ind].set_title(title)
        axelist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.savefig(file_png)
    plt.close("all")

    
    return


def plot_images_hist(main_title: str, figures: dict, nrows: int = 1, ncols: int = 1, folder: str = "", f: object = None)->(dict):
    """ Plot a dictionary of figures.

    :main_title: title for all dictionary
    :param figures: <title,figure> dictionary
    :param nrows: number of rows of subplots wanted in the figure
    :param ncols: number of columns of subplots wanted in the display
    :param folder: name of folder where plots will be saved
    :param f:  file handler for logging
    :return:
    """
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)
    histDecRec={}
    file_png = Path(Path(folder) / Path(main_title + ".png"))
    fig, axelist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in zip(range(len(figures)), figures):
        axelist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        axelist.ravel()[ind].set_title(title)
        axelist.ravel()[ind].set_axis_off()
        hist_channels={}
        for channel_id,c in zip(channel_ids,colors):
            image=figures[title].copy()
            histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=256, range=(0, 256), density=True)
            hist_channels[c]=histogram.copy()
            axelist.ravel()[ind+ncols].plot(bin_edges[0:-1],histogram,color=c)
        histDecRec[title]=hist_channels
    plt.tight_layout()
    plt.savefig(file_png)
    plt.close("all")

    return histDecRec

def plotHist(file_png:str, image:np.array,f:object=None)->(dict):
    colors=("r","g","b")
    channel_ids=(0,1,2)

    #create the histogram plot< with three lines, one for each color
    plt.xlim(256)
    hist_channel={}
    for channel_id,c in zip(channel_ids,colors):
        histogram, bin_edges = np.histogram(image[:,:,channel_id],bins=256, range=(0,256),density=True)
        plt.plot(bin_edges[0:-1],histogram,color=c)
        hist_channel[c]=histogram.copy()

    plt.xlabel("Color value")
    plt.elabels("Pixels")
    plt.show()
    plt.savefig(file_png)
    plt.close("all")
    return hist_channel

def flattenArray(a:np.array):
    (n,m,p)=a.shape
    return a.reshape((-1,3),order='F')

def mcovEstim(a:np.array)->(np.array,np.array):
    cov=np.round(np.cov(a.T),4)
    mean=np.round(np.mean(a,axis=0),4)
    return mean,cov

def kl_divergence(p:np.array,q:np.array)->(float):

    p=p + 1e-15
    q=q + 1e-15
    # try:
    #     sum=0.0
    #     n=min(len(p),len(q))
    #     for i in range(n):
    #         sum=sum + p[i] * (log(p[i])-log(q[i]))
    # except:
    #     print("exception","i={} p={} q={}".format(i,p[i],q[i]))

    return sum(p[i] * log(p[i]/q[i]) for i in range(len(p)) )
    # return sum

def KLdivergence(mnp0:tuple,mnp1:tuple,f:object=None)->float:

    m,cov=mnp0
    m1,cov1=mnp1
    (k,)=m.shape
    cov1det = det(cov1)
    if cov1det<1e-08:
        for i in range(k):
            cov1[i,i]=cov1[i,i] + 1e-05
    covdet = det(cov)
    if covdet < 1e-08:
        for i in range(k):
            cov[i, i] = cov[i, i] + 1e-05
    cov1inv=inv(cov1)
    cov1det=det(cov1)
    covdet=det(cov)
    lnpart=np.log(cov1det/covdet) -float(k)
    trpart=0.0
    for i in range(k):
        for j in range(k):
            trpart =trpart + cov1inv[i,j]*cov[j,i]
    fisherpart=0.0
    for j in range(k):
        for i in range(k):
            fisherpart=fisherpart + (m1[j]-m[j])*cov1inv[i,j]*(m1[i]-m[i])

    klDiv=round((trpart + fisherpart + lnpart)/2.0,6)

    return klDiv

def main(argc,argv):
    args = parseCL()
    if not args.cl_src:
        print("Please set the absolute path to image files folder. ")
        sys.exit(1)
    sources_path=args.cl_src
    wavelet=args.cl_wvtype
    cluster_max = int(args.cl_num_clusters)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    folder_paint = Path(dir_path) / "Paintings" / date_time
    folder_log = Path(dir_path) / "Paintings_Log" / date_time
    Path(folder_paint).mkdir(parents=True, exist_ok=True)
    Path(folder_log).mkdir(parents=True, exist_ok=True)
    suffics = '.log'
    file_for_logging = Path(folder_log, Path(__file__).stem).with_suffix(suffics)
    fl = open(file_for_logging, 'w+')

    dict_repository = Path(folder_log / "Paintings_dict")
    Path(dict_repository).mkdir(parents=True, exist_ok=True)


    painting_files = [f for f in os.listdir(sources_path) if os.path.isfile(os.path.join(sources_path, f))]

    mnd_msg,pdf_msg, X, labelX = paDrive(sources_path,painting_files, wavelet, folder_paint, folder_log, dict_repository, f=fl)
    msg="Multivariate Normal Distribution"
    msg2log("Wavelet {},Model -".format(wavelet),msg,fl)
    for item in mnd_msg:
        msg2log("",item,fl)
    msg2log("", "\n\n", fl)
    msg = "Probability Deniy Fution (Histogram)"
    msg2log("Wavelet {},Model -".format(wavelet), msg, fl)
    for item in pdf_msg:
        msg2log("", item, fl)
    #classifcation
    name = "Paintings"
    cluster_centers, cluster_labels = kMeanCluster(name, X, labelX, cluster_max = cluster_max, type_features = 'pca',
                                                   n_component = 6, file_png_path =folder_log, f=fl)



    fl.close()
    return 0





def paDrive(sources_path:str, imglist, wavelet:str, folder_paint:str, folder_log:str, dict_repository:str,
            f:object=None)->(list,list,np.array, list):



    all_paints={}
    histograms={}
    lmsg_mnd=[]
    lmsg_pdf=[]
    X=np.zeros((len(imglist),6),dtype=float)
    labelX=[]
    iRow=0
    for fimg in imglist:

        fimage=str(Path(Path(sources_path)/Path(fimg)))
        # fimage = "/home/dmitryv/Downloads/magrit49600.jpg"
        #wavelet='db2'#'db38'
        p=Path(fimage)
        title=p.stem
        labelX.append(title)
        suff = p.suffix
        fullparentpath = p.parents[0]

        rgbArray = plt.imread(fimage)
        row, col, ch = rgbArray.shape

        if ch==4 or suff=='.png':
            newimage = Path(fullparentpath / 'converted' /str(title + '_cnvrtd')).with_suffix('.jpg')
            if newimage.exists() or newimage.is_file():
                newimage.unlink()
            if ch==4:
                rgba2rgb(fimage, str(newimage))
            elif suff=='.png':
                png2jpg(fimage, str(newimage))
            rgbArray = plt.imread(str(newimage))

        paint_info, message=impPerform(title,rgbArray,wavelet=wavelet,f=None)
        lmsg_mnd.append(message)
        # msg2log("Model-Multivariate Normal Distribution", message, f)
        plot_images(                  title, paint_info["figures"], nrows=1, ncols=3, folder=folder_paint, f=f)
        all_paints[title] = paint_info
        histDecRec = plot_images_hist(title, paint_info["figures"], nrows=2, ncols=3, folder=folder_paint, f=f)
        kl_app,kl_det =  getKL(histDecRec, title)
        (nrow,ncol,ch)=rgbArray.shape
        message = "{:<30s} KL(Img||Appr)={:>10.3E} KL(Img||Appr)per pxl={:>10.3E} KL(Img||Appr)={:>10.3E} KL(Img||Appr)per pxl={:>10.3E}".format(title,
                            kl_app,kl_app/(nrow*ncol), kl_det, kl_det/(nrow*ncol))

        message = "{:<30s} ({:^4d}x{:^4d}x{:^1d} pixels):   KL(Img,Appr)={:>10.3E} KLpxl(Img,Appr)={:>10.3E} KL(Img,Detail)={:>10.3E} KLpxl(Img,Detail)={:>10.3E}".format(
            title,  nrow, ncol, ch, kl_app,kl_app/(nrow*ncol), kl_det, kl_det/(nrow*ncol))

        lmsg_pdf.append(message)
        # msg2log("Model-histogram pdf",message,f)
        histograms[title] = histDecRec
        channel_dict=histDecRec[title]
        X[iRow, 0] = kl_app
        X[iRow, 1] = kl_det
        X[iRow, 2] = paint_info[PAINTING_QUANTIFICATION]["KL_appr"]
        X[iRow, 3] = paint_info[PAINTING_QUANTIFICATION]["KL_detail"]
        himg254= min(channel_dict['r'][-1], channel_dict['g'][-1], channel_dict['b'][-1])+1e-15
        himg1 = min(channel_dict['r'][1], channel_dict['g'][1], channel_dict['b'][1])+1e-15
        X[iRow, 4] = -log(himg254)  #KL(White block ||Image)
        X[iRow, 5] = -log(himg1 )  #KL(Black block ||Image)
        iRow+=1

    return lmsg_mnd,lmsg_pdf,X, labelX


def getKL(histDecRec:dict, title:str):
    kl_app = max( kl_divergence(histDecRec[title]['r'],histDecRec[APPROXIMATION]['r']),
                  kl_divergence(histDecRec[title]['g'], histDecRec[APPROXIMATION]['g']),
                  kl_divergence(histDecRec[title]['b'], histDecRec[APPROXIMATION]['b']))
    kl_det = max( kl_divergence(histDecRec[title]['r'], histDecRec[DETAIL]['r']),
                  kl_divergence(histDecRec[title]['g'], histDecRec[DETAIL]['g']),
                  kl_divergence(histDecRec[title]['b'], histDecRec[DETAIL]['b']))
    return kl_app,kl_det



def impPerform(title:str,rgbArray:np.array,wavelet='db38',f:object=None)->(dict):
    figures ={}
    # fimage = "/home/dmitryv/Downloads/magrit49600.jpg"
    # rgbArray = plt.imread(fimage)
    row, col, ch = rgbArray.shape
    rgbflatten = flattenArray(rgbArray)
    m,cov=mcovEstim(  rgbflatten)
    rgbflatten=None
    del rgbflatten

    rgbApprox = createApprox(rgbArray, f = None, wavelet = 'db38' )
    approxflatten= flattenArray(rgbApprox)
    mapprox, covapprox = mcovEstim(approxflatten)
    approxflatten=None
    del approxflatten

    klRGB_approx = KLdivergence((m,cov), (mapprox, covapprox), f = None)

    rgbDetail = createDetail(rgbArray, f=None,   wavelet = 'db38')
    detailflatten = flattenArray(rgbDetail)
    mdetail, covdetail = mcovEstim(detailflatten)
    detailflatten=None
    del detailflatten

    klRGB_detail = KLdivergence((m, cov), (mdetail, covdetail), f=None)
    measurments = {}
    measurments["Width"] = col
    measurments["Height"] = row
    measurments["Channels"]= ch
    N= row* col* ch
    measurments["KL_appr"]=round(klRGB_approx,4)
    measurments["KL_appr_per_pixel"] = round(klRGB_approx/N,   6)
    measurments["KL_detail"] = round(klRGB_detail,4)
    measurments["KL_detail_per_pixel"] = round(klRGB_detail/N, 6)

    figures={title: rgbArray, APPROXIMATION:rgbApprox, DETAIL:rgbDetail}

    message = "{:<30s} ({:^4d}x{:^4d}x{:^1d} pixels):   KL(Img,Appr)={:>10.3E} KLpxl(Img,Appr)={:>10.3E} KL(Img,Detail)={:>10.3E} KLpxl(Img,Detail)={:>10.3E}".format(
        title,
        measurments["Height"], measurments["Width"], measurments["Channels"],
        measurments["KL_appr"], measurments["KL_appr_per_pixel"],
        measurments["KL_detail"], measurments["KL_detail_per_pixel"])

    paint_info={PAINTING_QUANTIFICATION: measurments, PAINTING_FIGURES:figures }

    return paint_info, message

if __name__=="__main__":
    imglist=[
    "/home/dmitryv/PastWithoutToughts/wp2481139_magritte_wallpaper.png",
    "/home/dmitryv/PastWithoutToughts/Magritte_ObstackleVoid.png",

    "/home/dmitryv/PastWithoutToughts/magrit49600.jpg",
    "/home/dmitryv/PastWithoutToughts/ADOWxEA_magritte_wallpaper.jpg",
    "/home/dmitryv/PastWithoutToughts/ThreatningWeather.png",
    ]

    # files = [f for f in os.listdir(SOURCES_PATH) if os.path.isfile(os.path.join(SOURCES_PATH, f))]
    main (len(sys.argv),sys.argv)
