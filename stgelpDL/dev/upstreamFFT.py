#!/usr/bin/python3
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from scipy.fft import fft, ifft,rfft, irfft
import pandas as pd

from predictor.utility import msg2log




pass
""" Time-domain interpolation by using FFT guaranteeing conjugate symmetry and therefore only real interpolated signal.
The algorithm comprises a following steps:
 - perform N-point FFT on an N-point x(n) time sequence, yielding N frequence samples, X(m).
 - create an M*N-point spectral sequence Xint(m) initial  set to zeros.
 - assign Xint(m)=X(m) for 0<=m<=(N/2)-1
 - assign Xint(N/2) and Xint(M*N-N/2) equal to X(N/2)/2.0. ( To maintain conjugate symmetry and improve interpolation 
 accuracy.)
 - assign Xint(m)=X(q), where M*N-(N/2)+1<=m<=M*N-1, and (N/2)+1<=q<=N-1.
 - perform the M*N point inverse FFT of X(int(m) yielding the desired M*N-length interpolated xint(n) sequence.
 - multiply xint(n) by M to compensate for 1/M amplitude loss included by interpolation.
 The effective calculations by using FFT are for N=2**n and M=2**m. 
"""
def interfft(x:np.array=None,m:int=4,  f:object=None)->np.array:
    n,=x.shape
    n2=int(n/2)
    n21=n2+1
    # m=10
    x=np.random.normal(loc=0.0,scale=2,size=n)
    X=fft(x).real
    XX=float(m)* np.concatenate((X[:n2],X[n2:n21]/2,np.zeros(((m-1)*n-1),dtype=float),X[n2:n21]/2,X[n21:n]))
    xx=ifft(XX).real

    t=np.array([float(i) for i in range(n*m)])
    t1 = np.array([float(i) for i in range(n)])
    fig, (ax1,ax2)=plt.subplots(2,sharey=True)
    ax1.plot(t,xx,'b--')
    ax2.plot(t1, x, 'g--')
    fig.suptitle("Upstream x {} and source time series".format(m))

    plt.show()
    plt.savefig("Interpolation.png")
    plt.close("all")
    return xx

def interfft1(x:np.array=None,M:int=4, dbg:bool=False, f:object=None)->np.array:
    """

    :param x: source x-array on N-size
    :param M: interpolation factor, M*(N-1) zeros are stuffed
    :param dbg: for debug print
    :param f: log file handler
    :return: interpolated xint array of M*N size
    """

    N,=x.shape
    if dbg: f.write("N={}".format(N))
    MN=M*N
    if dbg: f.write("N={} M={} M*N={}".format(N,M,MN))
    if dbg: f.write("x =\n{}".format(x))
    N2:int=int(N/2)
    Xr=fft(x).real
    if dbg: f.write("X (real) after FFT\n{}".format(Xr))
    Xint=np.zeros((N*M),dtype=float)
    if dbg: f.write("Xint (zeros)\n{}".format(Xint))
    for m in range(N2):
        Xint[m]=Xr[m]
    if dbg: f.write("Xint (0-N/2-1 was set)\n{}".format(Xint))
    Xint[N2]=Xr[N2]/2
    Xint[MN-N2]=Xr[N2]/2
    if dbg: f.write("Xint (N/2 and MN-N/2 were set)\n{}".format(Xint))
    m = MN - N2 + 1
    for q in range(N2+1,N):
        Xint[m]=Xr[q]
        m+=1
    if dbg: f.write("Xint (MM-N/2+1 <=m<=MN-1 were set)\n{}".format(Xint))

    xint=ifft(M*Xint).real
    if dbg: f.write("xint (real) after inverseFFT\n{}".format(xint))
    # xint=M*xint
    if dbg: f.write("xint (after * M)\n{}".format(xint))
    return xint

def filterAfterInterpolation(order:int=4, btype:str="low", analog:bool=False, fsampling_Hz:float=1.0,
                             fcutoff_Hz:float=0.5, intFactor:int=4, sig:np.array=None, title:str="",folder:str="",
                             f:object=None)->np.array:

    nyquvist_freq_Hz=fsampling_Hz/2.0
    cutoff_frequency=fcutoff_Hz/nyquvist_freq_Hz   #normalized cutoff frequency
    b,a =signal.butter(order,cutoff_frequency,btype=btype,analog=analog)
    w,h =signal.freqs(b,a)
    filtersignal = signal.filtfilt(b, a, sig)

    try:
        message=f"""
        F-sampling   : {fsampling_Hz} Hz, Nyquvist freq: {nyquvist_freq_Hz} Hz, 
        Cut-off freq.: {fcutoff_Hz} Hz,   Normalized cut-off freq.: {cutoff_frequency}
        Filter order : {order}    Type: {btype} Analog : {analog}    
        """
        msg2log(None,message,f)
        pass
    except:
        pass
    finally:
        pass

    return filtersignal

def butterPlot(w,h,fnyq,file_png,f:object=None):

    # b, a = signal.butter(10, fnyq, 'low', analog=False)
    # w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')

    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(fnyq, color='green')  # cutoff frequency
    plt.show()

def main():
    ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020.csv")
    # col_name = 'lasts'
    dt_col_name = 'Date Time'
    aux_col_name = "Programmed_demand"
    data_col_name = "PowerGen"
    order = 4
    chunk_size=512
    start_chunk=50
    x=np.array(ds[data_col_name][start_chunk:start_chunk+chunk_size])
    delta=float(10*60)
    # x=np.concatenate((x,x,x,x))
    # chunk_size=chunk_size*4
    # """ normalization """
    # x=(x-x.mean())/x.std()

    m=10
    with open("log.log",'w') as ff:

        xint=interfft(x, m=m, f=ff)

        fcutoff_Hz =1.0 / (4.0 * delta)
        xintfltr=filterAfterInterpolation(order = order, btype = "low", analog = False, fsampling_Hz=float(m)*1.0/delta,
                                 fcutoff_Hz = fcutoff_Hz, intFactor= m, sig = xint, title="inter", folder="",
                                 f = ff)

       
        upstreamSignalSPDPlot(x,xint,xintfltr, chunk_size, m, fcutoff_Hz,float(delta))

    # ds[dst_col_name] = [round(ds[src_col_name][i] - ds[src1_col_name][i], 2) for i in range(len(ds))]
    #
    # ds.to_csv("~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020_CommonAnalyze.csv", index=False)
    return

def signalPlot(x:np.array=None, y:np.array=None, start_index:int=0, size:int=512, titlex:str="",titley:str="",
               folder:str="",f:object=None):
    title4file=titlex.replace(' ','_')
    file_png=Path(folder)/Path(title4file).with_suffix(".png")
    t = np.linspace(start_index, start_index+size, size, False)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, x[:size])
    ax1.set_title(titlex)
    amin=min(x[:size].tolist())
    amax = max(x[:size].tolist())
    ax1.axis([start_index, start_index+size, amin, amax])
    ax1.set_xlabel('Observation index')
    if y is not None:
        ax2.plot(t, y[:size])
        ax2.set_title(titley)

        ax2.axis([start_index, start_index + size, amin, amax])
        ax2.set_xlabel('Observation index')
    plt.tight_layout()
    plt.show()
    plt.savefig(file_png)
    plt.close("all")
    return

def upstreamSignalSPDPlot(x:np.array,xint:np.array,xintfilt:np.array, n:int, m:int,cut_off:float,delta:float ):

    Fs=1.0/delta
    Fsint=float(m)/float(delta)
    t = np.array([float(i) for i in range(n * m)])
    t1 = np.array([float(i) for i in range(n)])
    fig, (ax1, ax2,ax3) = plt.subplots(3, sharey=True)
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(t, xintfilt, 'k--')
    ax00.set_title("Low pass Fs={} Hz, Fcut-off={} Hz".format(round(Fsint,5),round(cut_off,5)))
    ax00.set_xlabel("time index")
    ax00.set_ylabel('signal')
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.psd(xintfilt, NFFT=len(t), pad_to=len(t),Fs=Fsint)
    ax01.set_title("Power Spectral Density Fs={}".format(round(Fsint, 5)))

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(t, xint, 'b--')
    ax00.set_xlabel("time index")
    ax00.set_ylabel('signal')
    ax10.set_title("Upstream   x {}, Fs={} Hz".format(m, round(Fsint, 5)))
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.psd(xintfilt, NFFT=len(t), pad_to=len(t), Fs=Fsint)
    ax11.set_title("Power Spectral Density Fs={}".format(round(Fsint, 5)))

    ax20 = fig.add_subplot(gs[2, 0])
    ax20.plot(t1, x, 'r--')
    ax00.set_xlabel("time index")
    ax00.set_ylabel('signal')
    ax20.set_title("Time Series, Fs={} Hz".format(round(Fs,5)))
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.psd(xintfilt, NFFT=len(t), pad_to=len(t), Fs=Fs)
    ax21.set_title("Power Spectral Density Fs={} Hz".format(round(Fs, 5)))

    fig.suptitle("Upstream x {} and  signal".format(m))
    # fig.tight_layout()
    plt.show()
    plt.savefig("Interpolation.png")
    plt.close("all")

def upstreamPlot(x:np.array,xint:np.array,xintfilt:np.array, n:int, m:int,cut_off:float,delta:float ):
    t = np.array([float(i) for i in range(n * m)])
    t1 = np.array([float(i) for i in range(n)])
    fig, (ax1, ax2,ax3) = plt.subplots(3, sharey=True)
    ax1.plot(t, xintfilt, 'k--')
    ax1.set_title("Low pass filter F cut-off {}".format(round(cut_off,5)))
    ax2.plot(t, xint, 'b--')
    ax2.set_title("Upstream  {} factor, discretization {} sec".format(m, round(delta/m,2)))
    ax3.plot(t1, x, 'r--')
    ax3.set_title("Time Series, discretization {} sec".format(round(delta,2)))
    fig.suptitle("Upstream x {} and source time series".format(m))
    fig.tight_layout()
    plt.show()
    plt.savefig("Interpolation.png")
    plt.close("all")

def example():
    n=100
    n2=int(n/2)
    n21=n2+1
    m=10
    x=np.random.normal(loc=0.0,scale=2,size=n)
    X=fft(x).real
    XX=float(m)* np.concatenate((X[:n2],X[n2:n21]/2,np.zeros(((m-1)*n-1),dtype=float),X[n2:n21]/2,X[n21:n]))
    xx=ifft(XX).real

    t=np.array([float(i) for i in range(n*m)])
    t1 = np.array([float(i) for i in range(n)])
    fig, (ax1,ax2)=plt.subplots(2,sharey=True)
    ax1.plot(t,xx,'b--')
    ax2.plot(t1, x, 'g--')

    plt.show()





if __name__ == "__main__":
    example()
    main()
    pass
    # N=8
    # delta=600
    # t = np.arange(N)
    # x=np.sin(t)
    # m=4
    # xint=interfft(x, M = m)
    #
    # plt.plot(t,x)
    # plt.show()
    # t1=np.arange(len(xint))
    # er=np.zeros((len(xint)),dtype=float)
    # for i in range(0,len(xint),m):
    #     er[i]=xint[i]-x[int(i/m)]
    #
    # plt.plot(t1,xint)
    # plt.plot(t1, er)
    # plt.show()
    # fs=1.0/float(N*delta)
    # fnyq=1.0/(2.0*delta)
    #
    # print("delta={} sec N={} fs={} Hz Fnyquist ={} Hz".format(delta,N,fs,fnyq))
    # b,a=signal.butter(10,fnyq,'low',analog=False)
    # w,h =signal.freqs(b,a)
    # plt.semilogx(w, 20*np.log10(abs(h)))
    # plt.title('Butterworth filter frequency response')
    # # plt.xlabel('Frequency [radians / second]')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Amplitude [dB]')
    # plt.margins(0, 0.1)
    # plt.grid(which='both', axis='both')
    # plt.axvline(fnyq, color='green')  # cutoff frequency
    # plt.show()
    #
    #
    #
    # t = np.linspace(0, 1, 1000, False)  # 1 second
    # sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # ax1.plot(t, sig)
    # ax1.set_title('10 Hz and 20 Hz sinusoids')
    # ax1.axis([0, 1, -2, 2])
    #
    # sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
    # filtered = signal.sosfilt(sos, sig)
    # ax2.plot(t, filtered)
    # ax2.set_title('After 15 Hz high-pass filter')
    # ax2.axis([0, 1, -2, 2])
    # ax2.set_xlabel('Time [seconds]')
    # plt.tight_layout()
    # plt.show()
    #
    # t = np.linspace(0, delta*1000, 1000, False)  # 1 second
    # sig = np.sin(2 * np.pi * float(10.0/(1000.0*delta) )* t) + np.sin(2 * np.pi * float(20.0/(1000.0*delta)) * t)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # ax1.plot(t, sig)
    # ax1.set_title('10 Hz and 20 Hz sinusoids'.format(round(10.0/(1000.0*delta),6),round(20.0/(1000.0*delta)),6))
    # ax1.axis([0, delta*1000, -2, 2])
    #
    # sos = signal.butter(4, 10, 'low', fs=1000, output='sos')
    # filtered = signal.sosfilt(sos, sig)
    # ax2.plot(t, filtered)
    # ax2.set_title('After 15 Hz high-pass filter')
    # ax2.axis([0, 1000*delta, -2, 2])
    # ax2.set_xlabel('Time [seconds]')
    # plt.tight_layout()
    # plt.show()



    pass