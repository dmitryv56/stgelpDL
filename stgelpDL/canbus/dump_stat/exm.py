#!/usr/bin/python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
canbusdump = "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
""" (1613030440.608181) vcan0 166#D0320018 """

def stathist(canbusdump:str=None,f:object=None):
    ff=open(canbusdump,'r')
    line =ff.readline()
    values=[]
    sizes=[]
    a=ff.readline()
    v0=float(line.split(' ')[0][1:-1])

    ss=line.split('#')[1].rstrip()
    if len(ss)%2 !=0:
        ss='0'+ss
    s0=len(ss)/2
    for line in ff:
        v = float(line.split(' ')[0][1:-1])
        ss = line.split('#')[1].rstrip()
        if len(ss) % 2 != 0:
            ss = '0' + ss
        s = len(ss) / 2
        values.append(abs(round((v-v0)*1e6,3)))
        sizes.append(s0)
        v0=v
        s0=s
    ff.close()

    dt =np.array(values)
    n,=dt.shape
    minDt=dt.min()
    maxDt=dt.max()
    meanDt=dt.mean()
    stdDt=dt.std()
    sz = np.array(sizes)
    n, = sz.shape
    minSz = sz.min()
    maxSz = sz.max()
    meanSz = sz.mean()
    stdSz = sz.std()
    r=np.corrcoef(dt,sz)
    name_file=Path(canbusdump).stem
    file_png="PacketIntervals_{}_{}_packets.png".format(name_file,n+1)

    histPlot(a=dt, nbins=20,title="Packet interval histogram", file_png=file_png)

    file_png = "DataFieldSize_{}_{}packets.png".format(name_file, n + 1)

    histPlot(a=sz, nbins=4, title="Packet data size histogram", file_png=file_png)
    msg=f""" {canbusdump}
    Primary statistics for CANbus packet intervals (microSec)
    Number packets = {n+1} 
    mean           = {meanDt} 
    std            = {stdDt} 
    min            = {minDt} 
    max            = {maxDt}
    Data field size(bytes)
    mean           = {meanSz}
    std            = {stdSz}
    min            = {minSz}
    max            = {maxSz}
    Correlation coef (Packet Intervals, Data sizes) = {r[0,1]}
"""
    f.write(msg)
    return

def histPlot(a:np.array=None,nbins:int=10,title:str=None,file_png:str=None):
    count, bins, ignored = plt.hist(a, nbins, density=True)
    plt.title(title)
    # plt.show()

    plt.savefig(file_png)
    plt.close("all")

def dumpAnalize(l_canbusdump:list=None):
    with open("PrimStatPacketIntervals.log", 'w') as f:
        for canbusdump in l_canbusdump:
            stathist(canbusdump=canbusdump, f=f)

    return


pass
def main():
    l_canbusdump = ["/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log",
                    "/home/dmitryv/ICSim/ICSim/candump-2021-02-11_100040.log",
                    "/home/dmitryv/ICSim/ICSim/candump-2021-04-05_130814.log",
                    "/home/dmitryv/ICSim/ICSim/candump-2021-04-05_131748.log"]
    dumpAnalize(l_canbusdump=l_canbusdump)

if __name__=="__main__":
   main()
