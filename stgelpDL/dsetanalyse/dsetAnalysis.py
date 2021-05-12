#!/usr/bin/python3

import os
import sys
import pandas as pd
import copy
from pathlib import Path

import pandas as pd
import numpy as np
import math
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

from predictor.utility import msg2log
from simTM.auxapi import dictIterate,listIterate

def main(argc,argv)->int:
    data_col_names = ["Real_demand", "Diesel_Power", "WindGen_Power", "HydroTurbine_Power"]
    with open("Power_Stability.log","w") as f:
        for item in data_col_names:
            stab_area_estimation(data_col_name=item,f=f)
    return 0


"""Evaluate a statistics for segments of a time series.
For each segment(chunk) the scaling is carried out for mean segment value.
The bounds of stability area for the segment is evaluated as (1.0 +/- stability level) * mean.
The stability level us 0.1 by default (10%).
The number of observations into stability area and out of stability are are countered. 
"""
def stab_area_estimation(data_col_name:str="Diesel_Power",f:object=None):
    discret=10
    offset_start=42
    period=72
    stability_level=0.1
    dt_col_name="Date Time"
    # data_col_name="Diesel_Power"
    src_csv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_Diesel.csv"
    pass
    df=pd.read_csv(src_csv)
    n=len(df)
    l_agg=[]
    l_rate=[]
    main_stability_counter=0
    main_unstability_counter = 0
    main_off_counter = 0
    while offset_start+period<=n:
        d_agg={}
        v=df[data_col_name][offset_start:offset_start+period].values
        meanV=v.mean()
        min_stab=(1.0-stability_level)* meanV
        max_stab =(1.0 + stability_level) * meanV
        stdV=v.std()
        minV = v.min()
        maxV = v.max()
        dt=df[dt_col_name][offset_start]
        d_agg[dt_col_name]=dt

        d_agg["mean"]=round(meanV,3)
        d_agg["std"]=round(stdV,3)
        d_agg["min"] = round(minV,3)
        d_agg["max"] = round(maxV,3)
        d_agg["min_stab_level"] = round(min_stab,3)
        d_agg["max_stab_level"] = round(max_stab,3)
        d_agg["l" + data_col_name] = v.tolist()

        l_resid = [round(v[i] - meanV, 3) for i in range(len(v))]
        main_cnt,main_us_cnt,off_cnt,l_stab,l_unstab=stab_counter(v, min_stab, max_stab, discret = discret)
        powerRate(l_resid, offset_start,l_rate=l_rate)


        main_stability_counter=main_stability_counter +main_cnt
        main_unstability_counter=main_unstability_counter+main_us_cnt
        main_off_counter = main_off_counter + off_cnt
        d_agg["lresid"]=l_resid
        d_agg["lstab"] = l_stab
        d_agg["lunstab"] = l_unstab
        l_agg.append(d_agg)
        offset_start += period
    df1=pd.DataFrame(l_agg)
    csv_name="{}.csv".format(data_col_name)
    df1.to_csv(csv_name, index=False)
    df2 = pd.DataFrame(l_rate)
    csv_name = "{}_PowerRate.csv".format(data_col_name)
    df2.to_csv(csv_name, index=False)

    msg=f"""{data_col_name}
   
    Total operating time {offset_start * discret} minutes or {round(offset_start*discret/60,2)} hours.
    Total off time {main_off_counter} minutes or {round(main_off_counter/60,2)} hours.
    Stable operation for  {main_stability_counter} minutes or {round(main_stability_counter/60,2)} hours.
    Unstable operation for  {main_unstability_counter} minutes or {round(main_unstability_counter/60,2)} hours.
"""
    msg2log(None,msg,f)
    return 0


def stab_counter(v:np.array, min_stab:float,max_stab:float, discret:int=10)->(int,int,int,list,list):
    l_stab=[]
    l_unstab=[]
    main_cnt_stab=0
    main_cnt_unstab = 0
    off_cnt=0
    n,=v.shape
    cnt_stab=0
    cnt_unstab=0
    for i in range(n):
        if v[i]<=0.0:   # TODO
            off_cnt+=1
            continue
        if v[i]>=min_stab and v[i]<=max_stab:
            if cnt_unstab>0:
                l_unstab.append(cnt_unstab*discret)
                cnt_unstab=0
            cnt_stab+=1
            main_cnt_stab +=1
        else:
            if cnt_stab>0:
                l_stab.append(cnt_stab*discret)
                cnt_stab=0
            cnt_unstab+=1
            main_cnt_unstab+=1
    if cnt_stab>0:
        l_stab.append(cnt_stab*discret)
    if cnt_unstab>0:
        l_unstab.append(cnt_unstab*discret)
    return main_cnt_stab*discret, main_cnt_unstab*discret,off_cnt*discret,l_stab, l_unstab


""" Evaluation of the rate of change of the power value. 
The diff list is calculated like as diff[i]=l_resid[i+1]-l_resid[i], sorted.
The 'n' minimal and 'n' maximal diffs saved in the list by pairs {"timestamp":ind, 'rate":diff}  
"""
def powerRate(l_resid:list, start_offset:int,l_rate:list=[],discret:int=10):
    d_rate={}

    n=len(l_resid)
    n_rated=int (n/10) if int(n/10)>1 else 2
    l_diff=[ round((l_resid[i+1]-l_resid[i])/discret,3) for i in range(len(l_resid)-1)]
    l_sorted=sorted(l_diff)
    for i in range(n_rated):
        d_rate = {}

        d_rate["TimestampIndex"]=start_offset+l_diff.index(l_sorted[i])
        d_rate["PowerRate"]=l_sorted[i]
        l_rate.append(d_rate)
        d_rate = {}
        j=-i-1

        d_rate["TimestampIndex"] = start_offset + l_diff.index(l_sorted[j])
        d_rate["PowerRate"] = l_sorted[j]
        l_rate.append(d_rate)
    return


""" Evaluate the operation time of the engine at all different capacities with given power step.
The target plot is y=f(x). where x is the power (MWt) and Y- is operation time at this power level(minutes)"""
def operationTimeEnginesAtDifferentCapacites():
    data_col_names=["Diesel_Power","Real_demand","WindGen_Power","HydroTurbine_Power"]
    discret = 10

    dt_col_name = "Date Time"
    data_col_name = "Diesel_Power"
    src_csv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_Diesel.csv"

    df=pd.read_csv(src_csv)
    with open("../dev/engine.log", 'w') as ff:
        for item in data_col_names:
            res_list=[]
            v=df[item].values
            meanV=v.mean()
            stdV=v.std()
            minV=v.min()
            maxV=v.max()
            message=f"""
            
{item}
mean={round(meanV,3)}
std ={round(stdV,3)}
min ={round(minV,1)}
max ={round(maxV,1)}
"""
            msg2log(None,message,ff)

            cnt_dict = operationTimeEngine(data_col_name=item, dt_col_name=dt_col_name, v=v.tolist(), res_list=res_list,
                                discret=discret, minV=round(minV,1), maxV=round(maxV,1),f=ff)
            msg=dictIterate(cnt_dict)
            message=f"""
Power (MWt) Operation time (min) 
"""
            msg2log(None, message,ff)
            msg2log(None, msg, ff)

    return

def operationTimeEngine(data_col_name:str=None, dt_col_name:str=None,v:list=None, res_list:list=[],discret:int=10,
                        minV:float =0.0,maxV:float=0.0,step:float=0.1,f:object=None):
    nLevel=int(round((maxV-minV)/step,0))+1
    cnt_dict={round((minV +x*step),1):0 for x in range(nLevel)}
    for i in range(len(v)):
        cnt_dict[v[i]]=cnt_dict[v[i]]+discret

    plotOperationTime(dd = cnt_dict, data_col_name=data_col_name, f=f)
    return cnt_dict

def plotOperationTime(dd:dict=None, data_col_name:str=None,f:object=None):
    x,y=zip(*sorted(dd.items()))
    plt.plot(x,y)
    plt.title(data_col_name)
    plt.xlabel("Power , MWt")
    plt.ylabel("Operation time, minutes")
    png_file="{}.png".format(data_col_name)
    plt.savefig(png_file)
    plt.close("all")
    message = f"""
    Power (MWt) Operation time (min) 
    """
    msg2log(None, message, ff)
    for i in range(len(x)):
        msg2log(None,"{:<10.4f} {:<8d}".format(x[i],y[i]),f)

    return


""" Evaluation the total time of power gain and  drop."""
# D_STATES={0:"-1 MWt .. +1 MWt",
#           1:"+1 MWt .. +2 MWt",
#           2:"+2 MWt .. +3 MWt",
#           3:"+3 MWt .. +4 MWt",
#           4:"+4 MWt .. ",
#           5:"-1 MWt .. -2 MWt",
#           6:"-2 MWt .. -3 MWt",
#           7:"-3 MWt .. -4 MWt",
#           8:"-4 MWt .. ",
#           }

D_STATES={0:"-0.010 MWt/min .. +0.010 MWt/min",
          1:"+0.010 MWt/min .. +0.075 MWt/min",
          2:"+0.075 MWt/min .. +0.150 MWt/min",
          3:"+0.150 MWt/min .. +0.225 MWt/min",
          4:"+0.225 MWt/min .. ",
          5:"-0.010 MWt/min .. -0.075 MWt/min",
          6:"-0.075 MWt/min .. -0.150 MWt/min",
          7:"-0.158 MWt/min .. -0.225 MWt/min",
          8:"-0.225 MWt/min .. ",
          }

def totalPowerGainDropTimeEngines(repository:Path=None):


    data_col_names = ["Diesel_Power","Real_demand",  "WindGen_Power", "HydroTurbine_Power"]
    discret = 10

    dt_col_name = "Date Time"
    data_col_name = "Diesel_Power"
    state_col_name="State_{}".format(data_col_name)
    src_csv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_Diesel.csv"
    d_states={}
    df = pd.read_csv(src_csv)
    with open("../dev/PowerGainDropMWtMin.log", 'w') as ff:
        for item in data_col_names:
            res_list = []
            v = df[item].values
            meanV = v.mean()
            stdV = v.std()
            minV = v.min()
            maxV = v.max()
            state_col_name = "State_{}".format(item)

            message = f"""

{item}
mean={round(meanV, 3)}
std ={round(stdV, 3)}
min ={round(minV, 1)}
max ={round(maxV, 1)}
"""
            msg2log(None, message, ff)
            n,=v.shape
            states=[0 for i in range(n)]
            l_gain, l_drop = totalPowerGainDropTime(data_col_name=item, dt_col_name=dt_col_name, v=v.tolist(),
                                                    res_list=res_list, states=states, discret=discret,  minV=round(minV, 1),
                                                    maxV=round(maxV, 1), repository=repository, f=ff)
            df[state_col_name]=states
            un, cnt_st = np.unique(np.array(states), return_counts=True)
            states=[]
            # histGainDrop(l_gain=l_gain, data_col_name = item, title = "PowerGain", f=ff)
            # histGainDrop(l_gain=l_drop, data_col_name=item, title="Powerdrop", f=ff)

            msg=f""" {item}
{un}
{cnt_st}
"""
            msg2log(None,msg,ff)
            dd={}
            for i in range(len(un)):
                dd[D_STATES[un[i]]]=cnt_st[i]
            d_states[item]=dd
        msg = dictIterate(d_states)
        msg2log(None, msg, ff)

    csv_file=Path(repository /Path("updatedMwtMin")).with_suffix(".csv")
    df.to_csv(str(csv_file))

    return

""" Evaluation for each gain/drop power segment delta power (Pend -Pstart) where Pstart is the segment start power 
and Pend is the end of segment power. Evaluated pairs {deltaP,segement dyration} are collected into list.


"""
def totalPowerTime( v:list=None, res_list:list=[], states:list=None, discret:int=10,mode:str='gain',
                        minV:float =0.0,maxV:float=0.0,step:float=0.1,f:object=None):

    cnt_=0
    v_start=v[0]
    v_end=v[0]
    i_start=0
    i_end=0

    for i in range(1,len(v)):
        if v[i]==v[i-1] or (v[i]>v[i-1] and mode=='drop') or (v[i]<v[i-1] and mode =='gain'):
            if cnt_>0:
                res_list.append({cnt_:round(v_end-v_start,1)})
                if abs(v_end-v_start)>=1.0:
                    calcState(v_start=v_start, v_end=v_end, mode=mode,i_start=i_start,i_end=i_end, states=states,f=f)

                cnt_=0
            v_start=v[i]
            v_end=v[i]
            i_start=i
            i_end=i

        elif v[i]>v[i-1] and mode=='gain':  # power gain

            cnt_+=discret
            v_end=v[i]
            i_end=i
        elif v[i]<v[i-1] and mode=='drop':  # power drop

            cnt_+=discret
            v_end=v[i]
            i_end=i
        else:
            pass
    if cnt_>0:
        res_list.append({cnt_:round(v_end-v_start,1)})
        if abs(v_end - v_start) >= 1.0:
            calcState(v_start=v_start, v_end=v_end, mode=mode,i_start=i_start,i_end=i_end, states=states,f=f)

    return

""" Calculate the state according by rules:
    State1  1..2Mwt,   State1  2-3Mwt,   State3  3-4Mwt,        State4  4-...Mwt,
    State5  -1..-2Mwt, State6  -2..-3Mwt,State7  -3..-4Mwt,     State8  -4-...Mwt,
"""
def calcState(v_start:float=None,v_end:float=None, mode:str='gain', i_start:int=0,i_end:int=0, states:list=None,
              discret:int=10,f:object=None)->int:
    # vv = abs(v_end - v_start)
    # if mode == 'gain' and vv >= 1.0 and vv < 2.0:
    #     s = 1
    # elif mode == 'gain' and vv >= 2.0 and vv < 3.0:
    #     s = 2
    # elif mode == 'gain' and vv >= 3.0 and vv < 4.0:
    #     s = 3
    # elif mode == 'gain' and vv >= 4.0:
    #     s = 4
    # elif mode == 'drop' and vv >= 1.0 and vv < 2.0:
    #     s = 5
    # elif mode == 'drop' and vv >= 2.0 and vv < 3.0:
    #     s = 6
    # elif mode == 'grop' and vv >= 3.0 and vv < 4.0:
    #     s = 7
    # elif mode == 'drop' and vv >= 4.0:
    #     s = 8
    # else:
    #     s = 0

    vv = round(abs(v_end - v_start)/((i_end-i_start+1)*discret),4)
    if vv >= 0.01 and vv < 0.075:
        s = 1 if mode=='gain' else 5
    elif  vv >= 0.075 and vv < 0.150:
        s = 2 if mode=='gain' else 6
    elif vv >= 0.150 and vv < 0.225:
        s = 3 if mode=='gain' else 7
    elif  vv >= 0.225:
        s = 4 if mode=='gain' else 8
    else:
        s = 0
    for i in range(i_start, min( i_end+1,len(states)-1)):
        states[i]=s
    return s



A_COL="DeltaP"
B_COL="OperationTime"
C_COL="SegmentDuration"
def sumDeltaP(l_gain:list=None,f:object=None):
    d={}
    for item in l_gain:
        for key,val in item.items():
            if val in d.keys():
                d[val].append(key)
            else:
                d[val]=[key]

    #
    columns=[A_COL,B_COL,C_COL]
    dd={}
    df=pd.DataFrame(columns=columns)

    for deltaP,operTime in d.items():
        a=np.array(operTime)
        unique,counts = np.unique(a,return_counts=True)
        n,=unique.shape
        for i in range(n):
            df.loc[len(df)]={A_COL:deltaP,B_COL:unique[i]*counts[i],C_COL:unique[i]}


    return df


""" Evaluation Power gain drop for engine. 
The power gains/drops evaluate for given power observations at the segments of the increasing/decreaseing  powers.
The segments with same observated power are ignored.   
"""

def totalPowerGainDropTime(data_col_name:str=None, dt_col_name:str=None,v:list=None, res_list:list=[],states:list=None,
                           discret:int=10, minV:float =0.0,maxV:float=0.0,step:float=0.1,repository:Path=None,
                           f:object=None)->(list,list):


    nLevel=int(round((maxV-minV)/step,0))+1
    l_gain=[]
    l_drop=[]
    cnt_gain=0
    cnt_drop=0

    totalPowerTime(v=v, res_list = l_gain, states=states,discret = discret, mode= 'gain', f=f)

    df_Gain=sumDeltaP(l_gain=l_gain, f=f)
    csv_file=Path(repository / Path("{}_gain.csv".format(data_col_name)))
    df_Gain.to_csv(str(csv_file))
    msg2log(None, "Power Gain: {} saved in {}".format(data_col_name, csv_file), f)

    ind_list=[]
    totalPowerTime(v=v, res_list=l_drop, states=states,discret=discret, mode='drop', f=f)
    df_drop = sumDeltaP(l_gain=l_drop, f=f)
    msg2log(None,"{} saved in {}".format(data_col_name,csv_file),f)
    csv_file = csv_file=Path(repository / Path("{}_drop.csv".format(data_col_name)))
    df_drop.to_csv(str(csv_file))
    msg2log(None, "Power drop: {} saved in {}".format(data_col_name, csv_file), f)

    return l_gain, l_drop

def histGainDrop(l_gain:list=None, data_col_name:str=None, title:str="", f:object=None):
    plt.hist(l_gain, bins='auto')
    plt.title('{} {} histogram'.format(title,data_col_name))
    png_file = "{}_{}_histogram".format(title,data_col_name)
    plt.savefig(png_file)
    a = np.array(l_gain)
    mean_a = a.mean()
    std_a = a.std()
    min_a = a.min()
    max_a = a.max()

    message=f"""{title} : {data_col_name}
Sequence:{listIterate(l_gain)}

Mean duration : {mean_a} minutes.
Std           : {std_a}
Min duration  : {min_a}
Max duration  : {max_a}      
"""
    msg2log(None,message,f)
    return


if __name__=="__main__":
    # l_resid=[0.265, -0.035, 0.265, -0.335, 0.265, 0.065, 0.465, 0.665, 0.665, 0.265, 0.165, -0.535, -0.435, -0.435, 0.265,
    #  0.265, 0.665, 0.665, 0.465, 0.065, 0.865, 0.865, 0.865, 0.065, 0.165, 0.065, 0.065, -0.235, -0.135, -0.035, -0.235,
    #  -0.235, -0.435, -0.135, -0.235, -0.535, -0.335, -0.135, -0.235, -0.335, -0.435, -0.235, -0.235, -0.235, -0.035,
    #  0.065, 0.065, -0.035, -0.035, -0.035, -0.035, -0.035, -0.035, 0.065, -0.035, -0.035, 0.065, -0.035, -0.035, -0.035,
    #  -0.035, -0.035, -0.035, -0.135, -0.235, -0.235, -0.235, -0.235, -0.235, -0.535, -0.135, -0.035]
    # d1,d2=powerRate(l_resid,72 )

    src_csv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_Diesel.csv"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    repository = Path(dir_path) / Path("Repository" )
    with open("../dev/loglog.log", 'w+') as ff:
        totalPowerGainDropTimeEngines(repository=repository)

        # main(len(sys.argv),sys.argv)