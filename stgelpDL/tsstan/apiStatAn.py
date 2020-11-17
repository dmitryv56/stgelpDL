#!/usr/bin/python3

import numpy as np
import pandas as pd
from predictor.utility import msg2log

def aux_deleteRowWithZero(ar:np.array):
    (n,m)=ar.shape
    l_del=[]
    for i in range(n-1,-1,-1):
        if ar[i,m-1]==0:
            l_del.append(i)
        else:
            break

    ar=np.delete(ar,l_del,0)

    return ar

def getRowLongestSequence(ar:np.ndarray)->tuple:
    (n,m)=ar.shape
    max_long=ar[0,1]
    max_long_index=ar[0,0]
    for i in range(n):
        if ar[i,1]>max_long:
            max_long=ar[i,1]
            max_long_index=ar[i,0]
    return (max_long_index, max_long)


def createStatesSequencу(ds:pd.DataFrame, item:str,dt_col_name:str,minValue:float, maxValue:float,enum:tuple,abrupt_type:str='all',
                     abrupt_level:float =0.9, f:object=None)->(list,dict,dict):

    (LOWER_ABRUPT, NO_ABRUPT, UPPER_ABRUPT) = enum
    upper_dict = {}
    lower_dict = {}
    amplitude = maxValue - minValue
    delta = amplitude *(1.0 -abrupt_level) if abrupt_type=='all' else amplitude *(1.0 -abrupt_level)/2.0
    highLevel = maxValue - delta
    lowLevel  = minValue + delta

    msg2log("DelAfterDebug",
            "abrupt level={} max={} min={} amplitude={}  delta={} highLevel={} low={}".format(abrupt_level, maxValue,
             minValue, amplitude, delta, highLevel, lowLevel),f)

    states = []
    for i in range(len(ds[item])):
        stateValue = NO_ABRUPT
        if ds[item].values[i]>=highLevel and abrupt_type!="lower":
            upper_dict[ds[dt_col_name].values[i]]=ds[item].values[i]
            stateValue= UPPER_ABRUPT
        if ds[item].values[i] <= lowLevel and abrupt_type!="upper":
            lower_dict[ds[dt_col_name].values[i]] = ds[item].values[i]
            stateValue = LOWER_ABRUPT

        states.append(stateValue)
    return states, upper_dict,lower_dict

def createAbruptMatrixes(states:list, enum:tuple, abrupt_type:str='all',f:object=None)->(np.ndarray,np.ndarray,np.ndarray):


    (LOWER_ABRUPT, NO_ABRUPT, UPPER_ABRUPT) = enum
    sorted_array, count_array = np.unique(states, return_counts=True)
    msg2log(None, "\nSorted states array: {} Count array: {}\n".format(sorted_array, count_array), f)

    stPrev = states[0]
    indStart = 0
    IND_LOWER_ABRUPT = LOWER_ABRUPT
    IND_NO_ABRUPT    = NO_ABRUPT
    IND_UPPER_ABRUPT = UPPER_ABRUPT

    if abrupt_type=='lower':
        lower_abrupts = np.zeros((count_array[LOWER_ABRUPT], 2), dtype=int)
        no_abrupts    = np.zeros((count_array[NO_ABRUPT],    2), dtype=int)
        upper_abrupts = np.zeros((0,                         0), dtype=int)

    elif abrupt_type=='upper':
        lower_abrupts = np.zeros((0,              0),            dtype=int)
        no_abrupts    = np.zeros((count_array[0], 2),            dtype=int)
        upper_abrupts = np.zeros((count_array[1], 2),            dtype=int)
        # IND_NO_ABRUPT    = IND_NO_ABRUPT    - 1
        # IND_UPPER_ABRUPT = IND_UPPER_ABRUPT - 1
    else:
        lower_abrupts = np.zeros((count_array[LOWER_ABRUPT],  2), dtype=int)
        no_abrupts    = np.zeros((count_array[NO_ABRUPT],     2), dtype=int)
        upper_abrupts = np.zeros((count_array[UPPER_ABRUPT],  2), dtype=int)

    k = np.zeros((UPPER_ABRUPT + 1), dtype=int)

    for i in range(len(states)):
        if states[i] == LOWER_ABRUPT and stPrev == LOWER_ABRUPT and abrupt_type!='upper':
            lower_abrupts[k[IND_LOWER_ABRUPT], 0] = indStart
            lower_abrupts[k[IND_LOWER_ABRUPT], 1] += 1
            stPrev = LOWER_ABRUPT
        elif states[i] == NO_ABRUPT and stPrev == NO_ABRUPT:
            no_abrupts[k[IND_NO_ABRUPT], 0] = indStart
            no_abrupts[k[IND_NO_ABRUPT], 1] += 1
            stPrev = NO_ABRUPT
        elif states[i] == UPPER_ABRUPT and stPrev == UPPER_ABRUPT and abrupt_type!='lower':
            upper_abrupts[k[IND_UPPER_ABRUPT], 0] = indStart
            upper_abrupts[k[IND_UPPER_ABRUPT], 1] += 1
            stPrev = UPPER_ABRUPT
        else:
            k[stPrev] += 1
            indStart = i
            stPrev = states[i]
            if states[i] == LOWER_ABRUPT and abrupt_type!='upper':
                # k[LOWER_ABRUPT] += 1
                lower_abrupts[k[IND_LOWER_ABRUPT], 0] = indStart
                lower_abrupts[k[IND_LOWER_ABRUPT], 1] += 1
                stPrev = LOWER_ABRUPT
            elif states[i] == NO_ABRUPT:
                # k[NO_ABRUPT] += 1
                no_abrupts[k[IND_NO_ABRUPT], 0] = indStart
                no_abrupts[k[IND_NO_ABRUPT], 1] += 1
                stPrev = NO_ABRUPT
            elif states[i] == UPPER_ABRUPT and abrupt_type!='lower':
                # k[UPPER_ABRUPT] += 1
                upper_abrupts[k[IND_UPPER_ABRUPT], 0] = indStart
                upper_abrupts[k[IND_UPPER_ABRUPT], 1] += 1
                stPrev = UPPER_ABRUPT
            else:
                pass

    if abrupt_type!='upper': lower_abrupts =aux_deleteRowWithZero(lower_abrupts)
    no_abrupts = aux_deleteRowWithZero(no_abrupts)
    if abrupt_type!='lower': upper_abrupts = aux_deleteRowWithZero(upper_abrupts)

    return upper_abrupts, no_abrupts, lower_abrupts

def getAbruptChanges(ds:pd.DataFrame, item:str,dt_col_name:str,minValue:float, maxValue:float, enum:tuple,
                     abrupt_type:str='all',abrupt_level:float =0.9, f:object=None)->(list, dict,dict, tuple,tuple,tuple):

    (LOWER_ABRUPT, NO_ABRUPT,UPPER_ABRUPT)=enum
    states,upper_dict,lower_dict =  createStatesSequencу(ds, item, dt_col_name, minValue, maxValue, enum,abrupt_type,
                                                         abrupt_level, f)

    upper_abrupts, no_abrupts, lower_abrupts = createAbruptMatrixes(states, enum, abrupt_type, f)


    res_abrupt_dict={}
    if abrupt_type!='upper': res_abrupt_dict["LOWER_ABRUPT"]  = lower_dict
    if abrupt_type!='lower': res_abrupt_dict["UPPER_ABRUPT"] = upper_dict
    period_abrupt_dict={}
    if abrupt_type!='upper':period_abrupt_dict["LOWER_ABRUPT"] = lower_abrupts
    period_abrupt_dict["NO_ABRUPT"]    = no_abrupts
    if abrupt_type!='lower': period_abrupt_dict["UPPER_ABRUPT"] = upper_abrupts

    tuple_lower=getRowLongestSequence(period_abrupt_dict["LOWER_ABRUPT"]) if abrupt_type!='upper' else ('--','--')
    tuple_no_abrupt = getRowLongestSequence(period_abrupt_dict["NO_ABRUPT"])
    tuple_upper = getRowLongestSequence(period_abrupt_dict["UPPER_ABRUPT"]) if abrupt_type!='lower' else ('--','--')

    return states, res_abrupt_dict, period_abrupt_dict, tuple_lower, tuple_no_abrupt, tuple_upper

if __name__ == "__main__":
    pass