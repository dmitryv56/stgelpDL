#!/usr/bin/python3

""" 'func' module contains an api for an frequency (fr) analysis (an) of an packet appearances (pa)
at the transport level (tl) of CANBus-FD program ('franpatl')  and an intrusion detection system (ids) based on packet
classification (paccl) program ('idspaccl').
"""

from os import path
import sys
import  copy
from datetime import datetime,timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson,ks_2samp

from simTM.auxapi import dictIterate,listIterate
from predictor.utility import msg2log
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from canbus.api import readChunkFromCanBusDump,file_len,dict2csv,dict2pkl, pkl2dict,mleexp,manageDB,DB_NAME,fillQuery,\
fillQuery2Test,getSerializedModelData,KL_decision
from canbus.digitaltwin import DigitalTwinMLPids,chart_loss, bin_confusion

DB_DSET_KEY='ID'
DB_DSET_VAL='dataset'
DB_DL_KEY ='ID'
DB_DL_VAL='DLmodel'



""" parse unix timestamp. It is value of 'DateTime' key"""
def parseUnixTimestamp(ftstamp:float=None,tzone:str="local")->str:

    if ftstamp is None:
        utc_time=datetime.time()
    else:
        utc_time = datetime.fromtimestamp(ftstamp, timezone.utc)
    if tzone=="utc":
        stime = utc_time.strftime("%Y-%m-%d %H:%M:%S.%f+00:00 (UTC)")
    elif tzone=="local":
        local_time = utc_time.astimezone()
        stime = local_time.strftime("%Y-%m-%d %H:%M:%S.%f%z (%Z)")
    else:
        stime = utc_time.strftime("%Y-%m-%d %H:%M:%S.%f+00:00 (UTC)")

    return stime

""" Dictionary replenisment with taken packet data.-
The source_dict comprises parsed data from arrived packet 
{'DateTime':itemDateTime,
 'IF':itemIF, 
 'ID':itemID,
 'Data':itemData,
 'Packet':itemPacket,  
 'bitstr_list':bitstr_list,
 'bit_str':bit_str}
they have been extracted from the dump file line (see parseCanBusLine() -function).
For arrived packet its 'match_key' is checked in {match_key:[time from start in microsec]} dictionary. If match, time 
delta is added tothe list of values. If no match, new pair {key:[time delta] is created.
 The length of value list is increased (set to 1 for new key) in 'len_dict' dictionary.
 """
def refillDict(target_dict:dict=None,key_list:list=[], len_dict:dict= {}, source_dict:dict= None,match_key:str='ID',
               start_timestamp:float=0.0,f:object=None):

    delta=round((float(source_dict['DateTime'])-start_timestamp)*1000000.0,0) #microsecond
    sKey=source_dict[match_key]
    if sKey in key_list:
        target_dict[sKey].append(delta)
        len_dict[sKey]=len_dict[sKey]+1
    else:
        target_dict[sKey]=[delta]
        key_list.append(sKey)
        len_dict[sKey]=1

    return

""" Read ICsimdump and fill a target_dict with pairs {<match_key>: [ packet arrival times ].
'match_key' is one of fields 'ID,'Data' or 'Packet'. The packet arrival time is a delta between first packet od dump
fille arrival time and currrent packet arrival time (microSec).
On base this list the probability model of key appearing is estimated.
"""
def gendict(canbusdump:str="",target_dict:dict=None,match_key:str='ID', f:object=None):
    """

    :param canbusdump: -ICsim dump file
    :param target_dict: {} at input, filled target dictionary at output.
    :param match_key: key , one of {'ID','Packet','Data'} fields. These fields belong to the parsed line of the dump.
    :param f: log file handler
    :return:
    """
    offset_line=0
    chunk_size=128

    key_list=list(target_dict.keys())
    len_dict={}
    difftstamp=[]
    ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=1, canbusdump=canbusdump, f=f)
    start_ftstamp = float(ld[0]['DateTime'])
    start_time = parseUnixTimestamp(ftstamp= start_ftstamp)
    prev_ftstamp=round(start_ftstamp*1000000.0,7)
    difftstamp.append(0.0)
    offset_line+=1
    while ld:
        for item in ld:
            current_ftstamp = round(float(item['DateTime']) * 1000000.0, 7)
            difftstamp.append(round(current_ftstamp-prev_ftstamp,7))
            refillDict(target_dict=target_dict, key_list=key_list,len_dict=len_dict, source_dict=item,
                       match_key=match_key, start_timestamp=start_ftstamp, f=f)
            prev_ftstamp=current_ftstamp
        finish_ftstamp = float(ld[-1]['DateTime'])

        ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=chunk_size, canbusdump=canbusdump, f=f)
        offset_line=offset_line+chunk_size
    range_ftstamp=round(finish_ftstamp-start_ftstamp,6)
    finish_time = parseUnixTimestamp(ftstamp=finish_ftstamp)
    len_list=list(len_dict.values())
    minlen=min(len_list)
    maxlen=max(len_list)
    message=f""" 
    Start time (Epoch time) : {start_ftstamp} sec     Start timestamp:  {start_time}
    Finish time (Epoch time): {finish_ftstamp} sec    Finish timestamp: {finish_time}
    Time range              : {range_ftstamp} sec 
    Acquisited pairs   {match_key}:[]     : {len(target_dict)}
    Rarest : {minlen}  Frequent: {maxlen}
"""
    msg2log(None,message,f)
    return difftstamp

""" Create DB (dataset) should be use for anomaly detection at test path.
For each pair <match_key>:[timestamps for packets with matched key] performed a folowing features estimation:
- probability = len([timestamps])/number_samples_on_train_path.
- list of duration intervals between timestamps.
- mean and std of duration intervals if len([timestamps])>2
- histogram of duration intervals if len([timestamps])>20
<match_key>,<probaility>,<mean>,<std>,<hist>,<rdge bins for histogram> is saved as pandas.DataFrame to csv-file.
The <target_dict> is saved as pandas DataFrame to csv-dataset and deleted. 
"""
def createDataset(target_dict:dict=None,n_samples:int=1, match_key:str='ID', title:str=None, folder:str="",
                  dset_name:str=None, f:object=None):

    if dset_name is None or len(dset_name)<1:
        msg2log(None, "dataset name is not set correctly {}".format(dset_name), f=f)
        return

    """ <match_key>,<est.probability>,<[bins]>,<[hist]>"""
    df_dict={}
    m_key=[]
    prob=[]
    bins=[]
    mean=[]
    std=[]
    hist=[]
    for key,val in target_dict.items():
        m_key.append(key)
        prob.append(round(float(len(val)/n_samples), 5))
        if len(val)<=2:
            mean.append([])
            std.append([])
            bins.append([])
            hist.append([])
            continue
        interval=np.array([val[i+1]-val[i] for i in range(len(val)-1)])
        ave=round(np.mean(interval),2)
        sig =round(np.std(interval),2)
        if len(val)>20:
            h,b=np.histogram(interval)
            hist.append(h.tolist())
            bins.append(b.tolist())
        else:
            hist.append([])
            bins.append([])
        mean.append(ave)
        std.append(sig)
    #
    # dict2csv(d = target_dict, folder= folder, title= "Target_dict", match_key=match_key, f=f)
    del target_dict
    target_dict=None
    df_dict={match_key:m_key,'prob':prob, 'mean':mean, 'std':std,'hist':hist,'bins':bins}
    dict2csv(d = df_dict, folder= folder, title= title, dset_name=dset_name,match_key=match_key, f=f)
    del df_dict

    return





poisson_prob = lambda n,poisson_lambda: poisson.pmf(n,poisson_lambda)

""" Find anomaly algorithm comprises following steps:
1) create DataFrame object from repository dataset. The file name is <repository>/train_<match_key>.csv
2) read ICsim test dump file,parse lines, create test_dictionary {match_key:[timestamps]
3) transform DataFrame to dict {match_key:{'prob':<value>,'mean':<list[value]>,'std':<list[value]>,'hist':<list>,
'bins':<list>}}. The rows with empty 'hist' in DataFrame are ignored.
4) intersection dictionary via their sets
"""
def findAnomaly(canbusdump:str="",match_key:str='ID', dset_name:str=None, repository:str="", f:object=None)->list:
    anomaly_match_key=[]
    pass
    """ Repository dataset , DataFrame and dict """
    ds_name=Path(Path(repository)/Path("train_{}".format(match_key))).with_suffix(".csv")
    ds_name = Path(dset_name)
    if not ds_name.exists():
        msg="{} not found in Repository, the test stage finished.".format(str(ds_name))
        log2All(msg)
        return
    df=pd.read_csv(str(ds_name))
    train_dict = df2dict(df = df, key=match_key, f=f)
    """ ICsim dump"""
    test_dict,l_tstamps = genTestDict(canbusdump= canbusdump, match_key=match_key, f=f)
    n_test=len(l_tstamps) +1
    if test_dict is None  or len(test_dict)==0:
        msg = "test dictory can not build, the test stage finished.".format(str(ds_name))
        log2All(msg)
        return
    """ dict intersection """
    train_dictSet=set(train_dict)
    test_dictSet=set(test_dict)


    msg = "{:>4s} {:^22s} {:^10s} {:^10s} {:^5s} {:^12s} {:^12s}".format("NN", "Matched Key", "TrainProb", "lambda", "n",
                                                                 "P(n,lambda)", "TestProb")

    msg2log(None, msg, D_LOGS['predict'])
    ikey=0
    for key in train_dictSet.intersection(test_dictSet):
        key_prb=train_dict[key]['prob']
        poisson_lambda=key_prb*n_test
        n_appears=len(test_dict[key])
        p=[]
        for n in range(0,n_appears+1):
            p.append(poisson_prob(n,poisson_lambda))
        png_name=str(Path(Path(D_LOGS['plot'])/key).with_suffix(".png"))
        poissonPlot(p=p,title=match_key,png_name=png_name,f=f)

        msg="{:>4d} {:<22s} {:<10.4f} {:<10.4f} {:>5d} {:<10.4f} {:<10.4f}".format(ikey,key,key_prb,poisson_lambda, n_appears,
                                    poisson_prob(n,poisson_lambda),float(n_appears/n_test))
        msg2log(None,msg,D_LOGS['predict'])
        D,ks_stat,prb = histCompare(train_dict[key], test_dict[key], f=f)
        if prb*100 <5.0 :
            msg=f"""Warning! {key} : D={D},ks_stat={ks_stat}, prb={prb*100} <5% , 
            The train and test distributions may be differs!"""
            msg2log(None,msg,D_LOGS['control'] )
            anomaly_match_key.append(key)
        ikey+=1

    return anomaly_match_key

def histCompare(train_dict_key:dict,test_dict_key:list, f:object=None)->(float,float,float):
    n1 = len(train_dict_key['hist'])
    if n1<10:
        return 0.0,0.0,0.6
    s1 = sum([train_dict_key['hist'][i] for i in range(len(train_dict_key['hist']))])
    hist,bin = np.histogram(np.array(test_dict_key))
    n2=len(hist)
    if n2<10:
        return 0.0, 0.0, 0.6

    n=min(n1,n2)
    s2=sum([hist[i] for i in range(len(hist))])
    D=max([abs( float(train_dict_key['hist'][i]/s1)-float(hist[i]/s2)) for i in range( n)])
    ks_stat,prb = ks_2samp(np.array(train_dict_key['hist'])/s1,np.array(hist)/s2)
    return D,ks_stat,prb




def poissonPlot(p:list=[],title:str="",png_name:str="",f:object=None):
    if len(p)<20:
        return
    plt.plot([i for i in range(len(p))], p, color='red',label='Poisson prb.')
    plt.legend()
    plt.savefig(png_name)
    plt.close("all")
    return



""" Read ICsimdump and fill a test_dict with pairs {<match_key>: [timestamps ].
Where match_key is one of fields 'ID,'Data' or 'Packet'. The list ot timestamps is filled by timestamps when packet 
matched with  'match_key' key appeared in the dump.
On base this list the probability model of key appearing is estimated.
"""
def genTestDict(canbusdump:str="",match_key:str='ID', f:object=None)->(dict,list):

    offset_line=0
    chunk_size=128
    test_dict={}

    key_list=list(test_dict.keys())
    len_dict={}
    difftstamp=[]
    ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=1, canbusdump=canbusdump, f=f)
    start_ftstamp = float(ld[0]['DateTime'])
    start_time = parseUnixTimestamp(ftstamp= start_ftstamp)
    prev_ftstamp=round(start_ftstamp*1000000.0,7)
    difftstamp.append(0.0)
    offset_line+=1
    while ld:
        for item in ld:
            current_ftstamp = round(float(item['DateTime']) * 1000000.0, 7)
            difftstamp.append(round(current_ftstamp-prev_ftstamp,7))
            refillDict(target_dict=test_dict, key_list=key_list,len_dict=len_dict, source_dict=item,
                       match_key=match_key, start_timestamp=start_ftstamp, f=f)
            prev_ftstamp=current_ftstamp
        finish_ftstamp = float(ld[-1]['DateTime'])

        ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=chunk_size, canbusdump=canbusdump, f=f)
        offset_line=offset_line+chunk_size
    range_ftstamp=round(finish_ftstamp-start_ftstamp,6)
    finish_time = parseUnixTimestamp(ftstamp=finish_ftstamp)
    len_list=list(len_dict.values())
    minlen=min(len_list)
    maxlen=max(len_list)
    message=f""" 
    Start time (Epoch time) : {start_ftstamp} sec     Start timestamp:  {start_time}
    Finish time (Epoch time): {finish_ftstamp} sec    Finish timestamp: {finish_time}
    Time range              : {range_ftstamp} sec 
    Acquisited pairs   {match_key}:[]     : {len(test_dict)}
    Rarest : {minlen}  Frequent: {maxlen}
"""
    msg2log(None,message,f)
    return test_dict,difftstamp

""" Df (DataFrame object) to dict"""
def df2dict(df:pd.DataFrame=None,key:str='ID',f:object=None):
    l_features=list(df.columns)
    if key not in l_features:
        return

    df1 = df.drop(l_features[0], axis=1)
    l_features.pop(0) # remove Unnamed: 0
    d_target={}
    for i in range(len(df1)):
        a_d={}
        lv=[df1[l_features[j]][i] for j in range(1,len(l_features))]
        lvv = [item.strip('][').split(', ') if type(item) == str else item for item in lv]
        a_d = {l_features[j]: lvv[j - 1] for j in range(1, len(l_features))}
        # analysis {'prob': 1e-05, 'mean': [''], 'std': [''], 'hist': [''], 'bins': ['']}
        if len(a_d['hist'])==1 and len(a_d['hist'][0])==0:
            # empty 'hist','bins; lists
            pass
        """
lv1=[df1[lcols[j]][1] for j in range(1,len(lcols))]
lv1
[0.01267, '40241.23', '2206.22', '[1518, 0, 0, 0, 0, 0, 0, 0, 0, 19]', '[38994.0, 41137.7, 43281.4, 45425.1, 47568.8, 
49712.5, 51856.2, 53999.899999999994, 56143.6, 58287.3, 60431.0]']
lv11=[item.strip('][').split(', ') if type(item)==str else item for item in lv1]
lv11
[0.01267, ['40241.23'], ['2206.22'], ['1518', '0', '0', '0', '0', '0', '0', '0', '0', '19'], ['38994.0', '41137.7', 
'43281.4', '45425.1', '47568.8', '49712.5', '51856.2', '53999.899999999994', '56143.6', '58287.3', '60431.0']]
"""

        for item in l_features[2:]:
            try:
                a_d[item]=list(map(float,a_d[item]))
            except:
                a_d[item]=[]
        lvint=[]
        kv=df1[l_features[0]][i]
        d_target[kv]=a_d

    #
    return d_target





@exec_time
def testFreqModel(canbusdump:str= None, match_key:str='ID', repository:str=None, dset_name:str=None, title:str="",
                  f:object=None):
    pass
    """ check canbus dump """
    if not Path(canbusdump).exists():
        return None
    """ check database (dataset)"""
    if dset_name is None or not Path(dset_name).exists():
        return None
    """ check data base they should comprice tarined model"""
    # find pkl serialized file
    d_query = fillQuery2Test(match_key=match_key, method="MLE",  repository= repository, f=f)

    d_res = manageDB(repository=repository, db=DB_NAME, op='select', d_query=d_query, f=f)

    train_mleexp_dict= getSerializedModelData(d= d_res, f=f)
    if train_mleexp_dict is None or len(train_mleexp_dict)==0:
        msg2log(None,"Train data was not found. The test satge termibaned.", f=f)
        return None

    """ read, parse dump and create dictionary"""
    target_dict = {}
    mleexp_dict ={}
    n_min = 5
    difftstamp = gendict(canbusdump=canbusdump, target_dict=target_dict, match_key=match_key, f=f)
    n_samples = len(difftstamp) + 1
    print(len(target_dict))
    msg2log(None, (difftstamp), f)
    msg2log(None, (len(target_dict)), f)
    """ check packet appearing probability based on the poisson process """
    pass
    list_anomaly0= findAnomaly(canbusdump= canbusdump, match_key=match_key, repository= repository, dset_name=dset_name,
                               f=f)

    mleexp(target_dict=target_dict, mleexp_dict=mleexp_dict, n_min=n_min, title=title, f=f)

    list_anomaly1 = KL_decision(train_mleexp = train_mleexp_dict, test_mleexp = mleexp_dict,
                               title = "Anomaly packet checked by Exponential model", f=f)

    printPossibleAnomalies(list_anomaly0, list_anomaly1, f=f)
    return

def printPossibleAnomalies(anomaly1,anomaly2,f:object=None):
    pass
    if anomaly1 is None:
        msg2log(None,"KS anomaly is not set",f)
        return
    if anomaly2 is None:
        msg2log(None,"MLE anomaly is not set",f)
        return
    possible_anomalies=list(set.intersection(set(anomaly1),
                                             set([anomaly2[i]['matched_key'] for i in range(len(anomaly2))])))

    msg="List of possible anomalies detected by KS-statistic amd KL-divergence for exponential distributions"
    msg2log(None,msg,D_LOGS['main'])
    for item in possible_anomalies:
        msg2log(None,item,D_LOGS['main'])



@exec_time
def trainFreqModel(canbusdump:str= None, match_key:str='ID', repository:str=None, dset_name:str=None, title:str="",
                   f:object=None):
    if canbusdump is None or canbusdump=="" or not Path(canbusdump).exists():
        msg="canbus dump log {} does not exist".format("" if canbusdump is None else canbusdump)
        msg2log(None,msg,D_LOGS['control'])
        return
    if repository is None :
        msg = "Repository folder  does not set"
        msg2log(None, msg, D_LOGS['control'])
        return
    target_dict = {}
    mleexp_dict = {}
    n_min=5
    difftstamp = gendict(canbusdump=canbusdump, target_dict=target_dict, match_key=match_key, f=D_LOGS['control'])
    n_samples = len(difftstamp) + 1
    print(len(target_dict))
    msg2log(None, (difftstamp), f)
    msg2log(None, (len(target_dict)), f)

    createDataset(target_dict=target_dict, n_samples=n_samples, match_key=match_key, title=title, folder=repository,
                  dset_name=dset_name, f=f)
    mleexp(target_dict= target_dict, mleexp_dict=mleexp_dict, n_min= n_min, title= title, f=f)

    pkl_stem,pathPkl = dict2pkl(d= mleexp_dict, folder= repository, title= title, match_key= match_key, f=f)
    # add train to DB
    d_query = fillQuery( dump_log= canbusdump, match_key= match_key, method= "MLE", pkl=pkl_stem,
                         repository= repository,  misc= "", f=f)
    manageDB(repository = repository, db=DB_NAME, op = 'insert', d_query= d_query, f=f)

    return pkl_stem,pathPkl


""" IDS packet classification.


The path to DB of datasets is <dset_name>/<feature_type>/<title>.csv.
"""
@exec_time
def dsetIDSpacCl(l_canbusdump:list= [],feature_type:str = 'one', repository:str=None, chunk_size:int=128,
                  dset_name:str=None, title:str="", f:object=None)->dict:
    if l_canbusdump is None or l_canbusdump=="" : #or not Path(canbusdump).exists():
        msg="list of canbus dump logs is empty!"
        msg2log(None,msg,D_LOGS['control'])
        return
    if repository is None or dset_name is None:
        msg = "Data or Model  Repository folder  are not set"
        msg2log(None, msg, D_LOGS['control'])
        return
    train_target_dict={}
    for canbusdump in l_canbusdump:

        if canbusdump is None or canbusdump == "" or not Path(canbusdump).exists():
            msg = "canbus dump log {} does not exist".format("" if canbusdump is None else canbusdump)
            msg2log(None, msg, D_LOGS['control'])
            continue
        mleexp_dict = {}
        n_min=5

        difftstamp = featureExtract(canbusdump=canbusdump, train_target_dict=train_target_dict,
                                    feature_type = feature_type, chunk_size=chunk_size, f=f)
        n_samples = len(difftstamp) + 1

        msg2log(None, "Diffstamps\n{}".format(difftstamp), D_LOGS['block'])
        msg2log(None, "Number pairs in train_target dictionary : {}".format(len(train_target_dict)), f)


    dset_dict = genSLD(train_target_dict=train_target_dict,dset_name=dset_name, feature_type=feature_type, f=f)
    csv_file=Path(Path(dset_name)/Path(feature_type)/Path(title)).with_suffix(".csv")

    df=pd.DataFrame(dset_dict)
    df.to_csv(str(csv_file),index=False)
    log2All("SLD datasets saved in {}".format(str(csv_file,)))
    log2All()
    return dset_dict


""" Read ICSimdump and fill a target_dict with pairs {<ID>: [ feature list ].
.....
"""
def featureExtract(canbusdump:str="",train_target_dict:dict=None,feature_type:str = 'one', chunk_size:int=128,
                   f:object=None):
    offset_line=0
    # key_list=list(target_dict.keys())
    len_dict={}
    difftstamp=[]
    """ Read  and parse  a chunk of dump file. The list of dict  is returned. The dictionary pairs are following:
    
    'DateTime': itemDateTime, 
    'IF': itemIF, 
    'ID': itemID, 
    'Data': itemData, 
    'Packet': itemPacket
    'bitstr_list': bitstr_list, 
    'bit_str': bit_str
        
    """
    ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=chunk_size, canbusdump=canbusdump, f=f)
    start_ftstamp = float(ld[0]['DateTime'])
    start_time = parseUnixTimestamp(ftstamp= start_ftstamp)
    prev_ftstamp=round(start_ftstamp*1000000.0,7)
    difftstamp.append(0.0)
    offset_line+=chunk_size
    while ld and len(ld)==chunk_size: # last short chunk is ignored
        key_cntr_dict={}
        target_dict = {}
        for item in ld:
            current_ftstamp = round(float(item['DateTime']) * 1000000.0, 7)
            difftstamp.append(round(current_ftstamp-prev_ftstamp,7))

            key,val = fillfeatureItem(target_dict=target_dict,source_dict=item,feature_type = feature_type,f=f)
            if key not in train_target_dict:
                train_target_dict[key] = []
            if key not in key_cntr_dict:
                key_cntr_dict[key]=1
            else:
                key_cntr_dict[key] = key_cntr_dict[key] +1


            prev_ftstamp=current_ftstamp

        #norm feature
        setFeatureValues(feature_type= feature_type, target_dict = target_dict, key_cntr_dict= key_cntr_dict,
                         chunk_start = offset_line-chunk_size,   chunk_size = chunk_size, f=f)

        finish_ftstamp = float(ld[-1]['DateTime'])
        for key,val in target_dict.items():
            val.append(0) # add desired data : 0 -for no intrused packet
            train_target_dict[key].append(val)  # {<ID>:[<list of ones per bit>,<desired data>]}


        """ read and parse next chunk."""
        ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=chunk_size, canbusdump=canbusdump, f=f)
        offset_line=offset_line+chunk_size
    range_ftstamp=round(finish_ftstamp-start_ftstamp,6)
    finish_time = parseUnixTimestamp(ftstamp=finish_ftstamp)


    return difftstamp

def setFeatureValues(feature_type:str='one', target_dict:dict=None, key_cntr_dict:dict=None, chunk_start:int=0,
                     chunk_size:int=256, f:object=None):

    if target_dict is None or key_cntr_dict is None:
        msg2log(setFeatureValues.__name__, " Value dict or cntr dict missed. The feature values anot set",f)
        return

    if feature_type=="one":
        return
    elif feature_type=="prb" or feature_type=='0-1':
        sTarget=set(target_dict)
        sCntr=set(key_cntr_dict)
        for key in sCntr.intersection(sTarget):
            divisor=key_cntr_dict[key]
            val=target_dict[key]
            for i in range(len(val)):
                val[i]=val[i]/divisor
                if feature_type=='0-1': val[i]=1.0 if val[i]>=0.5 else 0.0

    elif feature_type=="other":
        pass #TODO
    else:
        pass
    return
def fillfeatureItem(target_dict:dict=None, source_dict:dict=None, feature_type:str='one', start_timestamp:float=0.0,
                    f:object=None)->(str,list):
    if target_dict is None or source_dict is None:
        msg2log(fillfeatureItem.__name__,"Invalid dicts(target or/and souce) are passed. ",f)
    # Note!!!! It is possible that packets with same 'Id' have varying size of Data? Maybe. So the size value array is 64.
    key=source_dict['ID']
    if key not in target_dict:
        target_dict[key]=[0 for i in range(64)]

    if feature_type=='one' or feature_type=='prb' or feature_type=='0-1':
        for i in range(len(source_dict['Data'])):
            l_res=featOne(source_dict['Data'][i])
            start_ind=i*4
            for j in range(start_ind,start_ind+4):
                target_dict[key][j]=target_dict[key][j] + l_res[j-start_ind]
    else:
        pass

    return key, target_dict[key]


""" Create Supervised Learning Data(SLD) for each 'key' in 'train_target_dict'.
Save the SLD in csv-dataset.
Clear the content of 'train_target_dict'.
Return the dictionary {'key':'csv-dataset'}.

'train_target_dict' contains the pairs {key:<feature-matrix>}, where key is Canbus packet ID field content,
<feature-matrix> is list of feature vectors(lists).
Each feature vector contains the accumulated feature values taken from Data-field  CANbus packets for given ID across 
single chunk of the ICSim dump log. The accumulation algorith implemented in fillfeatureItem() function.
Each feature vector is randomly multiplied with given coefficient to create new feature vectors with desired data '0' 
for valid and '1' for  intrusion states. The binomial generator with p=0.1 for intrusion is used.
Below the SLD matrix
[ [v00 v01 ... v0m  y0]
  [v10 v11 ... v1m  y1]
  ............
  [vN0 vN1 ... vNm  yN] ]
  where vij j-th value of i-th feature vector,
  yi- i-th value of desired data (0 or 1)

The path to dataset repository is <dset_name>/<feature_type> or <repository>/data/<feature_type>.
The path to SLD dataset is <dset_name>/<feature_type>/ID_<ID>.csv. 
"""
def genSLD(train_target_dict:dict=None, mult_koef:int=8, p_intrusion:float=0.1, dset_name:str=None,
           feature_type:str='one',f:object=None )->dict:


    rng=np.random.default_rng()
    l_ID=[]
    l_dset=[]
    for key,val in train_target_dict.items():
        n=len(val) # number rows
        msg2log(genSLD.__name__,"Key :{} {} rows was read from dict".format(key,n), D_LOGS['block'])
        log2All()
        l_ID.append(key)
        for i in range(n):
            mean_list=val[i][:-1]
            # drawn N binomial samples for 2 states :0 -not inrused, 1 intrused, where p(not intrused)=0.95
            N=mult_koef
            p=p_intrusion
            samples=rng.binomial(1,p,N)
            for s in samples:

                if s==0:  #correct packet
                    val1=copy.copy(val)

                    if False: # do not random correct packet data
                        for k in range(len(val1)):
                            if val1[k]==0:
                                continue
                            else:
                                if feature_type=='prb':
                                    r=round(rng.uniform(0,1,1)[0],3)
                                    c = val1[k] + r
                                    val1[k] = c if c <1.0 else val1[k]
                                else:
                                    r=round(rng.uniform(-5,5,1)[0],3)
                                    c=val1[k]+int(r*val1[k])
                                    val1[k]=c if c>0 else val1[k]
                else:  # intrused paket
                    val1 = copy.copy(mean_list)
                    for k in range(len(val1)):
                        if val1[k]==0:
                            if feature_type=='prb':
                                val1[k]=round(rng.uniform(0,1,1)[0],3)
                            else:
                                r = round(rng.uniform(0, 32, 1)[0],3)
                                val1[k]=int(r)
                        else:
                            if feature_type == 'prb':
                                val1[k] = 0.0
                            else:
                                r = round(rng.uniform(-5, 5, 1)[0],3)
                                c = val1[k] + int(r * val1[k])
                                val1[k] = c if c > 0 else val1[k]

                val1.append(s)
                val.append(val1)
        a=np.array(val)
        (n_row,n_col)=a.shape
        index=[str(i) for i in range(n_row)]
        temp_d={'bit_{}'.format(col):a[:,col] for col in range(n_col-1)}
        temp_d['y']=a[:,n_col-1]
        dataset=pd.DataFrame(temp_d,index=index)
        path_to=Path(Path(dset_name)/Path("{}".format(feature_type)))
        if not path_to.exists():
            path_to.mkdir(parents=True, exist_ok=True)
        file_csv=path_to/Path("ID_{}".format(key)).with_suffix(".csv")
        dataset.to_csv(file_csv)
        l_dset.append(file_csv)

        del val
        val=None
        del a
        a=None
    pass
    dset_dict={DB_DSET_KEY:l_ID,DB_DSET_VAL:l_dset}
    del train_target_dict
    train_target_dict={}
    return dset_dict


def featOne(nibble:str)->list:
    lre = [0,0,0,0]
    if nibble=='0' or nibble==' ' or nibble=='': lre = [0,0,0,0]
    elif nibble == '1': lre = [0,0,0,1]
    elif nibble == '2': lre = [0, 0, 1, 0]
    elif nibble == '3': lre = [0, 0, 1, 1]
    elif nibble == '4': lre = [0, 1, 0, 0]
    elif nibble == '5': lre = [0, 1, 0, 1]
    elif nibble == '6': lre = [0, 1, 1, 0]
    elif nibble == '7': lre = [0, 1, 1, 1]
    elif nibble == '8': lre = [1, 0, 0, 0]
    elif nibble == '9': lre = [1, 0, 0, 1]
    elif nibble == 'A' or nibble == 'a': lre = [1, 0, 1, 0]
    elif nibble == 'B' or nibble == 'b': lre = [1, 0, 1, 1]
    elif nibble == 'C' or nibble == 'c': lre = [1, 1, 0, 0]
    elif nibble == 'D' or nibble == 'd': lre = [1, 1, 0, 1]
    elif nibble == 'E' or nibble == 'e': lre = [1, 1, 1, 0]
    elif nibble == 'F' or nibble == 'f': lre = [1, 1, 1, 1]
    else:
        lre = [0,0,0,0]
    return lre

""" A path to DB of the models is <repository>/<feature_type>/<title>.csv.
"""
@exec_time
def trainIDSpacCl(feature_type:str = 'one', repository:str=None, chunk_size:int=128,exclude_ID:list=[],
                  dset_name:str=None, title:str="", f:object=None)->dict:
    pass
    input_size=64
    n_classes=2
    hyperprm = {
            'normalize':False,
            'batch_size':64,
            'epochs':10,
            'input_size': {input_size: None},
            'n_classes': {n_classes: 'softmax'},
            'hidden_layers': [{128: 'relu'}, {32: 'relu'},{16:'sigmoid'}]
        }
    if repository is None or dset_name is None:
        log2All("Data and /or Model Repositor arenot set.")
        return None
    DB_dset_name=Path(Path(dset_name)/Path(feature_type)/Path(title)).with_suffix(".csv")
    if not DB_dset_name.exists():
        log2All("Dataset DB {} is not found.".format(str(DB_dset_name)))
        return None

    df=pd.read_csv(str(DB_dset_name))
    dset_dict={DB_DSET_KEY:df[DB_DSET_KEY].values.tolist(), DB_DSET_VAL:df[DB_DSET_VAL].values.tolist()}
    msg = dictIterate(ddict=dset_dict)
    msg2log(None,msg,f)
    del df
    df=None
    l_ID=dset_dict[DB_DSET_KEY]
    l_file = dset_dict[DB_DSET_VAL]
    m_ID=[]
    m_file=[]
    for key, val in zip(l_ID,l_file):
        if key in exclude_ID:
            msg2log(None,"For ID={} DL model is not trained".format(key))
            continue
        msg=""
        try:
            df=pd.read_csv(val)
        except:
            pass
            msg = "\nOoops! Unexpected error: {}\n{}\nFor ID={} DL model is not trained".format(sys.exc_info()[0],
                    sys.exc_info()[1],key)
        finally:
            if len(msg)>0:
                msg2log(trainIDSpacCl.__name__,msg,D_LOGS['except'])

        if len(msg)>0:
            continue

        columns=list(df.columns)
        x=df[columns[0]].values
        (n,)=x.shape
        m=len(columns)-2
        X=np.zeros((n,m),dtype=float)
        j=0
        for item in columns[1:-1]:  # eliminate 'Unnamed 0' - index and 'y'-desired data
            x=df[item].values
            X[:,j]=x[:]
            j+=1

        y=df[columns[-1]].values
        model_saving_path =digitalTwinTrainClM(model = DigitalTwinMLPids, name = "DigitalTwinMLPids", X=X, y= y,
                            key=key,feature_type=feature_type, hyperprm = hyperprm, repository = repository, f=f)
        log2All()
        del X
        X=None
        if model_saving_path is not None:
            m_ID.append(key)
            m_file.append(model_saving_path)

    d_saved_model={DB_DL_KEY:m_ID,DB_DL_VAL:m_file}
    df=pd.DataFrame(data=d_saved_model)
    csv_file = Path(Path(repository) / Path(feature_type)/Path(title)).with_suffix(".csv")

    df.to_csv(str(csv_file), index=False)
    log2All("Deep Learning models saved in {}".format(str(csv_file )))
    log2All()

    return d_saved_model
"""
The path to folder where trained deep learning models are saved is <repository>/<feature_type>/<key>
"""
def digitalTwinTrainClM(model: object = DigitalTwinMLPids, name: str = "DigitalTwinMLP", X:np.array=None,
                        y:np.array=None, key:str="",hyperprm: dict = None, repository: str = "",
                        feature_type:str="", f: object = None)->str:
    pass
    digital_twin = model(name="DigitalTwin", model_repository=repository, f=f)
    # hyperprm= {
    #     input:{input_size:None},
    #     output:{n_classes:None},
    #     hidden_layers:[{32:'relu'},{32:'relu'}]
    #           }

    """ parse hyper parameters"""
    d_input_size = hyperprm['input_size']
    (input_size, _), = d_input_size.items()
    n_classes = hyperprm['n_classes']
    hidden_layers = hyperprm['hidden_layers']
    normalize = None
    # if hyperprm['normalize']:
    #     x, y, normalize = readChunk(filename=chunk_list[0], norm=True, f=f)

    digital_twin.setModel(input_size, normalize, n_classes, hidden_layers)
    digital_twin.compile()
    log2All()

    (n,m)=X.shape
    N=int(n*0.9)

    """ fit process """
    history = digital_twin.fit(X[0:N,:], y[0:N], batch_size=hyperprm['batch_size'], epochs=hyperprm['epochs'])
    chart_loss(digital_twin.model.name, 0, history)
    log2All()

    digital_twin.evaluate(X[N:,:], y[N:])
    path_to=Path(digital_twin.model_repository) / Path(feature_type)
    if not  path_to.exists():
        path_to.mkdir(parents=True, exist_ok=True)

    model_saving_path = path_to / Path(key)
    model_saving_path.mkdir(parents=True, exist_ok=True)
    model_saving_path =digital_twin.saveModel(key=key,model_saving_path=str(model_saving_path))
    log2All()

    return model_saving_path


@exec_time
def detectIDSpacCl(feature_type: str = 'one', repository: str = None, chunk_size: int = 128, exclude_ID: list = [],
                  dset_name: str = None, title: str = "", f: object = None) -> dict:
    pass
    input_size = 64
    n_classes = 2
    hyperprm = {
        'normalize': False,
        'batch_size': 64,
        'epochs': 10,
        'input_size': {input_size: None},
        'n_classes': {n_classes: None},
        'hidden_layers': [{32: 'relu'}, {32: 'relu'}]
    }
    if repository is None or dset_name is None:
        log2All("Data and /or Model Repositor arenot set.")
        return None
    DB_dset_name = Path(Path(dset_name) / Path(feature_type)/Path(title)).with_suffix(".csv")
    if not DB_dset_name.exists():
        log2All("Dataset DB {} is not found.".format(str(DB_dset_name)))
        return None

    DB_model_name = Path(Path(repository) / Path(feature_type)/Path(title)).with_suffix(".csv")
    if not DB_model_name.exists():
        log2All("Model DB {} is not found in repository.".format(str(DB_model_name)))
        return None

    df = pd.read_csv(str(DB_dset_name))
    dset_dict = {DB_DSET_KEY: df[DB_DSET_KEY].values.tolist(), DB_DSET_VAL: df[DB_DSET_VAL].values.tolist()}
    msg = dictIterate(ddict=dset_dict)
    msg2log(None, msg, f)
    del df
    df = None
    l_ID = dset_dict[DB_DSET_KEY]
    l_file = dset_dict[DB_DSET_VAL]

    df = pd.read_csv(str(DB_model_name))
    model_dict = {DB_DL_KEY: df[DB_DL_KEY].values.tolist(), DB_DL_VAL: df[DB_DL_VAL].values.tolist()}
    msg = dictIterate(ddict=model_dict)
    msg2log(None, msg, f)
    del df
    df = None
    m_ID = model_dict[DB_DL_KEY]
    m_file = model_dict[DB_DL_VAL]

    for key, val,saved_model in zip(m_ID, l_file,m_file):
        if key in exclude_ID:
            msg2log(None, "For ID={} detection is not carried out".format(key),f)
            continue
        msg = ""
        try:
            df = pd.read_csv(val)
        except:
            pass
            msg = "\nOoops! Unexpected error: {}\n{}\nFor ID={} DL model is not trained".format(sys.exc_info()[0],
                                                                                                sys.exc_info()[1], key)
        finally:
            if len(msg) > 0:
                msg2log(trainIDSpacCl.__name__, msg, D_LOGS['except'])

        if len(msg) > 0:
            continue

        columns = list(df.columns)
        x = df[columns[0]].values
        (n,) = x.shape
        m = len(columns) - 2
        X = np.zeros((n, m), dtype=float)
        j = 0
        for item in columns[1:-1]:  # eliminate 'Unnamed 0' - index and 'y'-desired data
            x = df[item].values
            X[:, j] = x[:]
            j += 1

        y = df[columns[-1]].values
        model_saving_path = digitalTwinDetect(model=DigitalTwinMLPids, name="DigitalTwinMLPids", X=X, y=y,
                            key=key, saved_model=saved_model, repository=repository, f=D_LOGS['predict'])
        log2All()
        del X
        X = None
        if model_saving_path is not None:
            m_ID.append(key)
            m_file.append(model_saving_path)

    d_saved_model = {'ID': m_ID, 'DLmodel': m_file}
    df = pd.DataFrame(data=d_saved_model)
    csv_file = Path(Path(repository) / Path(feature_type)/Path(title)).with_suffix(".csv")

    df.to_csv(str(csv_file), index=False)
    log2All("Deep Learning models saved in {}".format(str(csv_file)))
    log2All()

    return d_saved_model

@exec_time
def digitalTwinDetect(model:object=DigitalTwinMLPids, name:str="DigitalTwinMLPids", X:np.array=None, y:np.array=None,
                     key:str="",  saved_model:str=None, repository:str="", f:object=None):
    pass
    digital_twin=model(name = name,model_repository=saved_model, f=f)

    digital_twin.loadModel()
    """ test sequences """
    (n,m)=X.shape
    N=int(0.9*n)

    """ detect process """
    all_actual=[]
    all_prediction=[]

    predicted_y = digital_twin.predict(X[N:,:])
    actual = y[N:].astype('int32').tolist()
    (n1, _) = predicted_y.shape
    prediction = predicted_y.reshape((n1)).tolist()
    bin_confusion(actual, prediction, labels=['norm', 'anomaly'], target_state=1,
                  title="Binary Confusion Matrix ID={}".format(key),        f=D_LOGS['predict'])

    log2All()
    return




if __name__=="__main__":
    pass