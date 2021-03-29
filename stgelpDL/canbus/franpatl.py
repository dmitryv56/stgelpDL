#!/usr/bin/python3

""" 'franpatl' - frequency analysis of packet appearances at the transport level of Can Bus-FD.
"""

from os import path
import sys
from datetime import datetime,timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.stats import poisson,ks_2samp


from predictor.utility import msg2log
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from canbus.api import readChunkFromCanBusDump,file_len,dict2csv,dict2pkl, pkl2dict,mleexp,manageDB,DB_NAME,fillQuery,\
fillQuery2Test,getSerializedModelData,KL_decision
from canbus.clparser import parserPrAnFapTL,strParamPrAnFapTL


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


"""
canbusdump = "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
match_key = 'ID'  # 'ID',"Packet','Data'
"""
def main(arc,argv):
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")  # using in folder/file names
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    argparse, message0 = parserPrAnFapTL()
    """ command-line paremeters to local variables """
    title = argparse.cl_title
    mode = argparse.cl_mode
    method = argparse.cl_method
    n_train = int(argparse.cl_trainsize)
    n_test = int(argparse.cl_testsize)
    chunk_size=int(argparse.cl_chunk)
    canbusdump = argparse.cl_icsimdump

    filter_size = int(argparse.cl_bfsize)
    fp_prob = float(argparse.cl_bfprob)
    match_key=argparse.cl_match_key

    title1 = "{}_{}_{}_key_is_{}".format(title.replace(' ', '_'), mode, method, match_key)

    """ create log and repository folders """
    dir_path = path.dirname(path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(title1, date_time)
    folder_for_logging.mkdir(parents=True, exist_ok=True)
    folder_repository = Path(Path(dir_path) / Path("Repository") / Path(method))
    folder_repository.mkdir(parents=True, exist_ok=True)
    # folder_ppd = Path(folder_repository / title1 / method)
    # folder_ppd.mkdir(parents=True, exist_ok=True)


    param = (title, mode, method, n_train, n_test, canbusdump, match_key, chunk_size, filter_size, fp_prob,
             folder_for_logging, folder_repository)
    message2 = strParamPrAnFapTL(param)

    """ init logs """
    listLogSet(str(folder_for_logging))  # A logs are creating

    msg2log(None, message0, D_LOGS['clargs'])
    msg2log(None, message1, D_LOGS['timeexec'])
    msg2log(None, message2, D_LOGS['main'])
    n_samples=file_len(canbusdump)
    s_samples=n_samples if n_samples>0 else '0'
    msg2log(None,"\n{} lines in {}\n\n".format(s_samples, canbusdump))
    subtitle = "{}".format(mode)
    dset_name=Path( Path(folder_repository) / Path("{}_{}".format("dataset",match_key))).with_suffix(".csv")
    if mode != 'test':

        trainFreqModel(canbusdump= canbusdump, match_key=match_key, repository=str(folder_repository), title= subtitle,
                   dset_name=str(dset_name), f=D_LOGS['train'])

    if mode != 'train':
        testFreqModel(canbusdump= canbusdump, match_key=match_key, repository=str(folder_repository),
                      dset_name=str(dset_name), title= subtitle, f=D_LOGS['predict'])



    message1 = "Time execution logging finished at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    msg2log(None, message1, D_LOGS['timeexec'])
    closeLogs()
    return 0


if __name__=="__main__":
    # offset_line = 4
    # chunk_size = 2
    # canbusdump = "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
    # ft = open("log.log", 'w+')
    # ld = readChunkFromCanBusDump(offset_line=offset_line, chunk_size=chunk_size, canbusdump=canbusdump, f=ft)
    # time=float(ld[0]['DateTime'])
    # utc_time0 = datetime.utcfromtimestamp(time)
    # utc_time = datetime.fromtimestamp(time, timezone.utc)
    # local_time=utc_time.astimezone()
    # print(utc_time.strftime("%Y-%m-%d %H:%M:%S.%f+00:00 (UTC)"))
    # print(local_time.strftime("%Y-%m-%d %H:%M:%S.%f%z (%Z)"))
    # tst=datetime.fromtimestamp(math.floor(utc_time))
    nret = main(len(sys.argv),sys.argv)

    sys.exit(nret)