#!/usr/bin/python3

""" Api for HMM_acppgeg """
import copy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from hmm.fwd import fwd_bkw, viterbi, probNormDist
from hmm.HMM import hmm_gmix
from hmm.HMM_drive import emissionDistribution,logDist, imprLogDist, transitionsDist,plotArray,drive_HMM,input4FwdBkw,\
logFwdBkwDict,logViterbi,plotViterbiPath
from hmm.state import state8, state36, DIESEL, WIND, HYDRO
from predictor.control import ControlPlane
from predictor.utility import  msg2log,tsBoundaries2log,logDictArima


STATES_DICT ={0: {'BLNC': 'Balance of consumer power and generated power'},
              1: {'LACK': 'Lack electrical power'},
              2: {'SIPC': 'Sharp increase in power consumption'},
              3: {'EXCS': 'Excess electrical power'},
              4: {'SIGP': 'Sharp increase in generation power'}
              }

CSV_PATH ="~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv"
aux_col_name = "Programmed_demand"
DATA_COL_NAME = "Imbalance"
DT_COL_NAME   = "Date Time"



def printListState(states_dict: dict, f:object=None):
    msg="{:^20s} {:^20s} {:^40s}".format("Index State","Short State Name","Full State Name")
    msg2log(None,"\n\n{}".format(msg),f)
    for i in range(len(states_dict)):
        shName,Name=getStateNamesByIndex(i,states_dict)
        msg = "{:>20d} {:^20s} {:^40s}".format(i, shName, Name)
        msg2log(None, msg, f)
    msg2log(None, "\n\n", f)
    return

def getStateNamesByIndex(indx_state,states_dict):
    shortStateName = "UNDF"
    fullStateName  = "UNDF"
    for key,val in states_dict.items():
        if key==indx_state:
            (shortStateName,fullStateName)=list(val.items())[0]

            break
    return shortStateName,fullStateName


""" Linear extrapolation for sharp increase in the power detection.
    It gets 3 consecutive  time series values [Y(k-1),Y(k),Y(k+1)]. First, for the median point, a decision is made 
    about its belonging to one of the primary states of states according to the following rules:
    {y ~ 0 => balance}, {y >> 0 => dominance consumers power},{y << 0 => dominance of generation power}.
    Second , for median point Y(k) , is cheked an  occurrence  of a sharp increase in generation (state 'SIGP', index 4)
    or consumption (state 'SIPC', index 2)  with using a linear extrapolation Yextrp(k+1) =F(Y(k),Y(k-1)).
    If Y(k)>Y(k-1), i.e. the function grows, and Y(k+1) > Yextrp(K+1) then Y(k) belongs to 'SIPC' state (the sharp i
    ncrease in power consumption).
    If Y(k)<Y9k-1), i.e. decrease function, and Y(k+1)<Yextrp(k+1) then Y(k) is in 'SIGP' state (Sharp increase in 
    generation power )/    
    
"""
def getLnExtrapState(yk:list)->int:
    if len(yk)!=3:
        return None
    if yk[1]<-1e-03 :    ret = 3 # Excess electrical power
    elif yk[1]>1e-03:    ret = 1  # Lack electrical power
    else:                ret = 0  # balance of consumer power and generated power

    if yk[1]>yk[0] and yk[2]>2.0 *yk[1]-yk[0] : ret = 2 # Sharp increase in customer power
    if yk[1]<yk[0] and yk[2]<2.0 *yk[1]-yk[0]:  ret = 4 # Sharp increase in generation power

    return ret
def getBoundState(yk:float)->int:
    if yk<-1e-03 :    ret = 3  # dominance of generation power
    elif yk>1e-03:    ret = 1  # dominance of consumer power
    else:             ret = 0  # balance of consumer power and generated power
    return ret

def readDataset(csv_file: str, data_col_name:str,dt_col_name:str,f:object=None)->(pd.DataFrame, list):
    # ds = pd.read_csv("~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv")

    # aux_col_name = "Programmed_demand"
    # data_col_name = "Imbalance"
    # dt_col_name = "Date Time"
    # ds[aux_col_name] = [0.0 for i in range(len(ds[aux_col_name]))]
    #
    # ds[dest_col_name] = ds[src_col_name]
    ds = pd.read_csv(csv_file)
    tsBoundaries2log(data_col_name, ds, dt_col_name, data_col_name, f)
    states=[]
    states.append(getBoundState(ds[data_col_name].values[0]))
    lnTS=len(ds[data_col_name])
    for k in range(1,lnTS-1):
        states.append(getLnExtrapState(ds[data_col_name].values[k-1:k+2]))
    states.append(getBoundState(ds[data_col_name].values[lnTS-1]))
    ds["hidden_states"]=states

    return ds, states

def plotStates(ds:pd.DataFrame, data_col_name:str,state_sequence:list, title:str, path_to_file:str, start_index: int=0,
               end_index: int = 0, f:object=None):

    if end_index >len(state_sequence): end_index = len(state_sequence)
    if end_index == 0:                 end_index = len(state_sequence)
    suffics='.png'
    ss="{}_{}_".format(str(start_index), str(end_index))
    file_png = Path(path_to_file, data_col_name + "_States_" + ss + Path(__file__).stem).with_suffix( suffics)
    fig,axs=plt.subplots(2)
    fig.suptitle("{} (from {} till {}".format(title,start_index,end_index))
    axs[0].plot(ds[data_col_name].values[start_index:end_index])
    axs[1].plot(state_sequence[start_index:end_index])

    plt.savefig(file_png)
    plt.close("all")
    return


# def trainHMMprob(concat_list: list, states_concat_list: list, cp: ControlPlane) -> (np.array, np.matrix, np.matrix):
def trainHMMprob(df:pd.DataFrame,data_col_name, states: list, cp: object) -> (np.array, np.matrix, np.matrix, list):
    """

    :param concat_list:
    :param states_concat_list:
    :param cp:
    :return:
    """

    trainArray = np.array(df[data_col_name].values)
    ar_state = np.array(states)
    mean = np.mean(trainArray)
    std = np.std(trainArray)

    np_states, counts = np.unique(ar_state, return_counts=True)

    emisDist = emissionDistribution(trainArray, ar_state, np_states, counts, cp.fc)

    transDist = transitionsDist(ar_state, np_states, cp.fc)

    pai = np.zeros((np_states.shape), dtype=float)
    for i in range(len(np_states)):
        pai[i] = counts[i] / len(trainArray)

    logDist(pai, "Initial Distribution", cp.fc)

    plotArray(trainArray, "Train Time Series for hmm", "train_HMM", cp)
    del trainArray
    plotArray(ar_state[:64], "Train Hidden States for hmm (first 64 items)", "hiddenStates_HMM_first", cp)
    if (len(ar_state) - 64) >= 0:
        plotArray(ar_state[len(ar_state) - 64:], "Train Hidden States for hmm (last items)", "hiddenStates_HMM_last",
                  cp)
    del ar_state

    return (pai, transDist, emisDist,np_states.tolist())


def trainPath(cp):
    # (concat_list, states, states_concat_list, states_set, observations, observation_labels) =trainDataset(cp, mypath, LISTCSV)
    # (pai, transDist, emisDist) = trainHMMprob(concat_list, states_concat_list, cp)

    csv_file      = cp.csv_path
    data_col_name = cp.rcpower_dset
    dt_col_name   = cp.dt_dset
    discret       = cp.discret
    intermediateFolder ="hmm_"+data_col_name
    modelFileName = intermediateFolder+".json"


    # ds, state_sequence = readDataset(csv_file, data_col_name, dt_col_name, cp.fc)
    #states_dict=copy.copy(STATES_DICT)

    st = state36(cp.fc) #st = state8(cp.fc)
    ds, state_sequence= st.readDataset(csv_file, data_col_name, dt_col_name, [DIESEL, WIND, HYDRO])
    ds.to_csv("Imbalance_ElHierro_hiddenStates.csv")
    st.printListState()
    st.stateMeanStd(ds, data_col_name)
    st.logMeanStd()
    states_dict =copy.copy(st.states_dict)

    printListState(states_dict, cp.fc)
    printListState(states_dict, cp.fp)
    printListState(states_dict, cp.ft)






    csv_file_with_hidden_states=Path(cp.folder_control_log/"backup_ds.csv")
    ds.to_csv(str(csv_file_with_hidden_states))
    title = "{} and States".format(data_col_name)

    observations       = np.array(ds[data_col_name].values)
    observation_labels = np.array(ds[dt_col_name].values)
    step_index=512
    for start_index in range(0,len(state_sequence), step_index ):
        plotStates(ds, data_col_name, state_sequence, title, cp.folder_control_log, start_index=start_index,
                   end_index=start_index+step_index, f=cp.fc)

    pai, transDist, emisDist, states = trainHMMprob(ds, data_col_name, state_sequence, cp)

    imprLogDist(pai,       states, [0],             "Initial Distribution",    f=cp.ft)
    imprLogDist(transDist, states, states,          "Transition Distribution", f=cp.ft)
    imprLogDist(emisDist,  states, ['mean', 'std'], "Emission Distribution",   f=cp.ft)



    hmm_model = hmm_gmix(len(states), states, cp.ft)
    hmm_model.setInitialDist(pai)
    hmm_model.setTransitDist(transDist)
    hmm_model.setEmisDist(emisDist)
    pathHMMName = Path(Path(cp.path_repository) / Path(intermediateFolder ) / Path("continuos_obs"))
    pathHMMName.mkdir(parents=True, exist_ok=True)
    fileName = Path(pathHMMName / modelFileName)
    hmm_model.dumpModel(fileName)
    msg2log(None, "HMM dumped in {}\n".format(fileName), cp.ft)

    msg2log(None, "Test sequence: {} and 'Hidden' states\n".format(data_col_name), cp.ft)
    test_cutof = 256
    for i in range(len(observations[test_cutof:])):
        s = '{} {}'.format(observations[i], state_sequence[i])
        msg2log(None, s, cp.ft)

    # post_mode,post_marg_name, post_marg_logits  = drive_HMM(cp, pai, transDist, emisDist, observations[test_cutof:],
    #                                                        observation_labels[test_cutof:],state_sequence[test_cutof:])

    post_mode_test_cutof, post_marg_name_test_cutof, post_marg_logits_test_cutof = drive_HMM(cp, pai, transDist,
                                                                                             emisDist,
                                                                                             observations[:test_cutof],
                                                                                             observation_labels[
                                                                                             :test_cutof],
                                                                                             state_sequence[
                                                                                             :test_cutof])
    msg2log(None,"Test for first {} period started at {}".format(test_cutof,observation_labels[0]),cp.ft)
    diffSequences(  post_mode_test_cutof, state_sequence[:test_cutof], observation_labels[:test_cutof],
                    states_dict,cp.ft)


    post_mode,post_marg_name, post_marg_logits  = drive_HMM(cp, pai, transDist, emisDist, observations,
                                                           observation_labels, state_sequence)
    msg2log(None, "HMM. {} period. Started at {}, finished at {}\n".format(len(observations),observation_labels[0],
                                                                observation_labels[len(observation_labels)-1]), cp.fp)
    diffSequences(post_mode, state_sequence, observation_labels, states_dict,cp.fp)

    log_state_file = Path( cp.folder_predict_log / "statesTotal.log")
    with open(log_state_file,"w+") as flog:
        logStateSequence(ds, dt_col_name, data_col_name, state_sequence, post_mode,st, flog)
    flog.close()
    ds["ViterbiPath"] = post_mode.tolist()
    ds.to_csv("ElHiero_2409_27102020_statepathes.csv")
    if 1: return

    aux_states = [str(states[i]) for i in range(len(states))]
    del states
    states = tuple(aux_states)
    msg2log(None, "Prepare an input data for Forward-backward and Viterbi  algorithms\n", cp.fp)

    (d_start_prob, d_transDist, d_emisDist) = input4FwdBkw(states, observation_labels[test_cutof:],
                                                           observations[test_cutof:], pai, transDist, emisDist, cp.ft)

    msg2log(None, "\nForward-backward algorithm\n", cp.ft)
    fwd, bkw, posterior = fwd_bkw(observation_labels[test_cutof:], states, d_start_prob, d_transDist, d_emisDist,
                                  states[len(states) - 1], cp.ft)
    msg2log(None, "\nAposterior probability for each timestamp\n", cp.ft)
    logDictArima(posterior, 4, cp.ft)
    print(fwd)
    print(bkw)

    logFwdBkwDict(fwd, bkw, observation_labels[test_cutof:], cp.ft)

    msg2log(None, "\nViterbi algorithm\n", cp.fp)
    opt, max_prob, V = viterbi(observation_labels[test_cutof:], states, d_start_prob, d_transDist, d_emisDist, cp.ft)

    logViterbi(V, cp.ftttt)

    cp.fp.write('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

    plotViterbiPath("fwd", observation_labels[test_cutof:], opt, state_sequence[test_cutof:], cp)

    return


def predictPath(cp: ControlPlane, source_folder: str, source_file: str):

    if 1: return


    csv_file = cp.csv_path
    data_col_name = cp.rcpower_dset
    dt_col_name = cp.dt_dset
    discret = cp.discret\

    (observations, observation_labels, hidden_states) = predictDataset(cp, source_folder, source_file)

    hmm_model = hmm_gmix(2, [0, 1], cp.fc)

    pathHMMName = Path(Path(cp.path_repository) / Path("hmm"+data_col_name) / Path("continuos_obs"))

    fileName = Path(pathHMMName / "hmm"+data_col_name +".json")
    if False == fileName.is_file():
        return

    hmm_model.loadModel(fileName)

    (pai, transitDist, emisDist) = hmm_model.getModel()

    drive_HMM(cp, pai, transitDist, emisDist, observations, observation_labels, None)
    drive_HMM(cp, pai, transitDist, emisDist, observations, observation_labels, hidden_states)

    return


def diffSequences(seqViterbi: np.array,seqOrigin:np.array,observation_labels:np.array,states_dict:dict,f:object=None):
    diff = np.subtract(seqViterbi,seqOrigin)
    diff_value,diff_count =np.unique(diff,return_counts=True)

    msg2log(None, "Differences in Viterbi and original sequence of states", f)
    for i in range(len(diff_value)):
        msg="Distance {}  Amount {}".format(diff_value[i],diff_count[i])
        if diff_value[i] == 0:
            msg = "Match! Distance {}  Amount {}".format(diff_value[i], diff_count[i])
        msg2log(None,msg,f)

    msg="{:<30s}:{:>10s}{:>10s}{:<40s} ".format("Date Time", "Viterbi","Origin","Previous")
    msg2log(None, msg, f)
    if (diff[0]!=0):
        s0,_ = getStateNamesByIndex(seqViterbi[0], states_dict)
        s1,_ = getStateNamesByIndex(seqOrigin[0], states_dict)
        msg2log(None, "{:<30s}:{:>10s}{:<40s}".format(observation_labels[0], s0,s1 ), f)

    for i in range(1,len(diff)):
        if (diff[i]!=0):
            s0,_ = getStateNamesByIndex(seqViterbi[i], states_dict)
            s1,_ = getStateNamesByIndex(seqOrigin[i], states_dict)
            s2,_ = getStateNamesByIndex(seqOrigin[i - 1], states_dict)
            msg2log(None, "{:<30s}:  {:>10s}  {:>10s}  {:<60s} ".format(observation_labels[i],s0,s1,s2), f)

    return

def logStateSequence(ds, dt_col_name, data_col_name, state_sequence, viterbi_sequence,stateObj:object, f:object=None):
    pass

    savef=stateObj.f
    stateObj.f=f
    stateObj.printListState()
    stateObj.f=savef
    msg="{:<6s} {:<30s} {:<9s} {:<7s} {:<7s} {:<7s} {:<7s}".format("##", "Date Time", "Imbalance", "Hidden","States",
                                                                   "Viterby","Path" )
    msg2log(None, msg, f)
    for i in range(len(ds)):
        dt=ds[dt_col_name].values[i]
        val=ds[data_col_name].values[i]
        st =state_sequence[i]
        st_name,_=stateObj.getStateNamesByIndex(st)
        vt=viterbi_sequence[i]
        vt_name, _ = stateObj.getStateNamesByIndex(vt)
        f.write("{:>6d} {:<30s} {:<9.4f} {:>7d} {:<7s} {:>7d} {:<7s}\n".format(i,dt,val,st,st_name,vt,vt_name))

    return










