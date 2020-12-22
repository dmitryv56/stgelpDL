#!/us/bin/python3

""" Api - module"""
import sys
from pathlib import Path

from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt

from simTM.descriptor import Descriptor, SmartReq
from simTM.auxapi import dictIterate, listIterate
from predictor.utility import msg2log

# Request type
from simTM.cfg import DEC_PWR, INC_PWR
# A possible state of the units of DER
from simTM.cfg import D_LOGS, UNDF,S_OFF, S_LPWR, S_MPWR, S_HPWR, DIESEL_STATES_NAMES
# Types DER
from simTM.cfg import DIESEL, PV, CHP, WIND_TRB, HYDR_TRB, HYDR_PUMP, DER_NAMES, HYDR_TRB_STATES_NAMES, \
    HYDR_PUMP_STATES_NAMES
# Priorites
from simTM.cfg import PR_L, PR_M, PR_H

from simTM.cfg import DELTAPWR, AGE

""" Aux. functions and configuration data structures"""
D_PRIORITY = {PR_L: "Low Priority", PR_M: "Midle Priority", PR_H: "High Priority"}

def listLogSet(folder:str=None):
    path_folder=None
    if folder is None:
        path_folder=Path("Logs")
    else:
        path_folder = Path(folder)
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    suffics=".log"
    for log_name, handler in D_LOGS.items():
        log_name_path=Path(path_folder/log_name).with_suffix(suffics)
        if log_name == "plot":
            Path(path_folder/log_name).mkdir(parents=True, exist_ok=True)
            D_LOGS[log_name] = str(Path(path_folder/log_name))
            continue
        f=open(str(log_name_path),'w+')
        if f is not None:
            D_LOGS[log_name]=f
    return

def closeLogs():
    for log_name, handler in D_LOGS.items():
        if log_name=="plot":
            continue
        if handler is not None:
            handler.close()
            handler=None
    return
def log2All(msg:str=None):
    if msg is None or len(msg)==0:
        return

    for log_name, handler in D_LOGS.items():
        if log_name == "plot":
            continue
        if handler is not None:
           handler.write("{}\n".format(msg))
    return



def randStates(dict_states: dict) -> (int, str):
    n_states = len(dict_states)
    # sts = np.random.randint(low=0, high=n_states - 1, size=1)
    sts = np.random.randint(low=n_states,  size=1)
    return sts[0], dict_states[sts[0]]

def getStateName(typeDer:int=None, descr:Descriptor = None):
    msg=""
    st=""
    try:
        if typeDer is None or descr is None:
            return None
        if descr.desc["CurrentState"]==UNDF:
            message ="type ={} CurrentState is undefined. Changed to OFF to prevent except state".format(typeDer)
            msg2log("getStateName", message, f=D_LOGS["except"])
            descr.setDesc(CurrentState=S_OFF,CurrentPower=0.0)

        if typeDer==DIESEL:
            st=DIESEL_STATES_NAMES[descr.desc["CurrentState"]]
        elif typeDer == HYDR_TRB:
            st = HYDR_TRB_STATES_NAMES[descr.desc["CurrentState"]]
        elif typeDer == HYDR_PUMP:
            st = HYDR_PUMP_STATES_NAMES[descr.desc["CurrentState"]]
    except:
        msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())

    finally:
        if len(msg)>0:
            msg2log("getStateName", msg, f=D_LOGS["except"])
            msg2log("getStateName", str(descr.desc["CurrentState"]), f=D_LOGS["except"])
    return st

def logStep(step: int = 0, title: str = None, tm_sets:dict=None, ImbValue: float = None, f: object = None):
    try:
        message = "S T E P  {}\n {} Timestemp {} Predicted Imbalance {},MWt (simulated)".format(step, title, step,
                                                                                                round(ImbValue, 3))
        msg2log(None, message, f)
        message = "{:<3s} {:<20s} {:<8s} {:<8s} {:<8s} ".format('id', 'name', 'model', 'state', 'pwr, Mwt')
        msg2log(None, message, f)
        for typeDer, genset in tm_sets.items():

            try:
                for item in genset:
                    descr: Descriptor = item.descr
                    model: str = item.model

                    try:
                        name = DER_NAMES[item.typeDer]
                        id = item.id #descr.desc["Id"]
                    except:
                        msg4 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
                        msg2log("logStep_2", msg4, f=D_LOGS["except"])

                    try:
                        st = getStateName(typeDer=item.typeDer, descr=descr) #DIESEL_STATES_NAMES[descr.desc["CurrentState"]]
                        try:
                            pwr = descr.desc['CurrentPower']
                        except:
                            msg6 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
                            msg2log("logStep_4", msg6, f=D_LOGS["except"])
                    except:
                        msg5 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
                        msg2log("logStep_3", msg5, f=D_LOGS["except"])

                    message = "{:>3d} {:<20s} {:<8s} {:<8s} {:>6.3f} ...".format(id, name, model, st, pwr)
                    msg2log(None, message, f)
            except:
                msg3 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
                msg2log("logStep_1", msg3, f=D_LOGS["except"])

    except:
        msg2 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
        msg2log("logStep", msg2, f=D_LOGS["except"])
        msg2log("tm_sets",tm_sets,f=D_LOGS["except"])
        msg2log("tm_sets-dictIterate", dictIterate(tm_sets), f=D_LOGS["except"])


def state2history(tmsets: dict, sent_ids: dict):
    for typeDer, listDer in tmsets.items():
        for item in listDer:
            id = item.descr.desc["Id"]
            st = item.descr.desc["CurrentState"]
            snt = 0
            if id in sent_ids:
                snt = 1
            aging(item.descr, st=st, snt=snt)
    return


def aging(desc: Descriptor, st: int = None, snt: int = None, ):
    pass
    if st is not None:
        if len(desc.last_state_seq) >= AGE:
            desc.last_state_seq.pop()
        desc.last_state_seq.append(st)
    if snt is not None:
        if len(desc.last_send_seq) >= AGE:
            desc.last_send_seq.pop()
    desc.last_send_seq.append(snt)


def history2log(tmsets: dict, folder_png: str = None, f: object = None):
    for typeDer, listDer in tmsets.items():
        for item in listDer:
            id = item.id # item.descr.desc["Id"]
            stseq = item.descr.last_state_seq
            sntseq = item.descr.last_send_seq
            msg = "{:>4d} {:<20s} {:<24s}".format(id, DER_NAMES[typeDer], item.model)
            logsequences(stseq, sntseq, f=f)


def logsequences(s1: list, s2: list, title1: str = "state", title2: str = "send", n_group=4, f: object = None):
    n_size = len(s1)
    n_row = ceil(n_size / n_group)
    sheader = " NN {:<8s} {:<5s}  NN {:<8s} {:<5s}  NN {:<8s} {:<5s}  NN {:<8s} {:<5s} ".format(title1, title2,
                                                                                                title1, title2, title1,
                                                                                                title2, title1, title2)
    msg2log(None, sheader, f)
    for i in range(n_row):
        ss = DIESEL_STATES_NAMES[s1[i]]
        snt = "sent" if s2[i] > 0 else "no"
        i1 = i + n_row
        ss1 = DIESEL_STATES_NAMES[s1[i1]]
        snt1 = "sent" if s2[i1] > 0 else "no"
        i2 = i + 2 * n_row
        ss2 = DIESEL_STATES_NAMES[s1[i2]]
        snt2 = "sent" if s2[i2] > 0 else "no"
        i3 = i + 3 * n_row
        ss3 = DIESEL_STATES_NAMES[s1[i3]]
        snt3 = "sent" if s2[i3] > 0 else "no"
        srow = "{:>3d} {:<8s} {:<5s} {:>3d} {:<8s} {:<5s} {:>3d} {:<8s} {:<5s} {:>3d} {:<8s} {:<5s}".format(i, ss, snt,
                                                                                                            i1, ss1,
                                                                                                            snt1, i2,
                                                                                                            ss2, snt2,
                                                                                                            i3, ss3,
                                                                                                            snt3)
        if i3 >= n_size:
            srow = "{:>3d} {:<8s} {:<5s} {:>3d} {:<8s} {:<5s} {:>3d} {:<8s} {:<5s}".format(i, ss, snt, i1, ss1, snt1,
                                                                                           i2, ss2, snt2)

        msg2log(None, srow, f)
    msg2log(None, "\n", f)
    return


def plotsequences(tm_set: dict, ImbSeq: list, folder: str = None, f: object = None):

    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)

    n_size = len(ImbSeq)
    x=[i for i in range(n_size)]
    for typeDer, listDer in tm_set.items():
        if folder is None:
            path_png = "{}_.png".format(DER_NAMES[typeDer])
        else:
            path_path_png = Path(folder/Path("{}_.png".format(DER_NAMES[typeDer])))
            path_png = str(path_path_png)

        n = len(listDer)
        fig, aax = plt.subplots(n + 1)

        k=0
        aax[k].plot(x, ImbSeq)
        aax[k].set_title('Imbalance (tokens, 1 token = {} MWt )'.format(DELTAPWR))
        for item in listDer:
            id = item.descr.desc["Id"]
            aax[0].plot(x,ImbSeq)
            k=k+1
            aax[k].plot(x,item.descr.last_state_seq)
            aax[k].set_title('{}_{}'.format(id, item.model))
        plt.savefig(path_png)
        plt.close("all")
    return



def sentLog(ids_sent:dict = None,f:object=None):
    if ids_sent is None:
        return
    if not ids_sent:
        return

    msg=dictIterate(ids_sent)
    msg2log(None,msg,f)
