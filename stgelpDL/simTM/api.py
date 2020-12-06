#!/us/bin/python3

""" Api - module"""

from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt

from simTM.descriptor import Descriptor, SmartReq
from predictor.utility import msg2log

# Request type
from simTM.cfg import DEC_PWR, INC_PWR
# A possible state of the units of DER
from simTM.cfg import S_OFF, S_LPWR, S_MPWR, S_HPWR, DIESEL_STATES_NAMES
# Types DER
from simTM.cfg import DIESEL, PV, CHP, WIND_TRB, HYDR_TRB, HYDR_PUMP, DER_NAMES
# Priorites
from simTM.cfg import PR_L, PR_M, PR_H

from simTM.cfg import DELTAPWR, AGE

""" Aux. functions and configuration data structures"""
D_PRIORITY = {PR_L: "Low Priority", PR_M: "Midle Priority", PR_H: "High Priority"}


def randStates(dict_states: dict) -> (int, str):
    n_states = len(dict_states)
    sts = np.random.randint(low=0, high=n_states - 1, size=1)
    return sts[0], dict_states[sts[0]]


def logStep(step: int = 0, title: str = None, genset: list = None, ImbValue: float = None, f: object = None):
    message = "S T E P  {}\n {} Timestemp {} Predicted Imbalance {},MWt (simulated)".format(step, title, step,
                                                                                            round(ImbValue, 3))
    msg2log(None, message, f)
    for item in genset:
        descr: Descriptor = item.descr
        model: str = item.model
        name = DER_NAMES[item.typeDer]
        id = descr.desc["Id"]
        st = DIESEL_STATES_NAMES[descr.desc["CurrentState"]]
        pwr = descr.desc['CurrentPower']
        message = "{:>3d} {:<20s} {:<8s} {:<8s} {:>6.3f} ...".format(id, name, model, st, pwr)
        msg2log(None, message, f)


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
            id = item.descr.desc["Id"]
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
    n_size = len(ImbSeq)
    x=[i for i in range(n_size)]
    for typeDer, listDer in tm_set.items():
        n = len(listDer)
        fig, aax = plt.subplots(n + 1)

        k=0
        aax[k].plot(x, ImbSeq)
        aax[k].set_title('Imbalance ( {} MWt tokens)'.format(DELTAPWR))
        for item in listDer:
            id = item.descr.desc["Id"]
            aax[0].plot(x,ImbSeq)
            k=k+1
            aax[k].plot(x,item.descr.last_state_seq)
            aax[k].set_title('{}_{}'.format(id, item.model))
        plt.savefig("{}_.png".format(DER_NAMES[typeDer]))
        plt.close("all")
    return
