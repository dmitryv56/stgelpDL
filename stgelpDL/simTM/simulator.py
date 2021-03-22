#!/usr/bin/python3
import sys
import copy

from simTM.DER import Powersrc,Diesel, HydrTrb,HydrPump, dieselGenset,hydrtrbGenset,hydrpumpGenset,\
    dieselGensetRequests, derGensetRequests
from simTM.imbPred import ImbPred

from simTM.cfg import DIESEL_NUMBER, DIESEL_MODEL, MAX_CUSTOM, MAX_GENER,DELTAPWR, SIM_PERIOD,AGE,\
    HYDR_PUMP_NUMBER, HYDR_PUMP_MODEL, HYDR_TRB_NUMBER,HYDR_TRB_MODEL
from simTM.tm import TM
# Types DER
from simTM.cfg import D_LOGS,DIESEL, PV, CHP, WIND_TRB, HYDR_TRB, HYDR_PUMP, DER_NAMES, PUMP_PERIOD
from simTM.api import listLogSet,closeLogs,log2All,state2history,history2log, logStep,plotsequences,sentLog
from predictor.utility import msg2log



def dieselById(id,dieselgenset:list)->Diesel:

    for diesel in dieselgenset:
        if id ==diesel.descr.desc["Id"]:
            return diesel
    return None

def dieselGensetUpdate(dieselgenset:list,send_reqs:dict,f:object=None):

    for id, descr in send_reqs.items():
        diesel = dieselById(id, dieselgenset)
        if diesel is None:
            continue
        diesel.descr=copy.copy(descr)
    return dieselgenset






def main():

    # f=open("Logs/aaa.log",'w+')
    # fst = open("Logs/steps.log", 'w+')
    listLogSet(folder=None)
    tm_sets = {}
    tm_sets[HYDR_PUMP] = hydrpumpGenset(f = D_LOGS["main"])  # hydropump Genset creates a flows one per "smart" pump
    tm_sets[DIESEL]    = dieselGenset(f = D_LOGS["main"])           # diesel Genset creates a flows one per "smart" diesel
    tm_sets[HYDR_TRB]  = hydrtrbGenset(f = D_LOGS["main"])          # hydroturbine Genset creates a flows one per "smart" turbine


    tm=TM(tm_sets,f = D_LOGS["main"])                               # The simulator creates TM
    imb = ImbPred(MAX_CUSTOM, MAX_GENER, f = D_LOGS["main"])   # The simulator creates 'Imbalance predictor'
    ImbPowerSeq=[]
    preTitle="pre-actions"
    postTitle="post-actions"


    for step in range(SIM_PERIOD):
        msg=""
        try:
            imb.genImbSeq(nSize=PUMP_PERIOD)                   # Imb.predictor generates a forecast for next period
            imbPwr=imb.imbSeq[0]                               # TM receives a predict value and nransform it in the 'credit
            tm.getPredict(imb.imbSeq)
            imbToken=round(imbPwr/DELTAPWR,2)                  # tokens
            if len(ImbPowerSeq)>=AGE:
                ImbPowerSeq.pop()
            ImbPowerSeq.append(imbToken)
            log2All("\nStep: {} - {}".format(step, preTitle))
            logStep(step=step, title=preTitle, tm_sets=tm_sets,   ImbValue=imbPwr, f=D_LOGS["steps"])

            try:
                smartreqs=derGensetRequests(tm_sets, imbPwr, f = D_LOGS["main"])
            except:
                msg2 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
                msg2log("main_1", msg2, f=D_LOGS["except"])

            try:
                tm.inPath(imbToken,smartreqs)                      # the flow comes in TM
            except:
                msg2 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
                msg2log("main_2", msg2, f=D_LOGS["except"])
            try:
                sent_ids = tm.outPath()                            # the flow leaves the TM or discarded into
            except:
                msg2 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
                msg2log("main_3", msg2, f=D_LOGS["except"])

            logStep(step=step, title=postTitle, tm_sets=tm_sets,  ImbValue=tm.ImbPwrCredit * DELTAPWR, f=D_LOGS["steps"])
            log2All("Step: {} - {}".format(step,postTitle))
            state2history(tm_sets,sent_ids)
            sentLog(sent_ids, f=D_LOGS["sent"])


        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg = "O-o-ops! Unexpected error: {}".format( sys.exc_info())
        finally:
            if len(msg)>0:
                msg2log("main", msg,f = D_LOGS["except"])
    # val = imb.genImbSeq(nSize=128)
    # diesel =Diesel(f=None)
    # n = diesel.randState()
    history2log(tm_sets,  f = D_LOGS["main"])
    plotsequences(tm_sets, ImbPowerSeq,folder=D_LOGS["plot"], f = D_LOGS["main"])
    closeLogs()
    # f.close()
    # fst.close()
    return 0

if __name__ == "__main__":
    main()
    pass