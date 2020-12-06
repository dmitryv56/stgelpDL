#!/usr/bin/python3
import sys
import copy

from simTM.DER import Powersrc,Diesel
from simTM.imbPred import ImbPred

from simTM.cfg import DIESEL_NUMBER, DIESEL_MODEL, MAX_CUSTOM, MAX_GENER,DELTAPWR, SIM_PERIOD,AGE
from simTM.tm import TM
# Types DER
from simTM.cfg import DIESEL, PV, CHP, WIND_TRB, HYDR_TRB, HYDR_PUMP, DER_NAMES
from simTM.api import state2history,history2log, logStep,plotsequences
from predictor.utility import msg2log

def dieselGenset(f:object=None)->list:
    dieselgenset=[]
    for i in range(DIESEL_NUMBER):
        dieselgenset.append(Diesel(DIESEL_MODEL[i%len(DIESEL_MODEL)],f))
    return dieselgenset
""" smartreqs is a dictionary whose key is 'priority' and value is a list of the reqests  """
def dieselGensetRequests(dieselgenset:list, imbPwr:float, f:object=None)->dict:
    smartreqs={}
    for diesel in dieselgenset:
        msg=""
        try:
            diesel.incPowerReq=None
            diesel.decPowerReq=None
            token = diesel.createSmartRequest(imbPwr)
            priority=diesel.descr.desc["Priority"]
            if priority not in smartreqs:
                smartreqs[priority] = []
            if token==0:   # do not create request for this diesel
                continue
            if diesel.incPowerReq is not None:
                smartreqs[priority].append(diesel.incPowerReq)
            if diesel.decPowerReq is not None:
                    smartreqs[priority].append(diesel.decPowerReq)
        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info()[0])

        finally:
            msg2log("dieselGensetRequests", msg, f)
    return smartreqs

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

    f=open("aaa.log",'w+')
    fst = open("steps.log", 'w+')
    dgenset = dieselGenset(f)                            # diesel Genset creates a flowes one per "smart" diesel
    tm_sets={}
    tm_sets[DIESEL]=dgenset
    tm=TM(tm_sets,f)                                             # The simulator creates TM
    imb = ImbPred(MAX_CUSTOM, MAX_GENER, f=None)         # THe simulator creates 'Imbalance predictor'
    ImbPowerSeq=[]
    preTitle="pre-actions"
    postTitle="post-actions"
    for step in range(SIM_PERIOD):
        msg=""
        try:
            imb.genImbSeq(nSize=1)                             # Imb.predictor generates a forecast for next period
            imbPwr=imb.imbSeq[0]                               # TM receives a predict value and nransform it in the 'credit
            imbToken=round(imbPwr/DELTAPWR,2)                  # tokens
            if len(ImbPowerSeq)>=AGE:
                ImbPowerSeq.pop()
            ImbPowerSeq.append(imbToken)

            logStep(step=step, title=preTitle, genset=dgenset, ImbValue=imbPwr, f=fst)
            smartreqs=dieselGensetRequests(dgenset, imbPwr, f) # Diesel genset creates a smart request flow for incPower or decPower
            tm.inPath(imbToken,smartreqs)                      # the flow comes in TM
            sent_ids = tm.outPath()                            # the flow leaves the TM or discarded into
            logStep(step=step, title=postTitle, genset=dgenset, ImbValue=tm.ImbPwrCredit * DELTAPWR, f=fst)
            state2history(tm_sets,sent_ids)





        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg = "O-o-ops! Unexpected error: {}".format( sys.exc_info()[0])
        finally:
            msg2log("main", msg,f)
    # val = imb.genImbSeq(nSize=128)
    # diesel =Diesel(f=None)
    # n = diesel.randState()
    history2log(tm_sets,  f=f)
    plotsequences(tm_sets, ImbPowerSeq, f=f)
    f.close()
    fst.close()
    return 0

if __name__ == "__main__":
    main()
    pass