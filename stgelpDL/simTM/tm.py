#!/usr/bin/python3

"""Traffic Manager for Distributed Energy Reserve"""
import sys
from dataclasses import dataclass,field
from typing import Any
import queue

from simTM.descriptor import SmartReq
# from predictor.utility import msg2log

# Request type
from simTM.cfg import DEC_PWR, INC_PWR
# from simTM.DER import Diesel
from predictor.utility import msg2log

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


class TM:
    def __init__(self,tm_sets:list,f:object = None):
        self.OutQ=queue.PriorityQueue()
        self.ImbPwrCredit=0
        self.tm_sets=tm_sets  # dictionary contains all list of used DER
        self.f=f

    def push(self,priority:int, smartreq:SmartReq )->PrioritizedItem:
        item = PrioritizedItem(priority,smartreq)
        self.OutQ.put(item)

    def pop(self):
        item=None
        if self.OutQ.qsize()>0:
            item=self.OutQ.get()

        return item

    def clean(self):
        while not self.OutQ.empty():
            self.OutQ.get()
        return

    def policerTM(self):
        return

    def meteringTM(self):
        return

    def inPath(self, ImbPwrCredit:int, smartreqs: dict):
        self.clean()
        self.meteringTM()
        self.policerTM()

        self.ImbPwrCredit=ImbPwrCredit
        for priority,list_smartreqs in smartreqs.items():
            try:
                for smartreq in list_smartreqs:
                    self.push(priority,smartreq)
            except:
                pass
        return

    def outPath(self)->dict:
        pass
        dsent_ids= {}
        while abs(self.ImbPwrCredit)>0 and ( not self.OutQ.empty()):
            msg=""
            try:
                item=self.pop()
                priority=item.priority
                smartreq:SmartReq=item.item
                id=smartreq.id
                dsent_ids[id]=smartreq.descr
                der_device = self.getDevice(smartreq)

                UtilizedTokens={'inc':0,'dec':0}
                if smartreq.typeReq==DEC_PWR:
                    decrease= smartreq.token - self.ImbPwrCredit if smartreq.token > self.ImbPwrCredit else smartreq.token

                    self.ImbPwrCredit -= decrease
                    UtilizedTokens['dec']=decrease
                    # TODO update CurrentPower and CreditTokes in descriptor !!!!!!
                elif smartreq.typeReq==INC_PWR:
                    increase = smartreq.token + self.ImbPwrCredit if smartreq.token > abs(self.ImbPwrCredit) else smartreq.token
                    self.ImbPwrCredit += increase
                    UtilizedTokens['inc'] = increase
                    # TODO update Current Power and CreditTokens in descriptor

                der_device.updateDescr(mode='outTM', UtilizedTokens=UtilizedTokens)
            except KeyError as e:
                msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
            except:
                msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info()[0])
            finally:
                msg2log("outPath", msg, self.f)
            #while continue

        return dsent_ids

    def getDevice(self, smartreq:SmartReq)->object:
        pass
        der_list = self.tm_sets[smartreq.typeDer]
        der_device = None
        for item in der_list:
            if item.descr.desc["Id"]==smartreq.id:
                der_device=item
                break
        return der_device

    if __name__ == "__main__":
        pass