#!/usr/bin/python3

"""Traffic Manager for Distributed Energy Reserve"""
import copy
import sys
from dataclasses import dataclass,field
from typing import Any
import queue

from simTM.descriptor import SmartReq
# from predictor.utility import msg2log

# Request type
from simTM.cfg import D_LOGS, DEC_PWR, INC_PWR,NO_COLOR,GREEN,ORANGE,RED
# from simTM.menering  import Metering, policer
from simTM.metering import Metering, Policer
from predictor.utility import msg2log

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


class TM:
    def __init__(self,tm_sets:list,f:object = None):
        self.OutQGreen   = queue.PriorityQueue()
        self.OutQOrange  = queue.PriorityQueue()
        self.OutQRed     = queue.PriorityQueue()
        self.OutQNoColor = queue.PriorityQueue()
        self.ImbPwrCredit=0
        self.tm_sets=tm_sets  # dictionary contains all list of used DER
        self.f=f
        self.policer = None
        self.metering = None
        self.short_pred=[]

    def getPredict(self,y):
        self.short_pred=copy.copy(y)


    def push(self,priority:int, smartreq:SmartReq )->PrioritizedItem:
        item = PrioritizedItem(priority,smartreq)
        if smartreq.descr.desc["Color"]== GREEN:
            self.OutQGreen.put(item)
        elif smartreq.descr.desc["Color"]== ORANGE:
            self.OutQOrange.put(item)
        elif smartreq.descr.desc["Color"]==RED:
            self.OutQRed.put(item)
        else:
            self.OutQNoColor.put(item)

    def pop(self):
        item=None
        if self.OutQGreen.qsize()>0:
            item=self.OutQGreen.get()
        elif self.OutQOrange.qsize()>0:
            item = self.OutQOrange.get()
        elif self.OutQRed.qsize()>0:
            item = self.OutQRed.get()
        elif self.OutQNoColor.qsize()>0:
            item = self.OutQNoColor.put()


        return item

    def clean(self):
        while not self.OutQGreen.empty():
            self.OutQGreen.get()
        while not self.OutQOrange.empty():
            self.OutQOrange.get()
        while not self.OutQRed.empty():
            self.OutQRed.get()
        while not self.OutQNoColor.empty():
            self.OutQNoColor.get()
        return
    def isOutQempty(self):
        return self.OutQGreen.empty() and self.OutQOrange.empty() and self.OutQRed.empty() and self.OutQNoColor.empty()

    def policerTM(self):
        self.policer=Policer(self.metering, f=self.f)

    def meteringTM(self):
        self.metering=Metering(self.short_pred,f=self.f)
        return

    def inPath(self, ImbPwrCredit:int, smartreqs: dict):
        self.clean()
        self.meteringTM()
        self.metering.requestsParse(smartreqs)
        self.policerTM()

        self.ImbPwrCredit=ImbPwrCredit
        for priority,list_smartreqs in smartreqs.items():
            msg=""
            try:
                for smartreq in list_smartreqs:

                    self.metering.smartreq=smartreq
                    self.policer.setColor()

                    try:
                        if not self.policer.isDiscard():
                           self.push(priority,smartreq)
                    except:
                        msg1 = "O-o-ops!   I got an unexpected error - reason  {}".format(sys.exc_info())
                        msg2log("InPath _3", msg1, D_LOGS["except"])
                        msg1=""
            except :
                msg = "O-o-ops!   I got an unexpected error - reason  {}".format(sys.exc_info())
            finally:
                if len(msg)>0:
                    msg2log("InPath", msg,D_LOGS["except"])
        return

    def outPath(self)->dict:
        pass
        dsent_ids= {}
        while abs(self.ImbPwrCredit)>0 and ( not self.isOutQempty()):
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
                if len(msg)>0:
                    msg2log("outPath", msg, D_LOGS["except"])
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