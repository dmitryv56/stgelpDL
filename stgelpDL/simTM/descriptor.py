#!/usr/bin/python3

import copy
import sys

from pickle import dump, load

from predictor.utility import msg2log

from simTM.cfg import S_OFF, S_LPWR, S_MPWR, S_HPWR, DIESEL_STATES, HYDR_TRB_STATES, HYDR_PUMP_STATES,UNDF
# Types DER
from simTM.cfg import DER_NAMES, DIESEL,HYDR_TRB,HYDR_PUMP,D_LOGS
#Colors
from simTM.cfg import NO_COLOR,RED ,ORANGE ,GREEN ,D_COLOR
# Request type
from simTM.cfg import DEC_PWR , INC_PWR, DELTAPWR
from simTM.auxapi import dictIterate, listIterate

""" Descriptor """


class Descriptor:
    def __init__(self, typeDER,id,Priority: int = -1, CurrentState: int =-1,f: object = None ):
        self.desc={'type':typeDER,
                   'Id':id,
                   'Priority':Priority,
                   "QoS1":-1,
                   "QoS2":-1,
                   "QoS3":-1,
                   'CurrentState':CurrentState,
                   'PreviousState':-1,
                   'CurrentPower':0.0,
                   'CreditTokens': {'inc':0,'dec':0},
                   'NumStarts':0,
                   'TotalOperationTime':-1,
                   'OperatingTimes': {},
                   "LastStatePeriod":-1,
                   "Color":NO_COLOR}
        self.desc["OperatingTimes"]=self.initOperatingTimes()
        self.last_state_seq = []
        self.last_send_seq  = []
        self.f = f
        return

    def __str__(self):
        msg0 ="Descriptor : "
        max_width = 120
        curr_width=0
        msg = "{" + dictIterate(self.desc, max_width= max_width, curr_width=curr_width) +"}"
        s="\nState Sequence:"
        msg="{}{}".format(msg,s)
        msg1=listIterate(self.last_state_seq,  curr_width=len(s))
        s = "\nSent Requests: "
        # msg = "{}\n{}\n{}".format(msg, msg1,s)
        msg = "{}{}{}".format(msg, msg1, s)
        msg1 = listIterate(self.last_send_seq, curr_width=len(s))
        # msg = "{}\n{}\n".format(msg, msg1)
        msg = "{}{}{}".format(msg0,msg, msg1)
        return msg

    def initOperatingTimes(self)->dict:
        d_tmp = {}
        d_tmp[UNDF]=0
        if self.desc['type'] == DIESEL:

            for item in DIESEL_STATES:
                d_tmp[item]=0
        elif  self.desc['type']==HYDR_TRB:
            for item in HYDR_TRB_STATES:
                d_tmp[item]=0
        elif self.desc['type'] == HYDR_PUMP:
            for item in HYDR_TRB_STATES:
                d_tmp[item] = 0
        else:
            pass
        return d_tmp



    def dumpDesc(self, file_desc):
        with open(file_desc, 'wb') as fw:
            dump(self.desc, fw)
        message = '\nDescriptor "type": {} "Id": {} saved in {}\n'.format(self.desc["type"],self.desc["Id"],file_desc)
        msg2log(None, message, self.f)

    def loadDesc(self,file_desc):
        self.desc={}
        with open(file_desc, 'rb') as fr:
            self.desc = load(fr)
        message = '\nDescriptor "type": {} "Id": {} loaded from {}\n'.format(self.desc["type"], self.desc["Id"], file_desc)
        msg2log(None, message, self.f)


    def setDesc(self, type: str = None, Id: int = None,Priority: int = None, QoS1: int = None, QoS2: int =None,
                QoS3: int = None, CurrentState: int = None, CurrentPower: float = None,CreditTokens:dict = None,
                NumStarts: int = 0,TotalOperationTime: int =0, UpdateOperatingTimes:int = None, LastStatePeriod: int =0,
                Color:int=None):
        msg =""
        message="\n\nDescriptor updates:\n"

        try:
            if type is not None:
                self.desc["type"]=type
                message=message +'desc["type"]: {}\n'.format(type)
            if Id is not None:
                self.desc["Id"]=Id
                message = message + 'desc["Id"]: {}\n'.format(Id)
            if Priority is not None:
                self.desc["Priority"]=Priority
                message = message + 'desc["Priority"]: {}\n'.format(Priority)
            if QoS1 is not None:
                self.desc["QoS1"]=QoS1
                message = message + 'desc["QoS1"]: {}\n'.format(QoS1)
            if QoS2 is not None:
                self.desc["QoS2"]=QoS2
                message = message + 'desc["QoS2"]: {}\n'.format(QoS2)
            if QoS3 is not None:
                self.desc["QoS3"]=QoS3
                message = message + 'desc["QoS3"]: {}\n'.format(QoS3)
            if CurrentState is not None:
                prevState=self.desc["CurrentState"]
                self.desc["CurrentState"]=CurrentState
                self.desc["PreviousState"]=prevState

                self.desc["LastStatePeriod"]=self.desc["LastStatePeriod"]+1
                self.desc["OperatingTimes"][prevState]=self.desc["LastStatePeriod"]
                self.desc["LastStatePeriod"] = 0
                if prevState==S_OFF and CurrentState!=S_OFF:
                    self.desc["NumStarts"] = NumStarts + self.desc["NumStarts"]
                if self.desc["CurrentState"]==S_OFF:
                    self.desc["CurrentPower"] =0.0

                message = message + 'desc["CurrentState"]: {}\ndesc["PreviousState"]: {}\n'.format(CurrentState, prevState)
                message = message + 'desc["CurrentPower"]: {}\n'.format(self.desc["CurrentPower"])

            if CreditTokens is not None:
                self.desc["CreditTokens"]=CreditTokens
                message = message + 'desc["CreditTokens"]: {}\n'.format(CreditTokens)
            if UpdateOperatingTimes is not None:  # TODO
                #self.desc["OperatingTimes"]=copy.copy(OperatingTimes)
                pass
            if Color is not None:
                self.desc["Color"]=Color
                message = message + 'desc["Color"]: {}\n'.format(Color)
            if CurrentPower is not None:
                self.desc["CurrentPower"] = CurrentPower
                message = message + 'desc["CurrentPower"]: {}\n'.format(self.desc["CurrentPower"])

            self.desc["NumStarts"]=NumStarts + self.desc["NumStarts"]
            message = message + 'desc["NumStarts"]: {}\n'.format(self.desc["NumStarts"])
            self.desc["TotalOperationTime"]=TotalOperationTime + self.desc["TotalOperationTime"]
            message = message + 'desc["TotalOperationTime"]: {}\n'.format(self.desc["TotalOperationTime"])
            self.desc["LastStatePeriod"] = LastStatePeriod + self.desc["LastStatePeriod"]
            message = message + 'desc["LastStatePeriod"]: {}\n'.format(self.desc["LastStatePeriod"])

            if self.desc["OperatingTimes"] is not None:
                st = self.desc["CurrentState"]
                self.desc["OperatingTimes"][st] = self.desc["OperatingTimes"][st] + 1
                message = message + 'desc["OperatingTimes"]: {}\n'.format(self.desc["OperatingTimes"])

            msg2log(None, message,self.f)
            message=""
        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))

        except:
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())

        finally:
            if len(msg) > 0:
                msg2log(type(self).__name__, msg, D_LOGS["except"])
            if len(message) > 0:
                msg2log(type(self).__name__, message, self.f)
        return

""" Class SmartReq models a request for decrease or increase unit power"""

class SmartReq:

    def __init__(self,id:int,typeReq:int,typeDer:int, model:str, token:int, descr:Descriptor, f: object =None):
        self.id=id
        self.typeReq=typeReq
        self.typeDer=typeDer
        self.model = model
        self.token=token
        self.descr=descr
        self.f=f
        return



