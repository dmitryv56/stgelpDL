#!/usr/bin/python3

import sys

from math import floor,ceil
import numpy as np

from simTM.descriptor import Descriptor,SmartReq
from predictor.utility import msg2log
# Request type
from simTM.cfg import D_LOGS,DEC_PWR, INC_PWR
# A possible state of the units of DER
from simTM.cfg import S_OFF,S_LPWR, S_MPWR,S_HPWR,DIESEL_STATES_NAMES, S_ON, HYDR_TRB_STATES,HYDR_TRB_STATES_NAMES,\
    HYDR_PUMP_STATES,HYDR_PUMP_STATES_NAMES,D_LOGS
from simTM.cfg import DIESEL_NUMBER, DIESEL_MODEL, MAX_CUSTOM, MAX_GENER,DELTAPWR, SIM_PERIOD,AGE,\
    HYDR_PUMP_NUMBER, HYDR_PUMP_MODEL, HYDR_TRB_NUMBER,HYDR_TRB_MODEL
# Types DER
from simTM.cfg import DIESEL, PV, CHP, WIND_TRB, HYDR_TRB, HYDR_PUMP, DER_NAMES
# Priorites
from simTM.cfg import PR_L, PR_M, PR_H

from simTM.cfg import DELTAPWR,AGE

from simTM.api import D_PRIORITY, randStates,logStep,state2history,history2log,logsequences,plotsequences


"""Base class Power Source for DER distributed energy resources"""

class Powersrc:
    _id = 0
    _initDownTrb = 0
    _initDownPump = 0

    def __init__(self, id:int, typeDer:int,model:str,CurrentState:int, Priority:int, descr:Descriptor,f: object = None):
        self.f = f
        self.model=model
        self.typeDer =typeDer
        self.incPowerReq = None
        self.decPowerReq = None
        self.id = id
        self.descr = descr
        self.updateDescr(mode='init')
        self.updateDescr(mode='smart')
        return

    def __str__(self):
        st=self.descr.desc["CurrentState"]
        prio = self.descr.desc["Priority"]
        name=DER_NAMES[self.typeDer]
        message =f"""Id: {self.id} Type: {self.typeDer} {name}  Model: {self.model} State: {st} Priority {prio}
Increment Request: {self.incPowerReq}
Decrement Request: {self.decPowerReq}"""
        return message

    @staticmethod
    def allocID()-> int :
        id =Powersrc._id
        Powersrc._id+=1
        return id

    """ Updates descriptor according by mode:
       'init' - sets CurrentPower field  when a flow is created
       'smart' - calcs CreditTokens {'inc':x,'dec':y} , create the smart request for Inc and Dec power
       'inTM' - sets CreditTokens when the descriptor enters in TM
       'outTM'- sets fields when the descriptor leaves TM
       'discard' -sets fields when the descriptor is discarded
       """

    def updateDescr(self, mode: str = 'init', CreditTokens: dict = {'inc': 0, 'dec': 0}, CurrentState: int = None,
                    CurrentPower: float = None, UtilizedTokens: dict = {'inc': 0, 'dec': 0}):
        ds = self.descr
        msg  = ""
        msg1 = ""
        try:
            h_pwr, l_pwr = self.getStateLimits(self.model, ds.desc["CurrentState"])   # self.getStateLimits4DER(ds.desc["CurrentState"]) # Diesel.getStateLimits(self.model, ds.desc["CurrentState"])
            if mode == 'init':
                if self.typeDer==DIESEL:
                    ds.setDesc(CurrentPower=(h_pwr - l_pwr) / 2.0)
                elif self.typeDer==HYDR_PUMP:
                    ds.setDesc(CurrentPower=h_pwr)
                elif self.typeDer==HYDR_TRB:
                    ds.setDesc(CurrentPower=h_pwr)
            elif mode == 'smart':
                try:
                    self.calcCreditTokens()
                    print("I am here!")
                except:
                    msg2log("smart",sys.exc_info(),D_LOGS["except"])
                finally:
                    pass
            elif mode == 'inTM':
                pass

            elif mode == 'outTM':
                CreditTokens = ds.desc["CreditTokens"]
                deltaTokens = CreditTokens['inc'] - UtilizedTokens['inc']
                CreditTokens['inc'] = deltaTokens if deltaTokens > 0 else 0
                deltaTokens = CreditTokens['dec'] - UtilizedTokens['dec']
                CreditTokens['dec'] = deltaTokens if deltaTokens > 0 else 0

                CurrentPower = ds.desc["CurrentPower"]
                if self.typeDer==DIESEL or self.typeDer==HYDR_TRB:
                    pwr = CurrentPower + (UtilizedTokens['inc'] - UtilizedTokens['dec']) * DELTAPWR
                elif self.typeDer == HYDR_PUMP:
                    pwr = CurrentPower + (UtilizedTokens['dec'] - UtilizedTokens['inc']) * DELTAPWR
                    if pwr<0.0: pwr=0.0
                    if pwr>0.0: pwr,_= self.getMaxPower()
                st = self.getStatePerPwr(self.model, pwr) # st = Diesel.getStatePerPwr(self.model, pwr, f=self.f)

                if st is not None:
                    ds.setDesc(CurrentState=st, CurrentPower=pwr, CreditTokens=CreditTokens)
                else:
                    msg = msg + "Mode: {}. Invalid set of 'CurrentState'\n".format(mode)
                    ds.setDesc(CurrentPower=pwr, CreditTokens=CreditTokens)

                ds.setDesc(UpdateOperatingTimes=1)

            elif mode == 'discard':
                ds.setDesc(UpdateOperatingTimes=1)

            if len(msg)>0:
                msg2log("UpdateDescr", msg, self.f)
                msg=""
        except KeyError as e:
            msg1 = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg1 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
        finally:
            if len(msg1)>0:
                msg2log(type(self).__name__, msg1, D_LOGS["except"])
            if len(msg)>0:
                msg2log(type(self).__name__, msg, self.f)
        return

    def getMaxPower(self)->(float,int):
        ds = self.descr

        dstates = type(self).getStates() # call statmethod dstates = self.getStates4DER() #Diesel.getStates()

        last_st = len(dstates) - 1
        h_pwr_max, l_pwr = self.getStateLimits(self.model, last_st)  # Diesel.getStateLimits(self.model, last_st)
        return h_pwr_max, floor(h_pwr_max/DELTAPWR)

    def getStatePerPwr(self, model:str,pwr:float):
        dstpwr = type(self)._pwrModel[model]  #Diesel._pwrModel[model]

        for key,val in dstpwr.items():
            st=key
            if pwr>=val:
                continue
            st=key-1
            break
        return st
    """ This method calculates Credit Tokens for each device . (may be for all types of DER?)
          The device is in a certain state {i}, which is set by the upper  and lower power limits {h_pwr[i],l_pwr[i]. 
          In this state the device generates a power X[i],   l_pwr[i]<X[i]<=h_pwr[i].
          Inc-credit. The device can increase the generated power till h_pwr[i+1]. If i- last state, then till h_pwr[i].
                  Inc_credit= floor((h_pwr[i+1]-X[i])/deltaPwr), or Inc_credit=floor((h_pwr[i]-X[i])/deltaPwr),
          where deltaPwr - power resolution, i.e 0.1Mwt, 
          'floor(X)'  is the greatest integer less than or equal to X.
          Dec-credit: the device may decrease the generated power till l_pwr[i-1] or 0 if i-first state.
                  Dec_credit =ceil((X[i]-l_pwr[i-1])/deltaPwr) or Dec_credit=ceil(X[i]/deltaPwr)
         where 'ceil(X)' is the least integer greater to X.
      """

    def calcCreditTokens(self):
        ds = self.descr
        msg=""
        try:
            dstates = type(self).getStates()  # call staticmethod dstates = self.getStates4DER()  # =Diesel.getStates()
            last_st = len(dstates) - 1
            first_st = 0
            st = ds.desc["CurrentState"]
            pwr = ds.desc["CurrentPower"]
            h_pwr = np.zeros((3), dtype=float)
            l_pwr = np.zeros((3), dtype=float)
            h_pwr[0], l_pwr[0] = self.getStateLimits(self.model,first_st) # self.getStateLimits4DER(state=first_st)  # Diesel.getStateLimits(self.model, first_st)
            h_pwr[1], l_pwr[1] = self.getStateLimits(self.model,st)       # self.getStateLimits4DER(state=st)  # Diesel.getStateLimits(self.model, st)
            next_st = st + 1 if st < last_st else last_st
            h_pwr[2], l_pwr[2] = self.getStateLimits(self.model, next_st)  # self.getStateLimits4DER(state=next_st)  # Diesel.getStateLimits(self.model, next_st)
            inc = floor((h_pwr[2] - pwr) / DELTAPWR) if st < last_st else floor((h_pwr[1] - pwr) / DELTAPWR)
            dec = ceil((pwr - l_pwr[0]) / DELTAPWR) if st > first_st else ceil(pwr / DELTAPWR)
            ds.setDesc(CreditTokens={'inc': inc, 'dec': dec})
        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg = "O-o-ops! Unexpected error: {}".format( sys.exc_info()[0])
        finally:
            if len(msg)>0:
                msg2log(type(self).__name__, msg,f = D_LOGS["except"])
        return

    def createSmartRequest(self, imbPower: float) -> int:

        self.incPowerReq = None
        self.decPowerReq = None
        if imbPower == 0:
            return 0
        token = 0
        msg=""
        # calcs the credits according by Current Power for increment up next state or decrement down previous state.
        try :
            self.updateDescr(mode='smart')
            CurrentPower = self.descr.desc["CurrentPower"]
            CurrentPowerToken = floor(CurrentPower / DELTAPWR)
            max_pwr, max_tokens = self.getMaxPower()
            if imbPower > 0.0:  # proficit generation
                token = self.descr.desc["CreditTokens"]['dec']
                if CurrentPower == 0.0:
                    token = 0
                if token > 0:
                    self.decPowerReq = SmartReq(self.descr.desc["Id"], DEC_PWR, self.typeDer, self.model, token, self.descr)
            elif imbPower < 0.0:
                token = self.descr.desc["CreditTokens"]['inc']
                if CurrentPowerToken >= max_tokens:
                    token = 0
                if token > 0:
                    self.incPowerReq = SmartReq(self.descr.desc["Id"], INC_PWR, self.typeDer, self.model, token, self.descr)

            else:
                token = 0
        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info()[0])
        finally:
            if len(msg)>0:
                msg2log(type(self).__name__, msg, D_LOGS["except"])
        return token

    def getStateLimits(self,model,state)->(float,float):
        msg=""
        h_limit=0.0
        l_limit=0.0
        try:
            # h_limit = Diesel._pwrModel[model][state]
            h_limit = type(self)._pwrModel[model][state]
            if state>0:
                # l_limit = Diesel._pwrModel[model][state-1]
                l_limit = type(self)._pwrModel[model][state - 1]
        except:
            msg="Incorrect {} .No match\n".format(model)
            h_limit=0.0
            l_limit=0.0
        finally:
            if len(msg)>0:
                msg2log(None,msg,D_LOGS["except"])
        return h_limit,l_limit


"""Electricity network """


class ElNet(Powersrc):
    def __init__(self, id,typeDer, model, CurrentState, Priority,descr,f: object = None):

        super().__init__(id,typeDer,model,CurrentState, Priority, descr,f)




"""Heat network"""


class HeatNet(Powersrc):
    def __init__(self,id,typeDer,model,CurrentState, Priority,descr,f: object = None):
        self.descr={}
        super().__init__(id,typeDer, model,CurrentState, Priority,descr,f)
    """Specific : two states for Pumps
                  Off    On
        DecPwr    max_pwr    0
        IncPow      0    max_pwr
    """

    def calcCreditTokens(self):
        ds = self.descr
        msg =""
        try:
            dstates = type(self).getStates()  # call staticmethod dstates = self.getStates4DER()  # =Diesel.getStates()
            last_st = len(dstates) - 1
            first_st = 0
            st = ds.desc["CurrentState"]
            pwr = ds.desc["CurrentPower"]
            h_pwr = np.zeros((3), dtype=float)
            l_pwr = np.zeros((3), dtype=float)
            h_pwr[0], l_pwr[0] = self.getStateLimits(self.model,first_st) # self.getStateLimits4DER(state=first_st)  # Diesel.getStateLimits(self.model, first_st)
            h_pwr[1], l_pwr[1] = self.getStateLimits(self.model,st)       # self.getStateLimits4DER(state=st)  # Diesel.getStateLimits(self.model, st)
            next_st = st + 1 if st < last_st else last_st
            h_pwr[2], l_pwr[2] = self.getStateLimits(self.model, next_st)  # self.getStateLimits4DER(state=next_st)  # Diesel.getStateLimits(self.model, next_st)

            dec = floor((h_pwr[2] - pwr) / DELTAPWR) if st==first_st else 0  # pump ups
            inc = floor((pwr - h_pwr[0]) / DELTAPWR) if st==last_st else 0   # pump downs
            # inc = floor((h_pwr[2] - pwr) / DELTAPWR) if st < last_st else floor((h_pwr[1] - pwr) / DELTAPWR)
            # dec = ceil((pwr - l_pwr[0]) / DELTAPWR) if st > first_st else ceil(pwr / DELTAPWR)
            ds.setDesc(CreditTokens={'inc': inc, 'dec': dec})
        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg = "O-o-ops! Unexpected error: {}".format( sys.exc_info()[0])
        finally:
            if len(msg)>0:
                msg2log(type(self).__name__, msg,f = D_LOGS["except"])
        return
    """ specific for Pumps """
    def createSmartRequest(self, imbPower: float) -> int:

        self.incPowerReq = None
        self.decPowerReq = None
        if imbPower == 0:
            return 0
        tjken = 0
        msg = ""
        # calcs the credits according by Current Power for increment up next state or decrement down previous state.

        try:
            self.updateDescr(mode='smart')
            CurrentPower = self.descr.desc["CurrentPower"]
            CurrentPowerToken = floor(CurrentPower / DELTAPWR)
            max_pwr, max_tokens = self.getMaxPower()
            if imbPower > 0.0:  # proficit generation, the pump consumes energy to raise water into the pool.
                # if self.descr.desc["CurrentState"]==S_OFF:  #up the pump
                #     token=max_tokens
                # elif self.descr.desc["CurrentState"]==S_ON:
                #     token = 0 #self.descr.desc["CreditTokens"]['dec']
                token=self.descr.desc["CreditTokens"]["dec"]
                if token > 0:
                    self.decPowerReq = SmartReq(self.descr.desc["Id"], DEC_PWR, self.typeDer, self.model, token, self.descr)
            elif imbPower < 0.0: # deficit generation. the pump should be down in order to increment generation
                # if self.descr.desc["CurrentState"]==S_OFF: #if np pump
                #     token=0
                # elif self.descr.desc["CurrentState"]==S_ON:# down the Pump
                #     token = max_tokens
                token = self.descr.desc["CreditTokens"]["inc"]
                if token > 0:
                    self.incPowerReq = SmartReq(self.descr.desc["Id"], INC_PWR, self.typeDer, self.model, token, self.descr)

            else:
                token = 0
        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info()[0])
        finally:
            if len(msg)>0:
                msg2log(type(self).__name__, msg, D_LOGS["except"])

        return token
"""micro combined, heat and power"""


class CHP(ElNet):
    def __init__(self,model,f: object = None):
        id = Powersrc.allocID()
        typeDer = CHP
        CurrentState = 0
        Priority = 0
        descr = Descriptor(typeDer, id, CurrentState=CurrentState, Priority=Priority, f=f)
        super().__init__(id, typeDer, model, CurrentState, Priority, descr, f)


"""photovoltaics"""


class PV(ElNet):
    def __init__(self,model,f: object = None):
        id = Powersrc.allocID()
        typeDer=PV
        CurrentState=0
        Priority=0
        descr=Descriptor(typeDer, id, CurrentState=CurrentState, Priority=Priority, f=f)
        super().__init__(id, typeDer, model, CurrentState, Priority, descr, f)

""" Hydro Turbine"""

class HydrTrb(ElNet):
    _dposstates = {S_OFF: "Off", S_ON: "On"}
    _pwrModel = {"HydrTrb": {S_OFF: 0.0, S_ON:  1.0}}

    def __init__(self,  model,f: object = None):


        CurrentState, st_name = randStates(HydrTrb.getStates())
        CurrentState=S_OFF
        st_name=type(self)._dposstates[CurrentState]
        Priority, pr_name = randStates(D_PRIORITY)
        id = Powersrc.allocID()
        descr = Descriptor(HYDR_TRB, id, CurrentState=CurrentState, Priority=Priority, f=f)

        super().__init__(id, HYDR_TRB, model, CurrentState, Priority, descr, f)
        self.updateDescr(mode='init')

    @staticmethod
    def getStates():
        return HydrTrb._dposstates


    """ Specific : two states for Turbine
                  Off    On
        DecPwr    0      max_pwr
        IncPow   max_pwr    0
    """



    def calcCreditTokens(self):
        ds = self.descr
        dstates = type(self).getStates()
        last_st = len(dstates) - 1
        first_st = 0
        st = ds.desc["CurrentState"]
        pwr = ds.desc["CurrentPower"]
        h_pwr = np.zeros((3), dtype=float)
        l_pwr = np.zeros((3), dtype=float)
        h_pwr[0], l_pwr[0] = self.getStateLimits(self.model,first_st)
        h_pwr[1], l_pwr[1] = self.getStateLimits(self.model,st)
        next_st = st + 1 if st < last_st else last_st
        h_pwr[2], l_pwr[2] = self.getStateLimits(self.model, next_st)  #
        dec = floor((pwr - h_pwr[0]) / DELTAPWR) if st == last_st else 0  # trb downs
        inc = floor((h_pwr[2] - pwr) / DELTAPWR) if st == first_st else 0  # trb ups

        ds.setDesc(CreditTokens={'inc': inc, 'dec': dec})

        return
""" Diesel"""

class Diesel(ElNet):
    _dposstates = {S_OFF: "Off", S_LPWR: "LowPower", S_MPWR: "MidPower", S_HPWR: "HighPower"}

    _pwrModel={"VPP1250":{S_OFF:0.0,S_LPWR:0.2,S_MPWR:0.6,S_HPWR:1.0},
               "VPP590":{S_OFF:0.0,S_LPWR:0.1,S_MPWR:0.3,S_HPWR:0.5}}

    def __init__(self,  model,f: object = None):

        CurrentState, st_name = randStates(Diesel.getStates())
        Priority,pr_name = randStates(D_PRIORITY)
        Priority=PR_H
        pr_name=D_PRIORITY[PR_H]
        id = Powersrc.allocID()
        descr = Descriptor(DIESEL, id, CurrentState=CurrentState, Priority=Priority, f=f)

        super().__init__(id,DIESEL,model,CurrentState, Priority,descr, f)
        self.updateDescr(mode = 'init')
        self.updateDescr(mode='smart')

    @staticmethod
    def getStates():
        return Diesel._dposstates


    #
    # @staticmethod
    # def getStatePerPwr(model:str,pwr:float,f:object=None):
    #     dstpwr = Diesel._pwrModel[model]
    #
    #     for key,val in dstpwr.items():
    #         st=key
    #         if pwr>=val:
    #             continue
    #         st=key-1
    #         break
    #     return st

class HydrPump(HeatNet):
    _dposstates = {S_OFF: "Off", S_ON: "On"}
    _pwrModel = {"HydrPump": {S_OFF: 0.0, S_ON:  0.5}}

    def __init__(self,  model,f: object = None):


        CurrentState, st_name = randStates(HydrPump.getStates())
        CurrentState=S_OFF
        st_name=type(self)._dposstates[CurrentState]
        Priority, pr_name = randStates(D_PRIORITY)
        Priority=PR_L
        pr_name =D_PRIORITY[PR_L]
        id = Powersrc.allocID()
        descr = Descriptor(HYDR_PUMP, id, CurrentState=CurrentState, Priority=Priority, f=f)

        super().__init__(id,HYDR_PUMP, model, CurrentState, Priority,descr,f)
        # self.updateDescr(mode='init')
        # self.updateDescr(mode='smart')

    @staticmethod
    def getStates():
        return HydrPump._dposstates

""" API for Distributed Energy Resources """
def reportGenset( mode:str='init', lsGenset:list=[], step:int=0):
    f=D_LOGS['init'] if mode=='init' else D_LOGS['current']
    message="\nStep {}".format(step)
    msg2log(None,message,f)

    for item in lsGenset:
        msg=""
        try:
            msg2log(None,item.__str__(),f)
        except:
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info()[0])

        finally:
            if len(msg) > 0:
                msg2log("reportGenset", msg, D_LOGS["except"])
        msg2log(None,item.descr.__str__(),f)
    return

def dieselGenset(f:object=None)->list:
    dieselgenset=[]
    for i in range(DIESEL_NUMBER):
        dieselgenset.append(Diesel(DIESEL_MODEL[i%len(DIESEL_MODEL)],f))
    reportGenset(lsGenset= dieselgenset)
    return dieselgenset

def hydrpumpGenset(f:object=None)->list:
    hydrpumpgenset=[]
    for i in range(HYDR_PUMP_NUMBER):
        hydrpumpgenset.append(HydrPump(HYDR_PUMP_MODEL[i%len(HYDR_PUMP_MODEL)],f))
    reportGenset(lsGenset=hydrpumpgenset)
    return hydrpumpgenset

def hydrtrbGenset(f:object=None)->list:
    hydrtrbgenset=[]
    for i in range(HYDR_TRB_NUMBER):
        hydrtrbgenset.append(HydrTrb(HYDR_TRB_MODEL[i%len(HYDR_TRB_MODEL)],f))
    reportGenset(lsGenset=hydrtrbgenset)
    return hydrtrbgenset

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
            if len(msg)>0:
                msg2log("dieselGensetRequests", msg, D_LOGS["except"])
    return smartreqs
""" This function creates a dict {priority:list of requests """
def derGensetRequests(tm_sets:dict, imbPwr:float, f:object=None)->dict:
    smartreqs={}
    for typeDer,genset in tm_sets.items():
        for device in genset:
            msg=""
            try:
                device.incPowerReq=None
                device.decPowerReq=None
                token = device.createSmartRequest(imbPwr)
                priority=device.descr.desc["Priority"]
                if priority not in smartreqs:
                    smartreqs[priority] = []
                if token==0:   # do not create request for this diesel
                    continue
                if device.incPowerReq is not None:
                    smartreqs[priority].append(device.incPowerReq)
                if device.decPowerReq is not None:
                        smartreqs[priority].append(device.decPowerReq)
            except KeyError as e:
                msg = "O-o-ops! For {} type DER with {} id  I got a KeyError - reason  {}".format(typeDer, device.id,str(e))
            except:
                msg = "O-o-ops! For {} type DER with {} id  I got an Unexpected error: {}".format(typeDer, device.id, sys.exc_info())

            finally:
                if len(msg)>0:
                    msg2log("derGensetRequests", msg, D_LOGS["except"])
    return smartreqs

if __name__ == "__main__":

    a = Diesel("VPP1250",f=None)
