#!/usr/bin/python3

from math import floor,ceil
import numpy as np


from simTM.descriptor import Descriptor,SmartReq
from predictor.utility import msg2log

# Request type
from simTM.cfg import DEC_PWR, INC_PWR
# A possible state of the units of DER
from simTM.cfg import S_OFF,S_LPWR, S_MPWR,S_HPWR,DIESEL_STATES_NAMES
# Types DER
from simTM.cfg import DIESEL, PV, CHP, WIND_TRB, HYDR_TRB, HYDR_PUMP, DER_NAMES
# Priorites
from simTM.cfg import PR_L, PR_M, PR_H

from simTM.cfg import DELTAPWR,AGE

from simTM.api import D_PRIORITY, randStates,logStep,state2history,history2log,logsequences,plotsequences


"""Base class Power Source for DER distributed energy resources"""

class Powersrc:
    _id = 0

    def __init__(self, typeDer,model,f: object = None):
        self.f = f
        self.model=model
        self.typeDer =typeDer

    @staticmethod
    def allocID()-> int :
        id =Powersrc._id
        Powersrc._id+=1
        return id


"""Electricity network """


class ElNet(Powersrc):
    def __init__(self, typeDer, model, f: object = None):

        super().__init__(typeDer,model,f)


"""Heat network"""


class HeatNet(Powersrc):
    def __init__(self,typeDer,model,f: object = None):
        self.descr={}
        super().__init__(typeDer, model,f)


"""micro combined, heat and power"""


class CHP(ElNet):
    def __init__(self,typeDer,model,f: object = None):
        super().__init__(typeDer,model,f)


"""photovoltaics"""


class PV(ElNet):
    def __init__(self,typeDer,model,f: object = None):
        super().__init__(typeDer,model,f)


""" Diesel"""


class Diesel(ElNet):
    _dposstates = {S_OFF: "Off", S_LPWR: "LowPower", S_MPWR: "MidPower", S_HPWR: "HighPower"}

    _pwrModel={"VPP1250":{S_OFF:0.0,S_LPWR:0.2,S_MPWR:0.6,S_HPWR:1.0},
               "VPP590":{S_OFF:0.0,S_LPWR:0.1,S_MPWR:0.3,S_HPWR:0.5}}

    def __init__(self,  model,f: object = None):

        id=Powersrc.allocID()
        CurrentState, st_name = randStates(Diesel.getStates())
        Priority,pr_name = randStates(D_PRIORITY)
        self.descr=Descriptor(DIESEL,id,CurrentState=CurrentState,Priority=Priority,f = None)

        self.incPowerReq = None
        self.decPowerReq = None
        super().__init__(DIESEL,model,f)
        self.updateDescr(mode = 'init')
        pass

    @staticmethod
    def getStates():
        return Diesel._dposstates

    @staticmethod
    def getStateLimits(model,state,f:object = None)->(float,float):
        msg=""
        h_limit=0.0
        l_limit=0.0
        try:
            h_limit = Diesel._pwrModel[model][state]
            if state>0:
                l_limit = Diesel._pwrModel[model][state-1]
        except:
            msg="Incorrect {} .No match\n".format(model)
            h_limit=0.0
            l_limit=0.0
        finally:
            msg2log(None,msg,f)
        return h_limit,l_limit

    @staticmethod
    def getStatePerPwr(model:str,pwr:float,f:object=None):
        dstpwr = Diesel._pwrModel[model]

        for key,val in dstpwr.items():
            st=key
            if pwr>=val:
                continue
            st=key-1
            break
        return st

    def createSmartRequest(self, imbPower:float)->int :

        self.incPowerReq=None
        self.decPowerReq = None
        if imbPower==0:
            return 0

        # calcs the credits according by Current Power for increment up next state or decrement down previous state.
        self.updateDescr(mode='smart')
        CurrentPower = self.descr.desc["CurrentPower"]
        CurrentPowerToken=floor(CurrentPower/DELTAPWR)
        max_pwr, max_tokens = self.getMaxPower()
        if imbPower>0.0:  # proficit generation
            token = self.descr.desc["CreditTokens"]['dec']
            if CurrentPower==0.0:
                token=0
            if token>0 :
                self.decPowerReq = SmartReq(self.descr.desc["Id"], DEC_PWR, self.typeDer, self.model, token, self.descr)
        elif imbPower<0.0:
            token = self.descr.desc["CreditTokens"]['inc']
            if CurrentPowerToken>=max_tokens:
                token=0
            if token>0:
                self.incPowerReq = SmartReq(self.descr.desc["Id"], INC_PWR, self.typeDer, self.model, token, self.descr)

        else:
            token=0
        return token

    """ Updates descriptor according by mode:
    'init' - sets CurrentPower field  at creating flow
    'smart' - calc CreditTokens {'inc':x,'dec':y} , create the smart request for Inc and Dec power
    'inTM' - set CreditTokens when descriptor comes in TM
    'outTM'- set fields when descriptor leaves TM
    'discard' -set fields when descriptor discarder
    """
    def updateDescr(self,mode:str='init',CreditTokens:dict= {'inc':0,'dec':0}, CurrentState:int = None,
                    CurrentPower:float = None, UtilizedTokens:dict={'inc':0,'dec':0}):
        ds=self.descr
        msg=""
        h_pwr, l_pwr = Diesel.getStateLimits(self.model, ds.desc["CurrentState"])
        if mode=='init':
            ds.setDesc(CurrentPower=(h_pwr - l_pwr) / 2.0)
        elif mode =='smart':
            self.calcCreditTokens()

        elif mode=='inTM':
            pass

        elif mode=='outTM':
            CreditTokens=ds.desc["CreditTokens"]
            deltaTokens=CreditTokens['inc'] - UtilizedTokens['inc']
            CreditTokens['inc']=deltaTokens if deltaTokens>0 else 0
            deltaTokens = CreditTokens['dec'] - UtilizedTokens['dec']
            CreditTokens['dec'] = deltaTokens if deltaTokens > 0 else 0

            CurrentPower=ds.desc["CurrentPower"]
            pwr=CurrentPower + (UtilizedTokens['inc']-UtilizedTokens['dec'])*DELTAPWR
            st = Diesel.getStatePerPwr(self.model, pwr, f = self.f)
            if st is not None:
                ds.setDesc(CurrentState=st,CurrentPower=pwr,CreditTokens=CreditTokens)
            else:
                msg = msg + "Mode: {}. Invalid set of 'CurrentState'\n".format(mode)
                ds.setDesc( CurrentPower=pwr, CreditTokens=CreditTokens)


        elif mode=='discard':
            pass
        msg2log(None,msg,self.f)
        return

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
        ds=self.descr
        dstates=Diesel.getStates()
        last_st=len(dstates)-1
        first_st =0
        st=ds.desc["CurrentState"]
        pwr = ds.desc["CurrentPower"]
        h_pwr=np.zeros((3),dtype=float)
        l_pwr = np.zeros((3), dtype=float)
        h_pwr[0], l_pwr[0] = Diesel.getStateLimits(self.model, first_st)
        h_pwr[1], l_pwr[1] = Diesel.getStateLimits(self.model, st)
        next_st=st+1 if st<last_st else last_st
        h_pwr[2], l_pwr[2] = Diesel.getStateLimits(self.model, next_st)
        inc=floor((h_pwr[2]-pwr)/DELTAPWR) if st<last_st else floor((h_pwr[1]-pwr)/DELTAPWR)
        dec=ceil((pwr -l_pwr[0])/DELTAPWR) if st>first_st else ceil(pwr/DELTAPWR)
        ds.setDesc(CreditTokens={'inc':inc,'dec':dec})

        return

    def getMaxPower(self)->(float,int):
        ds = self.descr
        dstates = Diesel.getStates()
        last_st = len(dstates) - 1
        h_pwr_max, l_pwr = Diesel.getStateLimits(self.model, last_st)
        return h_pwr_max, floor(h_pwr_max/DELTAPWR)



if __name__ == "__main__":

    a = Diesel("VPP1250",f=None)
