#!/usr.bin/python3

"""
The policer rules description.

Each mechanism has its own characteristics in dynamic modes. For example. When we turn on the pump, first of all, any
electric motor has its own starting current. Then the current decreases. This gives us starting power and rated power.
Likewise, any other electrically driven mechanism. Further, when switching from one power level, there is also a certain
time delay. It is impossible to jump from one power level to another. This applies more to diesels and turbines.
Each mechanism has an optimal mode of operation (power) and suboptimal when the efficiency drops sharply.

The data of these mechanisms can be obtained only from statistical analysis of time series for each mechanism separately.
I tried to explain it to you. Let me give you one example. If the pumps and the hydraulic turbine would operate in full
nominal mode, their total efficiency would be 72-75%. This is taken from the technical characteristics of these
mechanisms But average annual data show that the average efficiency is 42-45%. Therefore, if we expand the time series
for a year (expand the series - break down by power levels. How long the mechanisms worked at the level of 25%, 50%,
75% and 100%). Then, assuming that the efficiency graph is described by a quadratic equation , you can calculate the
coefficients of this equation. This will be approximately, since dynamic processes are also superimposed. The operation
of all mechanisms is considered only from the standpoint of their operation at the nominal mode.

Wind turbines and solar panels are not considered. We cannot control them. We will consider the imbalance between the
generation power of wind turbines (or solar panels) and the power consumption. It is the difference between generation
and consumption that seems interesting.
Why the difference between the real power of consumers and the programmed power of generation is considered. This is
a special case when there is no wind and sun. Only diesel engines work. The programmed generation power is planned in
such a way that diesel engines work only at optimum operating conditions. However, in practice, diesel engines monitor
load fluctuations in the network and thus fuel consumption increases by 5-10%. Ideally, it would be more efficient for
the diesels to work strictly according to the planned schedule, and all deviations would be reset to some balancing
device. For the time being, for simplicity, we will call this device a storage battery, which takes on excess electrical
energy, and in case of a lack of diesel power, it discharges. In theory, a battery can operate in this mode, but
frequent charge and discharge cycles will quickly wear out. Plus it is very expensive. But at this point in time, this
is a valid assumption. Consider that the battery is working with the diesel engine. Moreover, there is an option to
organize such work. The essence of this idea is as follows. There are about 100 individual batteries in a battery
container. If diesel engines operate with some excess power, then the process of charging one specific battery is
constantly going on. Then he switches to charging another, and so on in turn. When a power shortage occurs, previously
charged batteries are discharged. The question of the forecast is just needed in order to determine how many batteries
are charged in parallel and how many are discharged. Thus, the number of cycles can be reduced.

The forecast requires the difference between generation and consumption. It can be the difference between the power of
wind generators or solar panels (or their sum) and the real power of consumers.

All possible options should be divided into four ranges:
0.The zero range is when the power of wind generators (hereinafter, under the power of wind generators,
We will indicate the power of solar panels or the sum of wind generators and solar panels, that is, the power
of renewable energy sources) significantly exceeds the power of consumers. We have a lot of excess energy. In this case,
the pumps are switched on. At the same time, the minimum pump power is 0.5 MW. Therefore, the difference between
the power of wind generators and the power of consumers should be more than 0.5 MW.

1.The first range is excess power less than 0.5 MW. All this power is discharged to the storage battery (a kind of
balancing device).

2.The second range is that the excess power is small and there is a risk of failure. Those the power of wind generators
drops sharply below the power level of consumers. In this case, either a diesel engine or a hydraulic turbine must be
started.

3.The third range - the power of wind generators is significantly lower than the power of consumers. Either diesel or
hydraulic turbine work, or they work together.

The Metering class implements the definition of imbalance ranges according by current values of the different between
generation and consumption.

Pump operation.
The pumps only work in the first range. Moreover, their total power must be a multiple of 0.5 MW. In addition, it makes
no sense to turn on the pump for a short period. Here you need to look at the forecast two or three steps ahead. If
the capacity is increasing (1 forecast for 10 minutes), and the following forecasts (for 20 and 30 minutes) show a drop,
then there is no point in running the pump for a short time.

Hydraulic turbine.
This power plant has 4 turbines of 2.83 MW each. Therefore, their rated power is 2.83 MW. But these turbines retain
their high efficiency up to 40-50%. Therefore, the threshold for switching on a hydraulic turbine can be considered
as 1 MW. Further, the turbine can increase its power higher with practically no significant decrease in efficiency.

Diesel.
The power plant has different diesel engines with different power. But we will assume that all diesels have the same
power of 1.02 MW. I have all the data on these engines and they are more modern. Their minimum power is 30%, below they
do not work. But the optimal operating mode is 75-100%. Short-term work at 110% is allowed. Startup time 10 minutes.

Parallel operation of the hydraulic turbine and the pumps is not permitted. This will immediately lead to large losses.

Parallel operation of diesel engines and turbines is allowed. But a balance must be struck here. The turbine cannot
operate at a power less than 1 MW, and it is undesirable to operate a diesel engine at a power less than 0.75 MW. Thus,
the turbine can be started in parallel with diesel engines only if the total power shortage is more than 3 MW.

The Policer class implements the shape of packet traffic, it gives the color and prioritet for  each entered packet
(descriptor of the engine) according by above rules.

"""

import sys
import copy
import numpy as np

from simTM.cfg import D_LOGS,NO_COLOR,RED,ORANGE,GREEN, DIESEL, HYDR_TRB, HYDR_PUMP
from simTM.descriptor import Descriptor, SmartReq
from predictor.utility import msg2log

""" Class Metering """
# Range limit constants
RNG_PWR    = 0  # Power greather than consumer demands
RNG_PLUS   = 1  # Pwr >~0
RNG_MINUS  = 2  # Pwr<~ 0
RNG_CONSUM = 3  # Consumer demands greather than power

RNG_BALANCE =1
RNG_LMT_PUMP=0.5
RNG_LMT_BALANCE=0.0
RNG_LMT2=-0.5
class Metering:

    def __init__(self,y:list, f:object =None):
        self.f=f
        self.y=copy.copy(y)
        self.range = RNG_PLUS
        self.calcRange()
        self.pumpEnable = False
        self.isPumpEnabled() # set self.pumpEnable
        self.smartreq=None
        self.d_parsing={}

    def calcRange(self):
        if self.y[0]>=RNG_LMT_PUMP:
            self.range=RNG_PWR
        elif self.y[0]>=RNG_LMT_BALANCE and self.y[0]<RNG_LMT_PUMP:
            self.range=RNG_PLUS
        elif self.y[0]>RNG_LMT2 and self.y[0]<RNG_LMT_BALANCE:
            self.range=RNG_MINUS
        elif self.y[0]<=RNG_LMT2:
            self.range=RNG_CONSUM
        else:
            self.range=1
        return self.range

    def isPumpEnabled(self):
        if self.range>0:
            self.pumpEnable=False
        else:
            # if self.y[0]<=self.y[1] and self.y[1]<=self.y[2]:
            if self.y[0] >=RNG_LMT_PUMP and self.y[1] >RNG_LMT_BALANCE and  self.y[2]>RNG_LMT_BALANCE:
                self.pumpEnable=True
            else:
                self.pumpEnable=False
        msg="Current Imbalance {:<6.3f} Predict 1th period {:<6.3f} Predict 2th period {:<6.3f}".format(self.y[0],
                                                                                                self.y[1],self.y[2])
        msg2log("Metering",msg,self.f)
        return self.pumpEnable
    """ This method parses all smart requests and estimates amount of power per type of DER"""
    def requestsParse(self,smartreqs:dict):

        self.d_parsing[DIESEL]    = {"PowerSum":0.0,"token":0,"inc":0,"dec":0}
        self.d_parsing[HYDR_TRB]  = {"PowerSum":0.0,"token":0,"inc":0,"dec":0}
        self.d_parsing[HYDR_PUMP] = {"PowerSum":0.0,"token":0,"inc":0,"dec":0}
        msg=""

        try:

            for priority ,req_list in smartreqs.items():
                for req in req_list:
                    request:SmartReq = req
                    descr:Descriptor =request.descr
                    token=request.token
                    if request.typeDer==DIESEL:
                        self.d_parsing[DIESEL]["PowerSum"]   +=descr.desc["CurrentPower"]
                    elif request.typeDer==HYDR_TRB:
                        self.d_parsing[HYDR_TRB]["PowerSum"] +=descr.desc["CurrentPower"]
                    elif request.typeDer==HYDR_PUMP:
                        self.d_parsing[HYDR_PUMP]["PowerSum"]+=descr.desc["CurrentPower"]
                    else:
                        pass
        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
        finally:
            if len(msg) > 0:
                msg2log(type(self).__name__, msg, f=D_LOGS["except"])
        return

class Policer:

    def __init__(self,metering:Metering=None,f:object = None ):
        self.metering=metering
        self.f=f

    def setColor(self):
        msg=""
        try:
            if self.metering is None:
                return
            self.metering.smartreq.descr.setDesc(Color=GREEN)

            if self.metering.smartreq.typeDer==HYDR_PUMP and self.metering.pumpEnable==False:
                self.metering.smartreq.descr.setDesc(Color=RED)
        except KeyError as e:
            msg = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
        finally:
            if len(msg) > 0:
                msg2log("setColor", msg, f=D_LOGS["except"])
        return

    def isDiscard(self):
        ret=False
        msg1=""
        try:
            if self.metering.smartreq.descr.desc["Color"] == RED:
                msg= f"""Discarded packet: Device id {self.metering.smartreq.id} Color {self.metering.smartreq.descr.desc['Color']} Device type {self.metering.smartreq.typeDer} Model {self.metering.smartreq.model}"""
                msg2log("Policer",msg,self.f)
                ret=True
        except KeyError as e:
            msg1 = "O-o-ops! I got a KeyError - reason  {}".format(str(e))
        except:
            msg1 = "O-o-ops! Unexpected error: {}".format(sys.exc_info())
        finally:
            if len(msg1) > 0:
                msg2log("isDiscard", msg1, f=D_LOGS["except"])
        return ret

class PolicerDiesel(Policer):

    def __init__(self,f:object=None):
        super().__init__(f)

class PolicerHTurbine(Policer):
    def __init__(self, f: object = None):
        super().__init__(f)


if __name__ == "__main__":
    pass