#!/usr/bin/python3

""" Configuration
Diesel Genset
"""

# hyperpearameters for simulator session
DIESEL_NUMBER = 4 # amount of the diesels
HYDR_TRB_NUMBER=2
HYDR_PUMP_NUMBER=4

AGE = 512   # size of saved siquences in descriptors
SIM_PERIOD = 128
PUMP_PERIOD = 3 # The pump changes state according by PUMP_PERIOD forecasts

DELTAPWR = 0.1 #100 Кwt
#Imbalance generation
MAX_CUSTOM=-10
MAX_GENER = 10

# log
D_LOGS={"main":None, "init":None, "current":None,"steps":None, "except":None, "desc":None, "sent":None, "tm":None,
        "metering":None, "policer":None, "plot":None}
# Global variables and structure constants
UNDF = -1

# Types D(istibuted) E(nergy) R(esources)
DIESEL    = 0
PV        = 1
CHP       = 2
WIND_TRB  = 3
HYDR_TRB  = 4
HYDR_PUMP = 5

DER_NAMES={DIESEL:"Diesel", PV:"PV", CHP:"CHP", WIND_TRB:"Wind Turbine", HYDR_TRB:"Hydrawlic Turbune",
           HYDR_PUMP:"Hydrawlic Pump"}

# Request type
DEC_PWR = 0
INC_PWR = 1

# Colors
NO_COLOR=0
RED = 1
ORANGE = 2
GREEN = 3
D_COLOR={NO_COLOR:'no color',RED:'red',ORANGE:'orange',GREEN:'green'}

# A possible state of the units of DER
S_OFF = 0
S_LPWR = 1
S_MPWR = 2
S_HPWR = 3
S_ON   = 1  # HYDR_TRB

DIESEL_MODEL  =['VPP1250','VPP590']
DIESEL_STATES=[S_OFF,S_LPWR,S_MPWR,S_HPWR]
DIESEL_STATES_NAMES={S_OFF:"Off",S_LPWR:"Low Pwr",S_MPWR:"Mid.Pwr",S_HPWR:"High Pwr"}
# Types DER
DIESEL    = 0
PV        = 1
CHP       = 2
WIND_TRB  = 3
HYDR_TRB  = 4
HYDR_PUMP = 5

DER_NAMES={DIESEL:"Diesel",PV:"PV",CHP:"CHP",WIND_TRB:"Wind Turbine",HYDR_TRB:"Hydrawlic Turbune",HYDR_PUMP:"Hydrawlic Pump"}

# Priorites
PR_L = 0
PR_M = 1
PR_H = 2


HYDR_TRB_MODEL=['HydrTrb']
HYDR_TRB_STATES=[S_OFF,S_ON]
HYDR_TRB_STATES_NAMES={S_OFF:"Off",S_ON:"On"}


HYDR_PUMP_MODEL=['HydrPump']
HYDR_PUMP_STATES=[S_OFF,S_ON]
HYDR_PUMP_STATES_NAMES={S_OFF:"Off",S_ON:"On"}


