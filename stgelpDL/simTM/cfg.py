#!/usr/bin/python3

""" Configuration
Diesel Genset
"""

DIESEL_NUMBER = 6 # amount of the diesels
DIESEL_MODEL  =['VPP1250','VPP590']

UNDF = -1
AGE = 512   # size of saved siquences in descriptors
SIM_PERIOD = 512
#Imbalance generation
MAX_CUSTOM=-5
MAX_GENER = 5

# Request type
DEC_PWR = 0
INC_PWR = 1
# A possible state of the units of DER
S_OFF = 0
S_LPWR = 1
S_MPWR = 2
S_HPWR = 3
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

DELTAPWR = 0.1 #100 Кwt
