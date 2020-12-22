#!/usr/bin/python3

from math import floor,ceil
import numpy as np

from simTM.cfg import D_LOGS,DELTAPWR, PUMP_PERIOD

""" Imbalance predictor """


class ImbPred:

    def __init__(self, low, high, f: object = None):
        self.low = floor(low/DELTAPWR)
        self.high = ceil(high/DELTAPWR)
        self.f = f
        self.imbSeq = []
        self.imbLst = []

    def genImbSeq(self, low: int = None, high: int = None, nSize: int = PUMP_PERIOD):
        if low is None:
            low = self.low
        else:
            low = floor(low / DELTAPWR)
        if high is None:
            high = self.high
        else:
            high = ceil(high / DELTAPWR)

        imbSeq = np.random.randint(size=nSize, low=low, high=high)
        if len(self.imbSeq) == 0:
            self.imbSeq=(np.round(imbSeq*DELTAPWR,2)).tolist()
        else:
            self.imbSeq.pop(0)
            self.imbSeq.append(np.round(imbSeq[0]*DELTAPWR,2))
        self.imbLst.append(round(self.imbSeq[0],3))
        return

    # not used
    def genImb(self, low: int = None, high: int = None):
        if low is None:
            low = self.low
        else:
            low = floor(low / DELTAPWR)
        if high is None:
            high = self.high
        else:
            high = ceil(high / DELTAPWR)
        val = np.random.randint(size=1, low=low, high=high)
        val = val*DELTAPWR
        self.imbLst.append(val[0])
        return val[0]


if __name__ == "__main__":
    pass
