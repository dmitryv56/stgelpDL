#! /usr/bin/python3

class GlobalConst():
    _verbose = 0

    def __init__(self):
        pass

    @staticmethod
    def setVerbose(verbose):
        GlobalConst._verbose =verbose

    @staticmethod
    def getVerbose():
        return GlobalConst._verbose


