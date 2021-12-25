#!/usr/bin/env python3

""" read data """
import os
import logging
from datetime import datetime,timedelta as td

from predictor.demandwidget import DemandWidget
from predictor.utility import cSFMT



logger=logging.getLogger(__name__)

DEFAUL_DAYS_SHIFT =7

class DataHelper(DemandWidget):
    """ class """

    def __init__(self, scaled__data:bool=False, start__date:str=None, end__date:str=datetime.now().strftime(cSFMT),
                 time__trunc:str='hour', geo__limit:str=None, geo__ids:str=None, f=None):
        self._log = logger
        super().__init__(scaled__data, start__date, end__date, time__trunc, geo__limit, geo__ids, f)

    def get_data(self,days:int=0, minutes:int=0):
        scaled_data = False
        # start_time = "2020-08-30 00:00:00"
        # end_time_t = datetime.now()
        # self.end_date = end_time_t.strftime(cSFMT)
        # start_time_t = end_time_t - td(days=days,minutes=minutes)
        # self.start_date = start_time_t.strftime(cSFMT)

        if self.start_date is None:
            if days==0 and minutes == 0: # no data
                self._log.info(
                    "No start date, no time delta passed. Default time delta is used: {} days".format(DEFAUL_DAYS_SHIFT))
                days = DEFAUL_DAYS_SHIFT
            start_time_t = datetime.now() - td(days=days, minutes=minutes)
            self.start_date = start_time_t.strftime(cSFMT)
        msg = "The request to retrieve data for a period fron {} till {}".format(self.start_date, self.end_date)
        self._log.info(msg)
        self.set_url()
        self._log.info(self.url)

        requested_widget = self.getDemandRT(None)
        if requested_widget is None:
            self._log.error("Can not get requested data")
            return None

        print("Requested widget has type {}".format(type(requested_widget)))

        results =[]

        for i in range(len(self.df)):
            results.append({'timestamp':self.df['Date Time'].values[i],
                            'real_demand':self.df['Real_demand'].values[i],
                             'programmed_demand':self.df['Programmed_demand'].values[i],
                             'forecast_demand':self.df['Forecast_demand'].values[i],
                             'status':i,
                             'statusmsg':"success"})

        return results

def get_data_for_train(start_time:str=None,end_time:str=None)->list:
    dh = DataHelper(start__date=start_time, end__date=end_time)
    results = None
    if start_time is None:
        logger.info("Start_date: {} End_date: {}".format(dh.start_date, dh.end_date))
        results = dh.get_data(days=7)
    else:
        results = dh.get_data()
    logger.info("Start_date: {} End_date: {}".format(dh.start_date, dh.end_date))
    return results

def get_data_for_predict(start_time:str=None,end_time:str=None)->list:
    dh = DataHelper(start__date=start_time, end__date=end_time)
    results = None
    if start_time is None:

        results = dh.get_data(minutes=60)
    else:
        results = dh.get_data()
    logger.info("Start_date: {} End_date: {}".format(dh.start_date, dh.end_date))
    return results

