#!/usr/bin/env python3

""" read data """

import logging
from datetime import datetime, timedelta as td

from predictor.demandwidget import DemandWidget
from predictor.utility import cSFMT
from msrvcpred.cfg import MMV, MAGIC_TRAIN_CLIENT, MAGIC_PREDICT_CLIENT, MAGIC_CHART_CLIENT, MAGIC_WEB_CLIENT

logger = logging.getLogger(__name__)

DEFAUL_DAYS_SHIFT = 7

# MMV = 1e-15  # Magic missed value


class DataHelper(DemandWidget):
    """ class """

    def __init__(self, scaled__data: bool = False, start__date: str = None,
                 end__date: str = datetime.now().strftime(cSFMT), time__trunc: str = 'hour', geo__limit: str = None,
                 geo__ids: str = None, f = None):
        self._log = logger
        super().__init__(scaled__data, start__date, end__date, time__trunc, geo__limit, geo__ids, f)

    def get_data(self, days: int = 0, minutes: int = 0):
        scaled_data = False
        # start_time = "2020-08-30 00:00:00"
        # end_time_t = datetime.now()
        # self.end_date = end_time_t.strftime(cSFMT)
        # start_time_t = end_time_t - td(days=days,minutes=minutes)
        # self.start_date = start_time_t.strftime(cSFMT)

        if self.start_date is None:
            if days == 0 and minutes == 0:  # no data
                days = DEFAUL_DAYS_SHIFT
                self._log.info(
                    "No start date, no time delta passed. Default time delta is used: {} days".format(days))

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

        results = []

        for i in range(len(self.df)):
            results.append({'timestamp': self.df['Date Time'].values[i],
                            'real_demand': self.df['Real_demand'].values[i],
                             'programmed_demand': self.df['Programmed_demand'].values[i],
                             'forecast_demand': self.df['Forecast_demand'].values[i],
                             'status': i,
                             'statusmsg': "success"})

        return results


def get_data_for_train(start_time: str = None, end_time: str = None) -> list:
    dh = DataHelper(start__date=start_time, end__date=end_time)
    results = None
    if start_time is None:
        logger.info("Start_date: {} End_date: {}".format(dh.start_date, dh.end_date))
        results = dh.get_data(days=7)
    else:
        results = dh.get_data()
    logger.info("Start_date: {} End_date: {}".format(dh.start_date, dh.end_date))
    return results


def get_data_for_predict(start_time: str = None, end_time: str = None)->list:
    dh = DataHelper(start__date=start_time, end__date=end_time)
    results = None
    if start_time is None:
        results = dh.get_data(minutes=60)
    else:
        results = dh.get_data()
    logger.info("Start_date: {} End_date: {}".format(dh.start_date, dh.end_date))
    return results

class PredictReport():
    """ Predict report """

    def __init__(self,timestamp: str = "", timeseries: str ="", model0: str = "", model1: str = "", model2: str ="",
                 model3: str = "", model4: str = "", model5: str = "", model6:str = "" ,model7: str = "",
                 model8: str = "", model9: str = ""):
        self._log = logging.getLogger(self.__class__.__name__)
        self.l_data = []

        self.timestamp = timestamp
        self.timeseries = timeseries
        self.model0 = model0
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        self.model8 = model8
        self.model9 = model9

    @staticmethod
    def error_predict_result():
        return {'status': -1, 'timestamp': "", 'timeseries': MMV, 'model0': MMV, 'model1': MMV,
                        'model2': MMV, 'model3': MMV, 'model4': MMV, 'model5': MMV, 'model6': MMV,
                        'model7': MMV, 'model8': MMV, 'model9': MMV}


    def save_predict(self, timestamp: str, timeseries: float, model0:float, model1: float, model2: float, model3: float,
                     model4: float, model5: float, model6: float, model7: float, model8: float, model9: float):
        """ Received predict data contains the timeseries value for previous predict.
        The timeseries value for previous predict is updated and last predict timeserires value is set to MMV """
        if self.l_data:
            last_predicted_period = self.l_data[-1][self.timestamp]
            if timestamp == last_predicted_period:
                self._log.info("The predict on {} was already saved".format(timestamp))
                return

        if timeseries>MMV and len(self.l_data)>1:
            self.l_data[-1]['timeseries'] = timeseries
            timeseries = MMV

        single_predict = {'readByClient': [], self.timestamp: timestamp, self.timeseries: timeseries,
                          self.model0: model0, self.model1: model1, self.model2: model2, self.model3: model3,
                          self.model4: model4, self.model5: model5, self.model6: model6, self.model7: model7,
                          self.model8: model8, self.model9: model9}
        self._log.info("Saved: {}".format(single_predict))
        self.l_data.append(single_predict)
        self._log.info(" Saved predicts")
        self.saved_predicts_log()
        self._log.info("")

    def get_predict(self, clientid):
        results = []
        for item in self.l_data:
            if clientid in item['readByClient']:
                continue

            item['readByClient'].append(clientid)
            results.append({'status': 0, 'timestamp': item[self.timestamp], 'timeseries': item[self.timeseries],
                            'model0': item[self.model0], 'model1': item[self.model1], 'model2': item[self.model2],
                            'model3': item[self.model3], 'model4': item[self.model4], 'model5': item[self.model5],
                            'model6': item[self.model6], 'model7': item[self.model7], 'model8': item[self.model8],
                            'model9': item[self.model9]})

            self._log.info("Read by {} client: {}".format(clientid, results[-1]))

        if len(results) == 0:  # error
            results.append(type(self).error_predict_result())

            self._log.error("No predict retrived for {} client".format(clientid))
        return results

    def get_titles(self) -> dict:
        result = {'status': 0, 'timestamp': self.timestamp, 'timeseries': self.timeseries,
                            'model0': self.model0, 'model1': self.model1, 'model2': self.model2,
                            'model3': self.model3, 'model4': self.model4, 'model5': self.model5,
                            'model6': self.model6, 'model7': self.model7, 'model8': self.model8,
                            'model9': self.model9}

        self._log.info("get_titles(): {}".format(result))
        return result

    def saved_predicts_log(self):
        for item in self.l_data:
            self._log.info(item)





# def save_title(start_time: str = None, end_time: str = None) -> list:
#     pr = PredictReport(timestamp, timeseries, model0, model1, model2, model3, model4, model5, model6, model7, model8,
#                        model9)
#     results = None
#     if start_time is None:
#         logger.info("Start_date: {} End_date: {}".format(dh.start_date, dh.end_date))
#         results = dh.get_data(days=7)
#     else:
#         results = dh.get_data()
#     logger.info("Start_date: {} End_date: {}".format(dh.start_date, dh.end_date))
#     return results






