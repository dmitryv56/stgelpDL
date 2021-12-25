#!/usr/bin/env python3

"""

"""

import logging
import pandas as pd

from predictor.demandwidget import DemandWidget
from predictor.control import ControlPlane
from predictor.api import createDeltaList
from msrvcpred.app.clients.client_Predictor import PredictorClient

logger = logging.getLogger(__name__)


class ClientDemandWidget(DemandWidget):
    """
    ClientDemandWidget
    """

    def __init__(self, scaled__data, start__date, end__date, time__trunc, geo__limit, geo__ids, f=None):
        super().__init__(scaled__data, start__date, end__date, time__trunc, geo__limit, geo__ids, f)
        self.client = PredictorClient()

    def getDemandRT(self, requested_widget=None):
        r"""
               This method sends parameterized GET-request to
               'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real ' site, parses received widget in
               json-format and creates DataFrame object for received time series (TS).
               The received request is an object contains two dictionaries 'data' and 'included' .
               The 'data' is a header while 'included' comprises the time series of the demand in MWatt and scale TS
               with values between 0 and 1.

               Scaled data mode:
               In order to get the time series scaled between 0 -1, self.scaled_data is True, this json requested
               widget was previously received and saved. Now it passed through parameter list.
               The requested widget parsed and created DataFrame object.

               Pure data mode:
               For pure TS, self.scaled_data is False, is need to send request to receive the json requested widget.
               The None passed through parameter list. This method sends GET-request to
               'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real ' site,
               parses received json-widget and  creates DataFrame object.
                  'GET' request like as
                  'start_date=2020-08-23T00:00&end_date=2020-08-23T02:00&time_trunc=hour&geo_trunc=electric_system
                  &geo_limit=canarias&geo_ids=8742'

               Examples of URL
                 url_demanda = 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?'
                 url = 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?start_date=2020-08-23T00:00
                        &end_date=2020-08-23T02:00&time_trunc=hour&geo_trunc=electric_system&geo_limit=canarias
                        &geo_ids=8742'

               :param requested_widget if  self.scaled_data else None
               :return: requested_widget
               """
        pass
        self.client.start_time = self.start_date
        self.client.end_time = self.end_date
        dt, real_demand, programmed_demand, forecast_demand = self.client.get_streaming_data("")

        # if not self.scaled_data:
        #     try:
        #         r = requests.get(self.url)
        #     except ConnectionError as e:
        #         print(e)
        #         r = "No response"
        #         self.logErrorResponse(r)
        #         return None
        #     if r.status_code != 200:
        #         self.logErrorResponse(r)
        #         return None
        #
        #     requested_widget = json.loads(r.content.decode())

        # self.logRequestedWidgetHeader(requested_widget)

        # flag -all time series sizes are equal
        # shortest time series if flag is False else size of time series
        # the list of time series sizes
        dt_size = len(dt)
        real_demand_size = len(real_demand)
        programmed_demand_size = len(programmed_demand)
        forecast_demand_size = len(forecast_demand)
        short_size = min(dt_size, real_demand_size, programmed_demand_size, forecast_demand_size)

        self.n_ts = 3
        if self.n_ts != 3 or short_size == 0 or short_size is None:
            msg = f"""
                           For this Data Adapter, we expect to get three time series (TS) of  non-zero length ( Demand, 
                           Programmed, Forecast) on every GET-request. The received widget has two TS< so the widget is 
                           discarded.
                           The GET-request will be repeated after short time-out.
                           Number of time series : {self.n_ts}
                           Time series length    : {short_size}
                   """
            self._log.info(msg)
            return None

        self.ts_size = short_size  # time series size
        self.names = ['Date Time', 'Real demand', 'Programmed demand', 'Forecast demand']
        self.names[1] = self.names[1].replace(' ', '_')
        self.names[2] = self.names[2].replace(' ', '_')
        self.names[3] = self.names[3].replace(' ', '_')

        (imbalance_dset, programmed_dset, demand_dset) = ControlPlane.get_modeImbalanceNames()
        mode_imbalance = ControlPlane.get_modeImbalance()
        if mode_imbalance:
            self.names.append(imbalance_dset)

        self.df = pd.DataFrame(columns=[self.names[0], self.names[1], self.names[2], self.names[3]])

        # if self.scaled_data:
        #     for i in range(self.ts_size):
        #         self.df = self.df.append(
        #             {self.names[0]: requested_widget['included'][0]['attributes']['values'][i]['datetime'],
        #              self.names[1]: requested_widget['included'][0]['attributes']['values'][i][
        #                  'percentage'],
        #              self.names[2]: requested_widget['included'][1]['attributes']['values'][i][
        #                  'percentage'],
        #              self.names[3]: requested_widget['included'][2]['attributes']['values'][i][
        #                  'percentage']},
        #             ignore_index=True)
        # else:
        #     for i in range(self.ts_size):
        #         self.df = self.df.append(
        #             {self.names[0]: requested_widget['included'][0]['attributes']['values'][i]['datetime'],
        #              self.names[1]: requested_widget['included'][0]['attributes']['values'][i]['value'],
        #              self.names[2]: requested_widget['included'][1]['attributes']['values'][i]['value'],
        #              self.names[3]: requested_widget['included'][2]['attributes']['values'][i]['value']
        #              },
        #             ignore_index=True)
        # Only not scaled data processed
        for i in range(self.ts_size):
            self.df = self.df.append(
                {self.names[0]: dt[i],
                 self.names[1]: real_demand[i],
                 self.names[2]: programmed_demand[i],
                 self.names[3]: forecast_demand[i]
                 },
                ignore_index=True)

        self.title = self.names[1]
        self.last_time = self.df[self.names[0]].max()
        print(self.df)

        if mode_imbalance:
            deltalist = createDeltaList(self.df[programmed_dset], self.df[demand_dset])
            self.df[self.names[4]] = deltalist
            self.title = self.names[4]

        self.one_word_title = self.title.replace(' ', '_')
        self.type_id = 7
        # if self.f is not None:
        #     self.logDF()   # it prints only int value, but our values are float  - TODO
        return None  # return for compatible with method in parent class