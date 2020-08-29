#!/usr/bin/python3
import os
import requests
import json
import pandas as pd
import dateutil.parser
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
from collections import OrderedDict
from predictor.api import show_autocorr



"""
 This class reads in real time the data about electrical load from  https://www.ree.es/en/apidatos  site (RED Electrics 
 de Espana).
 The paramereized GET-request is sent ,the requested widget in json-format is  received.
 This widget is parsed and DataFrame object is created.
 The example of full GET request is below

 GET https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?start_date=2020-08-28T00:00&end_date=2020-08-28T18:00
     &time_trunc=hour

 So the class name is DemandWidget.

"""


class DemandWidget():
    _start_date = None
    _end_date = None
    _time_trunc = None
    _geo_limit = None
    _geo_ids = None
    _url_apidatos = 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?'
    _title = None
    _scaled_data =False

    def __init__(self, scaled__data,start__date, end__date, time__trunc, geo__limit, geo__ids, f=None):
        self.scaled_data = scaled__data
        self.f = f
        self.start_date = start__date
        self.end_date = end__date
        self.time_trunc = time__trunc
        self.geo_limit = geo__limit
        self.geo_ids = geo__ids
        self.df = None
        self.n_ts = 0
        self.ts_size = 0
        self.names = []
        self.last_time = None
        self.url = None

        pass

    @staticmethod
    def ISO8601toPyStr(ISO8601str):
        """
        The python date/datetime string likes as '2020-08-28 16:45:01' while ISO8601 string likes as
        '2020-08-28T16:45:01.000+02:00'

        :param ISO8601str:string in ISO8601 format , i.e. '2020-08-28T16:45:01.000+02:00'
        :return: should be '2020-08-28 16:45:01'
        """
        ourdatetime = dateutil.parser.parse(ISO8601str)
        return ourdatetime.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def ISO8601toDateTime(ISO8601str):
        """
        The python date/datetime is object while ISO8601 string likes as
        '2020-08-28T16:45:01.000+02:00'

        :param ISO8601str:string in ISO8601 format , i.e. '2020-08-28T16:45:01.000+02:00'
        :return: should be DateTime object
        """
        return dateutil.parser.parse(ISO8601str)

    @staticmethod
    def PyStrtoISO8601(pystr):
        """
        The python date/time string likes as '2020-08-28 16:45:01' while ISO8601 string likes as
        '2020-08-28T16:45:01.000+02:00'

        :param pystr: string in pytho fomat, i.e. '2020-08-28 16:45:01'
        :return: should be '2020-08-28T16:45:01.000+02:00'
        """
        ourdatetime = pystr.strftime('%Y-%m-%d %H:%M:%S')
        return ourdatetime.isoformat()

    # getter/setter
    def set_start_date(self, val):
        type(self)._start_date = val

    def get_start_date(self):
        return type(self)._start_date

    start_date = property(get_start_date, set_start_date)

    def set_end_date(self, val):
        type(self)._end_date = val

    def get_end_date(self):
        return type(self)._end_date

    end_date = property(get_end_date, set_end_date)

    def set_time_trunc(self, val):
        type(self)._time_trunc = val

    def get_time_trunc(self):
        return type(self)._time_trunc

    time_trunc = property(get_time_trunc, set_time_trunc)

    def set_geo_limit(self, val):
        type(self)._geo_limit = val

    def get_geo_limit(self):
        return type(self)._geo_limit

    geo_limit = property(get_geo_limit, set_geo_limit)

    def set_geo_ids(self, val):
        type(self)._geo_ids = val

    def get_geo_ids(self):
        return type(self)._geo_ids

    geo_ids = property(get_geo_ids, set_geo_ids)

    def set_url_apidatos(self, val):
        type(self)._url_apidatos = val

    def get_url_apidatos(self):
        return type(self)._url_apidatos

    url_apidatos = property(get_url_apidatos, set_url_apidatos)

    def set_title(self, val):
        type(self)._title = val

    def get_title(self):
        return type(self)._title

    title = property(get_title, set_title)

    def set_scaled_data(self, val):
        type(self)._scaled_data = val

    def get_scaled_data(self):
        return type(self)._scaled_data

    scaled_data = property(get_scaled_data, set_scaled_data)

    # methods
    def set_url(self):
        """
        This method forms the url for GET-request. the site is
        https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?

        The parameters are following
        start_date=<val>>
        end_date=<val>
        time_trunc=<val>
        sgeo_limit=<val> or empty string
        sgeo_ids=<val> or empty string

        :return:
        """
        sgeo_limit = ""
        if self.geo_limit is not None:
            sgeo_limit = "&geo_limit={}".format(self.geo_limit)

        sgeo_ids = ""
        if self.geo_ids is not None:
            sgeo_ids = "&geo_ids={}".format(str(self.geo_ids))

        self.url = "{}start_date={}&end_date={}&time_trunc={}{}{}".format(self.url_apidatos, self.start_date,
                                                                          self.end_date,
                                                                          self.time_trunc,
                                                                          sgeo_limit,
                                                                          sgeo_ids)

        print("GET {}".format(self.url))
        if self.f is not None:
            self.f.write("\nGET {}\n".format(self.url))

        return

    """
           This method sends GET-request to 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real ' site,
           parses received json-widget and  creates DataFrame object.
           'GET' request like as
           'start_date=2020-08-23T00:00&end_date=2020-08-23T02:00&time_trunc=hour&geo_trunc=electric_system&geo_limit=canarias&geo_ids=8742'


           :return: requested widget for using in future. There is a list that comprises dictionaries and lists/
           """
    def getDemandRT(self, requested_widget = None ):
        """
        Scaled data mode:
        In order to get the time series scaled between 0 -1, self.scaled_data is True, this json requested widget was
        previously received and saved. Now it passed through parameter list.
        The requested widget parsed and created DataFrame object.

        Pure data mode:
        For pure time series, self.scaled_data is False, is need to send request to receive the json requested widget.
        The None passed through parameter list. This method sends GET-request to
        'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real ' site,
        parses received json-widget and  creates DataFrame object.
           'GET' request like as
           'start_date=2020-08-23T00:00&end_date=2020-08-23T02:00&time_trunc=hour&geo_trunc=electric_system&geo_limit=canarias&geo_ids=8742'

        :param requested_widget if  self.scaled_data else None
        :return: requesyed_widget
        """
        pass

        if not self.scaled_data:
            r = requests.get(self.url)

            if r.status_code != 200:
                self.logErrorResponse(r)
                return None

            requested_widget = json.loads(r.content.decode())

        self.logRequestedWidgetHeader(requested_widget)

        # flag -all time series sizes are equal
        # shortest time series if flag is False else size of time series
        # the list of time series sizes
        sizes_equal, short_size, ts_sizes = self.logRequestedWidgetTimeSeries(requested_widget)

        self.n_ts = len(requested_widget['included'])  # number of time series

        self.ts_size = short_size  # time series size

        self.df = pd.DataFrame(columns=[self.names[0], self.names[1], self.names[2], self.names[3]])

        # for i in range(self.ts_size):
        #     self.df = self.df.append({names[0]: requested_widget['included'][0]['attributes']['values'][i]['datetime'],
        #                               names[1]: requested_widget['included'][0]['attributes']['values'][i]['value'],
        #                               names[2]: requested_widget['included'][0]['attributes']['values'][i][
        #                                   'percentage'],
        #                               names[3]: requested_widget['included'][1]['attributes']['values'][i]['value'],
        #                               names[4]: requested_widget['included'][1]['attributes']['values'][i][
        #                                   'percentage'],
        #                               names[5]: requested_widget['included'][2]['attributes']['values'][i]['value'],
        #                               names[6]: requested_widget['included'][2]['attributes']['values'][i][
        #                                   'percentage']},
        #                              ignore_index=True)
        if self.scaled_data:
            for i in range(self.ts_size):
                self.df = self.df.append(
                    {self.names[0]: requested_widget['included'][0]['attributes']['values'][i]['datetime'],
                     self.names[1]: requested_widget['included'][0]['attributes']['values'][i][
                         'percentage'],
                     self.names[2]: requested_widget['included'][1]['attributes']['values'][i][
                         'percentage'],
                     self.names[3]: requested_widget['included'][2]['attributes']['values'][i][
                         'percentage']},
                    ignore_index=True)
        else:
            for i in range(self.ts_size):
                self.df = self.df.append(
                    {self.names[0]: requested_widget['included'][0]['attributes']['values'][i]['datetime'],
                     self.names[1]: requested_widget['included'][0]['attributes']['values'][i]['value'],
                     self.names[2]: requested_widget['included'][1]['attributes']['values'][i]['value'],
                     self.names[3]: requested_widget['included'][2]['attributes']['values'][i]['value']
                     },
                    ignore_index=True)



        for i in range(self.ts_size):
            dt_obj = DemandWidget.ISO8601toDateTime(self.df[self.names[0]][i])
            self.df[self.names[0]][i] = dt_obj

        self.last_time = self.df[self.names[0]].max()
        # pd.set_option('display.max_rows', None)
        print(self.df)

        if self.f is not None:
            self.logDF()

        return requested_widget

    def logErrorResponse(self, r):
        """
        This method writes to log the response error info( code !=200)
        :param r:
        :return:
        """
        message = f"""
                        Error  : {r.status_code}.
                        Reason : {r.reason}.
                        URL    : {r.url}.
                        Detail : {r.text}.

                    """
        print(message)
        if self.f is not None:
            self.f.write(message)

        return

    def logRequestedWidgetHeader(self, requested_widget):
        """
        This method selects and writes to log the info from the header of received widgt.
        Sets the class variable 'title' (string)
        :param requested_widget:
        :return:
        """
        self.title = requested_widget['data']['attributes']['title']
        if self.scaled_data:
            self.title = "{} scaled between 0-1".format(self.title)

        message = f"""
                   Type        : {requested_widget['data']['type']}
                   ID          : {requested_widget['data']['id']}
                   Title       : {requested_widget['data']['attributes']['title']}
                   Last Update : {requested_widget['data']['attributes']['last-update']}
                   Description : {requested_widget['data']['attributes']['description']}
                   Title       : {self.title}
               """
        print(message)
        if self.f is not None:
            self.f.write("\n{}\n".format(message))
        return

    def logRequestedWidgetTimeSeries(self, requested_widget):
        """
        This method selects a time series into received widget and logs them.
        Selects the time series names for requsted widget and writes them in class variable 'names'  of list type.
        :param requested_widget:
        :return: sizes_equal, short_size, ts_sizes,
                where size_equal is a boolean flag: True - all time series have an equal size; False - different sizes
                short_size - the size of time series or shortest time series size
                ts_sizes - list of time series sizes.

        """

        n_ts = len(requested_widget['included'])
        ts_sizes = []
        self.names.append("Date Time")
        for i in range(n_ts):
            ts_sizes.append(len(requested_widget['included'][i]['attributes']['values']))
            message = f"""
                        Type : {requested_widget['included'][i]['type']}
                        Id   : {requested_widget['included'][i]['id']}
                        Title : {requested_widget['included'][i]['attributes']['title']}
                        Description : {requested_widget['included'][i]['attributes']['description']}
                        Type        : {requested_widget['included'][i]['attributes']['type']}
                        Color       : {requested_widget['included'][i]['attributes']['type']}
                        Magnitude   : {requested_widget['included'][i]['attributes']['magnitude']}
                        Last Update :  : {requested_widget['included'][i]['attributes']['last-update']}
                        Time Series size : {ts_sizes[i]}
                    """
            print(message)
            if self.f is not None:
                self.f.write("\n{}\n".format(message))

            self.names.append(requested_widget['included'][i]['attributes']['title'])

        # compare time series sizes
        sizes_equal = True
        short_size = ts_sizes[0]
        for i in range(n_ts - 1):
            if short_size != ts_sizes[i + 1]:
                sizes_equal = False
                if ts_sizes[i + 1] < short_size:
                    short_size = ts_sizes[i + 1]

        if not sizes_equal:
            message = f"""
                The time series sizes are nor equal
                The shortest time series has {short_size} size.
            """

            print(message)

            if self.f is not None:
                self.f.write("\n{}\n".format(message))

        return sizes_equal, short_size, ts_sizes

    def logDF(self):

        pass
        if self.f is None:
            return
        if self.scaled_data:
            self.f.write("{0:^60s}\n".format(self.title))
            stemplate = "{:<5s} {:<20s} {:>15.5f} {:>15.5f} {:>15.5f}\n"
        else:
            self.f.write("{0:^60s} (MW)\n".format(self.title))
            stemplate = "{:<5s} {:<20s} {:>15d} {:>15d} {:>15d}\n"

        print_dict = OrderedDict()
        print_dict["####"] = '{0:<5s}'

        for i in range(len(self.names)):
            if i == 0:
                print_dict[self.names[i]] = '{0:<20s}'
            else:
                print_dict[self.names[i]] = '{0:<15s}'

        for k, v in print_dict.items():
            self.f.write(v.format(k))
        self.f.write('\n')

        for i in range(self.ts_size):
            self.f.write(
                stemplate.format(str(i), self.df.values[i][0].strftime('%Y-%m-%d %H:%M:%S'), self.df.values[i][1],
                                 self.df.values[i][2], self.df.values[i][3] ))

        return

    # url_demanda = 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?'
    # url = 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?start_date=2020-08-23T00:00
    # &end_date=2020-08-23T02:00&time_trunc=hour&geo_trunc=electric_system&geo_limit=canarias&geo_ids=8742'

    def plot_ts(self, logfolder, stop_on_chart_show=True):
        plt.close("all")
        plt.style.use('seaborn-darkgrid')

        # create a color palette
        palette = plt.get_cmap('Set1')

        # times = mdates.drange(self.df['Date Time'].min(), self.df['Date Time'].max(), timedelta(minutes=10))
        df1=self.df.copy(deep=True)
        num = 0
        for column in df1.drop(['Date Time'], axis=1):
            num += 1
            plt.plot(df1['Date Time'], df1[column], marker='', color=palette(num), linewidth=1, alpha=0.9,
                     label=column)
        plt.legend(loc=2, ncol=2)

        plt.title('{}'.format(self.title))
        plt.xlabel("Time")
        plt.ylabel("Power (MW)")
        if self.scaled_data:
            plt.ylabel("Power (0 - 1)")


        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        if self.ts_size <= 500:
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        elif self.ts_size <= 1000:
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
        else:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

        plt.gcf().autofmt_xdate()
        plt.show(block=stop_on_chart_show)
        if logfolder is not None:
            plt.savefig("{}/{}.png".format(logfolder, self.title.replace(' ', '_')))
        del df1
        return

    def autocorr_show(self, logfolder, stop_on_chart_show=False):
        pass

        x = self.df[self.names[1]].values
        show_autocorr(x, (int)(len(x) / 4), self.title, logfolder, stop_on_chart_show, self.f)

        return


if __name__ == "__main__":
    with open("abc.log", 'w') as flog:
        dwdg = DemandWidget(False, "2020-08-26T00:00", "2020-08-29T18:00", "hour", None, None, flog)
        dwdg.set_url()

        print(dwdg.url)

        dwdg.getDemandRT()
        dwdg.plot_ts(os.getcwd(), False)
        dwdg.autocorr_show(os.getcwd(), False)

        pass
