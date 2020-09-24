#!/usr/bin/python3
r""" This module 'demandwidget' allows the Control Plane functionality.
The module contains a base class DataAdapter  and derived class DemandWidget are used for real time a dataset creating.
Now we have limited access to  RED Electrica De Espana
            https://www.https://www.ree.es/en/datos/todate
through REData Information access API that provides a simple REST service to allow third parties to access the baсkend
data used in REData application. By using this API , we can be able to retrieve data fom the REData widgets and use tn
for the short term prediction by using Deep Learning and Statistcal models.
The use of this service is simply. Onle GET requests are allowed since he purpose of this API is provide data related
to REData app. Each widget is set up bye series indicators time series) which provide data related to particular
category. The detailed form of the URI can be found on the site
            https://www.ree.es/en/apidatos

The  DemandWidget class is used to read data in real time, create a dataset in the format pandas' DataFrame and save
it as csv-file.

We plan , if possible,if exists information access to backend data another electrical grids, to add other classes  in
this module which will be retrieve the time series in the real time.

"""
import os
import time
import requests
import json
import pandas as pd
import dateutil.parser
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
from collections import OrderedDict
from api import show_autocorr,createDeltaList
from utility import cSFMT,incDateStr,decDateStr,msg2log, PlotPrintManager
from pathlib import Path
from control import ControlPlane

class DataAdapter():
    r""" The base class DataAdapter.
    Now we work only with one DemandWidget data adapter.

    """

    def __init__(self, f = None):
        self.f = f
        pass


class DemandWidget(DataAdapter):
    r"""
    This class reads in real time the electrical load backend data from
         https://www.ree.es/en/apidatos  site (RED Electrics  de Espana).
    It sends the paramereized GET-request to REData API interface ,receives the requested widget in json-format .
    This widget is parsed and DataFrame object is created.
    The example of full GET request is below.

        GET https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?start_date=2020-08-28T00:00
        &end_date=2020-08-28T18:00&time_trunc=hour
    """

    _start_date = None
    _end_date = None
    _time_trunc = None
    _geo_limit = None
    _geo_ids = None
    _url_apidatos = 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?'
    _title = None
    _type_id = None
    _scaled_data =False

    def __init__(self, scaled__data,start__date, end__date, time__trunc, geo__limit, geo__ids, f=None):
        r"""


        :param scaled__data:
        :param start__date:
        :param end__date:
        :param time__trunc:
        :param geo__limit:
        :param geo__ids:
        :param f:
        """

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
        self.one_word_title=""
        super().__init__(f)

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
        r"""
        The python date/datetime is object while ISO8601 string likes as
        '2020-08-28T16:45:01.000+02:00'. This static method converts the datetime string to datatime object.


        :param ISO8601str:
        :return: datetime object
        """

        return dateutil.parser.parse(ISO8601str)

    @staticmethod
    def PyStrtoISO8601(pystr):
        r"""
        The python date/time string likes as '2020-08-28 16:45:01' while ISO8601 string likes as
        '2020-08-28T16:45:01.000+02:00'. This static method converts the PyString datetime to ISO-8061 string.

        :param pystr: string in pytho fomat, i.e. '2020-08-28 16:45:01'
        :return: should be '2020-08-28T16:45:01.000+02:00'
        """

        ourdatetime = pystr.strftime('%Y-%m-%d %H:%M:%S')
        return ourdatetime.isoformat()

    """ getter/setter """
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

    def set_type_id(self, val):
        type(self)._type_id= val

    def get_type_id(self):
        return type(self)._type_id

    type_id = property(get_type_id, set_type_id)

    def set_scaled_data(self, val):
        type(self)._scaled_data = val

    def get_scaled_data(self):
        return type(self)._scaled_data

    scaled_data = property(get_scaled_data, set_scaled_data)

    """ methods """
    def set_url(self):
        r"""
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

        msg ="GET {}".format(self.url)
        msg2log(self.set_url.__name__, msg, self.f)

        return


    def getDemandRT(self, requested_widget = None ):
        r"""
        This method sends parameterized GET-request to 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real '
        site, parses received widget in json-format and creates DataFrame object for received time series.
        The recived request is an object comprises two dictionares 'data' and 'included' .
        The 'data' is a header while 'included' comprises the time series of the demand in MWatt and scale time series
        with values between 0 and 1.

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

        Examples of URL
          url_demanda = 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?'
          url = 'https://apidatos.ree.es/en/datos/demanda/demanda-tiempo-real?start_date=2020-08-23T00:00
                 &end_date=2020-08-23T02:00&time_trunc=hour&geo_trunc=electric_system&geo_limit=canarias&geo_ids=8742'

        :param requested_widget if  self.scaled_data else None
        :return: requesyed_widget
        """
        pass

        if not self.scaled_data:
            try:
                r = requests.get(self.url)
            except ConnectionError as e:
                print(e)
                r = "No response"
                self.logErrorResponse(r)
                return None
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
        if self.n_ts!=3 or short_size == 0 or short_size is None:
            msg=f"""
                    For this Data Adapter, we expect to get three time series of  non-zero length ( Demand,Programmed, 
                    Forecast) on every GET-request. The received widget has two time series< so the widget is discarded.
                    The GET-reuest will be repeated after short time-out.
                    Number of time series : {self.n_ts}
                    Time series length    : {short_size}
            """
            msg2log(self.getDemandRT.__name__, msg, self.f)
            return None

        self.ts_size = short_size  # time series size


        self.names[1] = self.names[1].replace(' ','_')
        self.names[2] = self.names[2].replace(' ', '_')
        self.names[3] = self.names[3].replace(' ', '_')

        (imbalance_dset,programmed_dset, demand_dset) = ControlPlane.get_modeImbalanceNames()
        mode_imbalance =ControlPlane.get_modeImbalance()
        if mode_imbalance:
            self.names.append(imbalance_dset)

        self.df = pd.DataFrame(columns=[self.names[0], self.names[1], self.names[2], self.names[3]])

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

        self.last_time = self.df[self.names[0]].max()
        print(self.df)

        if mode_imbalance:

            deltaList=createDeltaList(self.df[programmed_dset], self.df[demand_dset])
            self.df[self.names[4]]=deltaList


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
        msg2log(self.logErrorResponse.__name__, message, self.f)
        return

    def logRequestedWidgetHeader(self, requested_widget):
        """
        This method selects and writes to log the info from the header of received widgt.
        Sets the class variable 'title' (string)
        :param requested_widget:
        :return:
        """

        self.title   = requested_widget['data']['attributes']['title']
        self.one_word_title=self.title.replace(' ','_')
        self.type_id = {requested_widget['data']['type']}
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
        msg2log(self.logRequestedWidgetHeader.__name__, message, self.f)
        return

    def logRequestedWidgetTimeSeries(self, requested_widget):
        """
        This method selects a time series into received widget and logs them.
        Selects the time series names for requested widget and writes them in class variable 'names'  of list type.
        The first is "main" time series, for example 'Real demand', followed by 'Programmed demand' and 'Forecast demand'
        For the current times, the 'Real Demand' must be missed and so we selects  the size of' Real demand' time series
         as ts_size
        :param requested_widget:
        :return: sizes_equal, ts_size, ts_sizes,
                where size_equal is a boolean flag: True - all time series have an equal size; False - different sizes
                ts_size - the size of main time series
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
            msg2log(self.logRequestedWidgetTimeSeries.__name__, message, self.f)

            self.names.append(requested_widget['included'][i]['attributes']['title'])

        """  compare time series sizes """
        sizes_equal = True
        ts_size = ts_sizes[0]
        for i in range(n_ts - 1):
            if ts_size != ts_sizes[i + 1]:
                sizes_equal = False
                break

        if not sizes_equal:
            message = f"""
                The time series sizes are nor equal
                The shortest time series has {ts_size} size.
            """
            msg2log(self.logRequestedWidgetTimeSeries.__name__, message, self.f)

        return sizes_equal, ts_size, ts_sizes

    def logDF(self):
        """

        :return:
        """

        if self.f is None:
            return
        stemplate = "{:<5s} {:<20s} "
        if self.scaled_data:
            self.f.write("{0:^60s}\n".format(self.title))
            for i in range(1,len(self.names)):
                stemplate = stemplate + "{:>15.5f} "

            # stemplate = "{:<5s} {:<20s} {:>15.5f} {:>15.5f} {:>15.5f}\n"
        else:
            self.f.write("{0:^60s} (MW)\n".format(self.title))
            for i in range(1,len(self.names)):
                stemplate = stemplate + "{:>15d} "
            # stemplate = "{:<5s} {:<20s} {:>15d} {:>15d} {:>15d}\n"
        stemplate = stemplate + "\n"
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
            if ControlPlane.get_modeImbalance():
                self.f.write(
                    stemplate.format(str(i), self.df.values[i][0], self.df.values[i][1], self.df.values[i][2],
                                     self.df.values[i][3], self.df.values[i][4]))
            else:
                self.f.write(
                    stemplate.format(str(i), self.df.values[i][0], self.df.values[i][1], self.df.values[i][2],
                                     self.df.values[i][3] ))

        return

    def plot_ts(self, logfolder, stop_on_chart_show=False):
        """

        :param logfolder:
        :param stop_on_chart_show:
        :return:
        """
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')
        fig,ax=plt.subplots()
        num=0
        df1 = self.df.copy(deep=True)
        years = mdates.YearLocator()
        months = mdates.MonthLocator()
        year_fmt = mdates.DateFormatter('%Y')
        for column in df1.drop(['Date Time'], axis=1):
            num+=1
            ax.plot(df1['Date Time'], df1[column], marker='',color=palette(num), label=column)
        plt.legend(loc=2, ncol=2)
        ax.set_title('{}'.format(self.title))
        fig.autofmt_xdate()

        ax.xaxis.set_major_locator(years)
        # ax.xaxis.set_major_formatter(year_fmt)
        ax.xaxis.set_minor_locator(months)
        datemin = df1['Date Time'].min()
        datemax = df1['Date Time'].max()
        ax.set_xlim(datemin, datemax)
        fig.autofmt_xdate()

        ax.set_xlabel("Time")
        ax.set_ylabel("Power (MW)")
        if self.scaled_data:
             plt.ylabel("Power (0 - 1)")

        #plt.show(block=stop_on_chart_show)


        sfile = "{}.png".format(self.title.replace(' ', '_'))
        sFolder =PlotPrintManager.get_ControlLoggingFolder()
        filePng = Path(sFolder) / (sfile)

        plt.savefig(filePng)
        if PlotPrintManager.isNeedDestroyOpenPlots(): plt.close("all")
        del df1

        return

    def autocorr_show(self, logfolder, stop_on_chart_show=False):
        """

        :param logfolder:
        :param stop_on_chart_show:
        :return:
        """
        if ControlPlane.get_modeImbalance():
            x = self.df[self.names[4]].values
        else:
            x = self.df[self.names[1]].values
        show_autocorr(x, (int)(len(x) / 4), self.title, logfolder, stop_on_chart_show, self.f)
        return

    def to_csv(self,path_to_serfile):
        """

        :param path_to_serfile:
        :return:
        """
        self.df.to_csv(path_to_serfile, index=False)
        nloops=0
        while not os.path.exists(path_to_serfile):
            time.sleep(1)
            nloops +=1
            if nloops>32 :
                break

        if (not os.path.exists(path_to_serfile)) or (not os.path.isfile(path_to_serfile)) :
            msg = "{} file is not ready".format(path_to_serfile)
            msg2log(self.to_csv.__name__, msg,self.f)
            raise ValueError(msg)
            return None

        msg="DataFrame serialized to file {}".format(path_to_serfile)
        msg2log(self.to_csv.__name__, msg,self.f)
        return path_to_serfile

    def concat_with_df_from_csv(self, path_to_serfile):
        """

        :param path_to_serfile:
        :return:
        """

        df_old=pd.read_csv(path_to_serfile)

        message = f"""
                                Old DataFrame (Odf) TS size    : {len(df_old)}
                                Update DataFrame (Udf) TS size : {self.ts_size}
                                Udf TS numbers                 : {self.n_ts}
                                Udf TS names                   : {self.names}
                                Udf Last Time                  : {self.last_time}
                    """
        msg2log(self.concat_with_df_from_csv.__name__, message,self.f)
        df_new_reindex=pd.concat([df_old, self.df], ignore_index=True)
        self.ts_size = len(df_new_reindex)
        message = f"""
                                New DataFrame (Ndf) TS size    : {len(df_new_reindex)}
                                Ndf TS numbers                 : {self.n_ts}
                                Ndf TS names                   : {self.names}
                                Ndf Last Time                  : {self.last_time}
                            """
        msg2log(self.concat_with_df_from_csv.__name__, message, self.f)

        return df_new_reindex

if __name__ == "__main__":
    with open("abc.log", 'w') as flog:
        dwdg = DemandWidget(False, "2020-08-26T00:00", "2020-08-29T18:00", "hour", None, None, flog)
        dwdg.set_url()

        print(dwdg.url)

        dwdg.getDemandRT()
        dwdg.plot_ts(os.getcwd(), False)
        dwdg.autocorr_show(os.getcwd(), False)

        pass
