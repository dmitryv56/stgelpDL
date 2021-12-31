#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" My first tk GUI

"""

import logging
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import *
import numpy as np
import time
from msrvcpred.app.clients.client_Predictor import PredictorClient
from msrvcpred.cfg import DISCRET, ALL_MODELS



logger=logging.getLogger(__name__)


def getmaxplots():
    n = 0
    for key,type_model_list in ALL_MODELS.items():
        n = n + len(type_model_list)
    logger.info("The timeseries and predicts for {} models are plotted".format(n))
    return n + 1

class ChartGui():
    """ base class """
    def __init__(self,tk_title:str="Tk", geometry:str="900x500",background:str='light blue'):
        self._log=logger
        self.root = tk.Tk()
        self.root.title(tk_title)
        self.root.configure(background='light blue')
        m = None
        try:
            m = self.root.maxsize()
            self.root.geometry('{}x{}+0+0'.format(*m))
        except Exception as ex:
            self._log.error("Exception for {} geometry. {}".format(*m,ex))
            self.root.geometry(geometry)  # set the window size
        finally:
            pass
        self.root.configure(background=background)
        self.lines = [None, None, None, None, None,None,None,None,None]
        self.fig = None
        self.ax = None
        self.ax_title = "Imbalance Forecasting"
        self.ax_xlabel = "Sample"
        self.ax_ylabel = "MWatt"
        self.ax_xlim =(0,50)
        self.ax_ylim = (0.0,100.0)
        self.canvas_x = 180
        self.canvas_y = 10
        self.canvas_width = 1200
        self.canvas_height = 900

        self.var = tk.StringVar()
        self.var_timestamp = tk.StringVar()
        self.figaxes()
        self.figcanvas()


        self.discret_in_sec = 60 * DISCRET
        self._log.info("Discret is {} sec".format(self.discret_in_sec))

    def figaxes(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(self.ax_title)  # ' Serial Data')
        self.ax.set_xlabel(self.ax_xlabel)  # 'Sample')
        self.ax.set_ylabel(self.ax_ylabel)  # 'Voltage')
        self.ax.set_xlim(self.ax_xlim)  # 0, 100)
        self.ax.set_ylim(self.ax_ylim)  # -0.5, 6)
        n_plots = getmaxplots()
        self.lines=[None for i in range(n_plots)]
        for i in range(n_plots):
            self.lines[i]=self.ax.plot([],[])[0]




    def figcanvas(self ):

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea
        self.canvas.get_tk_widget().place(x=self.canvas_x, y=self.canvas_y, width=self.canvas_width,
                                          height=self.canvas_height)
        self.canvas.draw()

    def __widgets(self):
        """ should be implemented in PlotChar"""
        pass





    """ T.B.D.   screensize ????"""
    def screensize(self,row,col):
        self.withdraw()
        self.width = self.winfo_screenwidth()
        self.height =  self.winfo_screenheight()
        width_mm = self.winfo_screenmmwidth()
        height_mm = self.winfo_screenmmheight()
        self.width = int(0.95*self.width)
        self.height = int(0.95*self.height)
        self.geometry("{}x{}".format(row,col))

class PlotChart(ChartGui):
    """ dataplots is a list of DataPlot-objects created for entyre timeseries and for all forecasts models.
    The forecasting models are listed in msrvcpred.cfg.ALL_MODELS.
    """

    def __init__(self, tk_title:str="Short-time Green Energy Load Predictor",geometry:str="900x500"):
        super().__init__(tk_title=tk_title,geometry=geometry)
        self._log.info("{} initialized".format(super().__class__.__name__))
        self.models = []
        self.dataplots = []
        self.d_currdata = {}
        self.curr_index = 0
        self.start_x = 0
        self.cond = False
        self.obtained = False
        self.chart_size = 50
        self.widgets()   # self.models is filled into
        self._log.info("Widgets added")
        self._log.info("list models obtained")


    def widgets(self):
        # ----------create button-----------
        self.root.update()
        self.start = tk.Button(self.root, text='Start', font=('calibri', 12), command=lambda: self.plot_start())
        self.start.place(x=100, y=10 + self.canvas_y + self.canvas_height+10)

        self.root.update()
        self.stop = tk.Button(self.root, text='Stop', font=('calibri', 12), command=lambda: self.plot_stop())
        self.stop.place(x=self.start.winfo_x() + self.start.winfo_reqwidth() + 20, y=self.start.winfo_y())

        self.root.update()
        self.timestamp = tk.Label(self.root, textvariable=self.var_timestamp, font=('calibri', 12), width=21)
        self.timestamp.place(x=self.stop.winfo_x() + self.stop.winfo_reqwidth() + 20, y=self.start.winfo_y())

        self.root.update()
        self.buy = tk.Label(self.root, textvariable=self.var, font=('calibri', 12))
        self.buy.place(x=self.timestamp.winfo_x() + self.timestamp.winfo_reqwidth() + 20, y=self.start.winfo_y())

        self.root.update()

        self.listbox = tk.Listbox(self.root, width=20, height=49,selectmode=MULTIPLE)
        self.listbox.place(x=self.canvas_x + self.canvas_width +5, y=10)
        self.get_models()

        self.root.update()
        self.obtain = tk.Button(self.root, text="Obtain Historical data", font=('calibri', 12),
                             command=lambda: self.obtain_data())
        self.obtain.place(x=self.canvas_x + self.canvas_width +5, y=self.start.winfo_y())

    def run(self):

        # self.root.after(1000* self.discret_in_sec, self.plot_data)
        self.root.after(1000*20 , self.plot_data)
        self._log.info("Mainloop is running")
        self.root.mainloop()

    def selectedmodels(self):
        if self.cond == True:
            return
        for i in self.listbox.curselection():
            self.models.append(self.listbox.get(i))




    def plot_data(self):
        """
        if condition satisfied, the grpc client is created. This client requests prediction data from server.
        The client returns a list of predict sample timestamps and a actual predict data as dictionary. The dict has a
        following structure:
        {'0':[predicts],
         '1':[predicts],
         ...
         '<n>':[predicts]},
         where <predicts> for '0' -key are real samples of predicted time series, may content NaN for future timestamp,
               <predicts> for '1' -key are predicts for first model (see ALL_MODELS -dict)m and e.t.c.
        :return:
        """
        bShift = False
        if self.cond == True:
            iplot = 0
            l_var=[]
            msg =""

            client = PredictorClient()
            dt, d_predict_data = client.get_predicts("Get predicts")
            if not dt or not d_predict_data:
                self._log.info(" No data received")
                self.root.after(1000 * 20, self.plot_data)
                return

            ymin=1e+25
            ymax=-1e+25
            for dp in self.dataplots:

                # dp =self.data4plot( dp,dt,d_predict_data[str(iplot)])
                dp.updateplotdata(dt, d_predict_data[str(iplot)])

                ymin=min(ymin, dp.data.min())
                ymax=max(ymax, dp.data.max())

                if len(dp.data)>=self.chart_size:
                    bShift = True

                start_line = self.start_x
                end_line = self.start_x + len(dp.data)
                x = np.arange(start_line, end_line)
                self.lines[iplot].set_xdata(x)
                self.lines[iplot].set_ydata(dp.data)

                self.var_timestamp.set(dt[-1])
                iplot=iplot + 1

            self.var_timestamp.set(msg)
            self.var.set("")
            if len(l_var)>0:
                self.var.set(l_var)
            handles, _ = self.ax.get_legend_handles_labels()
            self.fig.legend(handles=handles, labels=self.models)

            self.ax.set_xlim(self.start_x, self.start_x + self.chart_size)
            delta =(ymax-ymin) * 0.1  # 10% of the range

            self.ax.set_ylim(ymin-delta, ymax+delta)
            self.canvas.draw()

            if bShift:
                self.start_x = self.start_x + 1
            self.curr_index = self.curr_index + 1

        # self.root.after(1000*self.discret_in_sec, self.plot_data)
        self.root.after(1000 * 20, self.plot_data) # Change after debug


    def plot_start(self):

        if self.obtained == False:
            return
        self.cond = True


    def plot_stop(self):
        # global cond
        self.cond = False

    def obtain_data(self):
        self.cond = False
        self._log.info("list of used models: {}".format(self.models))
        self.selectedmodels()
        self._log.info("Selected models{}".format(self.models))
        if len(self.models) > 0:
            self.cond = True
        # amin=1500000.0
        # amax=0.0
        self.ax_title ="Short-time {} prediction".format(self.models[0])
        self.var_timestamp.set("")
        len_history= {}

        for item in self.models:

            self.dataplots.append(DataPlot(item))
            self.d_currdata[item] = []




        # print (len_history)
        # self._log.info(len_history)
        # self._log.info("Shortest history is {}".format(self.min_len_ts))
        # self.var.set("")
        self.ax_ylim = (0.0,100.0)
        self.ax.set_ylim(self.ax_ylim)
        self.ax.set_title(self.ax_title)

        self.canvas.draw()

        self.obtained = True


    def get_models(self):

        client = PredictorClient()
        d_res = client.get_titles("model titles")
        self.ax.set_xlabel(d_res['x_label'])
        self.models=d_res['models']


        i=0
        for item in self.models:
            i=i+1
            self.listbox.insert(i,item)
        return i

class DataPlot():
    """ class """
    def __init__(self,model:str="", chart_size:int = 100 ):
        self._log = logger
        self.model = model
        self.data = np.array([]).astype('float32')
        self.dtdt = np.array([]).astype('str')
        self.chart_size = chart_size

    def updateplotdata(self,dt:list=[],data:list=[]):

        if not dt or not data:
            return
        last_index = self.chart_size - 1
        for i in range(len(data)):
            if len(self.data)<self.chart_size:
                self.data = np.append(self.data,data[i])
                self.dtdt = np.append(self.dtdt, dt[i])
            else:
                self.data[0:last_index] = self.data[1:self.chart_size]
                self.data[last_index] = data[i]
                self.dtdt[0:last_index] = self.dtdt[1:self.chart_size]
                self.dtdt[last_index] = dt[i]







# #-------Main GUI code ------
# root=tk.Tk()
# root.title('Real Time plot')
# root.configure(background = 'light blue')
# root.geometry("900x500")  # set the window size
#
# #------create Plot object on GUI---------
# # add figure canvas
# fig = Figure()
# ax = fig.add_subplot(111)
#
# #ax = plt.axes(xlim=(0,100),ylim=(0,120))  # display only 100 sam..
# ax.set_title('Serial Data')
# ax.set_xlabel('Sample')
# ax.set_ylabel('Voltage')
# ax.set_xlim(0,100)
# ax.set_ylim(-0.5,6)
# lines = ax.plot([],[])[0]
#
# canvas=FigureCanvasTkAgg(fig,master=root) # A tk.DrawingArea
# canvas.get_tk_widget().place(x = 10,y = 10, width =500, height = 400)
# canvas.draw()
#
# #----------create button-----------
# root.update()
# start = tk.Button(root, text = 'Start', font=('calibri',12),command = lambda: plot_start())
# start.place(x = 100, y = 450)
#
# root.update()
# stop = tk.Button(root, text = 'Stop', font=('calibri',12), command = lambda: plot_stop())
# stop.place(x=start.winfo_x()+start.winfo_reqwidth() +20, y = 450)
#
# #----start serial port -------
# # s = sr.Serial('COM8',115200)
# # s.reset_input_buffer()
#
#
#
#
#
#
#
# root.after(1,plot_data)
# root.mainloop()


if __name__=="__main__":
    # root=tk.Tk()
    # root.geometry("900x500")
    # root.mainloop()
    pltf=PlotChart()
    pltf.get_models()
    pltf.run()
