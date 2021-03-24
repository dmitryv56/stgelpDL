#!/usr/bin/python3

""" transform time series (feature in source dataset) to matrix Nrows * Mcols.

The source dataset must comprise  a timestamp feature(column) named 'dt_col_name' or index column. The 'data_col_name'-
feature must be a time series (TS) are ordered by  index ot timestamp feature('dt_col_name') , that is, the observation
must be equidistant and without gaps. Additional, the beginning timestamp must be 00:00:00 (or 0 for index).
The 'period' and 'direction' determine the formation of the matrix.
If the 'direction' is along the 'X'- axis, then the segments of the TS corresponding to the 'period' are the
rows of the matrix. Column names are derived from  observation times within a period, for example, "00: 00,00: 10, ...,
23:50" to 10 minutes discretization and a period of 1 day.
If the direction is along 'Y'-axis, then the segment of  the TS corresponding to the 'period' are the column of matrix.
The column names are derived from the period in the timestamp, i.g. data string if the period is 1 day like as
'2016-03-27', and row names are derived from observations within period, for example, "00:00,...,23:50".


"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import math
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from stcgelDL.cfg import GlobalConst
from predictor.utility import msg2log

__version__="0.0.1"
def parser() -> (argparse.ArgumentParser, str):
    # command-line parser
    sDescriptor = "Transform time series (TS,feature in source dataset) to matrix according by given direction."

    sCSVhelp = "Absolute path to a source dataset (csv-file)."
    sTSnameHelp = "The time series (feature in the dataset). "
    sTimeStampHelp = "Timestamp (column) name in the dataset."
    sdiscretHelp = "TS sampling (discretization) in minutes."
    sdirectionHelp = "Determine a formation of the matrix."
    speriodHelp = "TS segment size if the sampling resolution for formation rows or columns of the matrix."
    soutrowindexnameHelp = "Out row index name."
    sscaleHelp = "Scale time series, element by element, using 'center', that is, z=(x-mean),'standard' is "+ \
        " z=(x-mean)/std, 'minmax' to range (0,1)  is z=(x-minX)/(maxX-minX)+minx, 'absmax' to range (-1,1) is " + \
        " z=2 *(x-minX)/(maxX-minX)-1. By default, no normalize"
    stitleHelp = "Title, one word using as log folder name."

    parser = argparse.ArgumentParser(description=sDescriptor)


    parser.add_argument('-c', '--csv_dataset', dest='cl_dset', action='store', help=sCSVhelp)
    parser.add_argument('-t', '--ts', dest='cl_ts', action='store', default='',   help=sTSnameHelp)
    parser.add_argument('--timestamp', dest='cl_timestamp', action='store', default='Date Time',  help=sTimeStampHelp)
    parser.add_argument('-r', '--direction', dest='cl_direction', action='store', default='X', help=sdirectionHelp)
    parser.add_argument('-d', '--discret', dest='cl_discret', action='store', default=10, help=sdiscretHelp)
    parser.add_argument('-p', '--period', dest='cl_period', action='store', default=144,        help=speriodHelp)
    parser.add_argument('-s', '--scale', dest='cl_scale', action='store', default="NO",
                        choices=['NO','center','standard','minmax','maxabs'], help=sscaleHelp)
    parser.add_argument('-o', '--outrowind', dest='cl_outrowind', action='store', default="rowNames",
                        help=soutrowindexnameHelp)
    parser.add_argument('-i', '--title', dest='cl_title', action='store', default='ElHiero', help=stitleHelp)

    parser.add_argument('--verbose', '-v', dest='cl_verbose', action='count', default=0)
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()
    # command-line parser
    arglist = sys.argv[0:]
    argstr = ""
    for arg in arglist:
        argstr += " " + arg
    message0 = f"""  Command-line arguments

{argstr}
Dataset path                       : {args.cl_dset}
Time Series in dataset             : {args.cl_ts}
Timestamp in dataset               : {args.cl_timestamp} 
Discretization (min)               : {args.cl_discret}
Direction formation out matrix     : {args.cl_direction}
Period formation out matrix        : {args.cl_period}
Out row index name                 : {args.cl_outrowind}
Scale time series                  : {args.cl_scale}
Title:                             : {args.cl_title}
"""

    return args, message0


def logParams(param: tuple) -> str:
    (csv_source, dt_col_name, data_col_name, discret, dt_start,n, direction, period, outrowind, scale, title, \
     folder_for_logging)  = param
    nret=0
    nseg=int(n/period)
    msg=""
    if n%period !=0:
        msg="Warning!!!!! TS size {} should be multiply of segment size {}. Please correct time series data.".format( n,
             nseg)
        nret+=1

    sstart =pd.to_datetime(dt_start,dayfirst=True).time().strftime('%H:%M:%S')
    msg1=""
    if "00:00:00" not in sstart:
        msg1="Warning!!!!! TS should start at midnt., 00:00:00."
        nret+=1

    message0 = f"""  Task parameters

Dataset path                       : {csv_source}
Time Series in dataset             : {data_col_name}
Timestamp in dataset               : {dt_col_name} 
Time series length                 : {n}
Time series length                 : {dt_start}
Number of segments                 : {nseg}
Discretization (min)               : {discret}
Direction formation out matrix     : {direction}
Period formation out matrix        : {period}
Out row index name                 : {outrowind}
Scale time series                  : {scale}
Title:                             : {title}
Folder for logging                 : {folder_for_logging}
    {msg}
    {msg1}

    """
    return nret, message0


def ts2matrix(source_csv: str = None,  dt_col_name: str = "Date Time",
              data_col_name: str = 'Real_demand', outRowIndex="rowNames", discret: int = 10, period: object = None,
              direction: str = 'x', title: str = "", f: object = None):
    if source_csv is None or not Path(source_csv).exists():
        msg = "The source dataset path is invalid:  {}".format(source_csv)
        msg2log(None, msg, f)
        return None

    ds = pd.read_csv(source_csv)
    if dt_col_name not in ds.columns or data_col_name not in ds.columns:
        msg = "{} or {} not found in the dataset {}".format(dt_col_name, data_col_name, source_csv)
        msg2log(None, msg, f)
        return None
    folder_csv = Path(source_csv).parent
    dest_csv = Path(folder_csv / "{}_{}".format(title, data_col_name)).with_suffix(".csv")
    dest_se_csv = Path(folder_csv / "{}_{}_StatEst".format(title, data_col_name)).with_suffix(".csv")

    if direction == "X" or direction == "x":
        dsMatrix = ts2matrix_X(ds=ds, dt_col_name=dt_col_name, data_col_name=data_col_name, outRowIndex=outRowIndex,
                               discret=discret, period=period, f=f)
    elif direction == "Y" or direction == "y":
        dsMatrix = ts2matrix_Y(ds=ds, dt_col_name=dt_col_name, data_col_name=data_col_name, outRowIndex=outRowIndex,
                               discret=discret, period=period, f=f)
    else:
        msg = "{} invalid direction ".format(direction)
        msg2log(None, msg, f)
        return None
    dsMatrix.to_csv(dest_csv)
    message =f"""
'Matrix view' of {data_col_name} time series saved in {dest_csv}.
Number rows: {len(dsMatrix)}    Number columns: {len(dsMatrix.columns)} with indexes .
"""
    msg2log(None,message, f)

    dsMatrixStatEst = ts2matrix_statest(ds=dsMatrix, rowIndexName=outRowIndex, f=f)
    dsMatrixStatEst.to_csv(dest_se_csv)

    message = f"""
    'Matrix view' with primary statistics of {data_col_name} time series saved in {dest_se_csv}
    Number rows: {len(dsMatrixStatEst)}    Number columns: {len(dsMatrixStatEst.columns)} with indexes.
    """
    msg2log(None, message, f)

    return


def ts2matrix_X(ds: pd.DataFrame = None, dt_col_name: str = 'Date Time', data_col_name: str = 'Real_demand',
                outRowIndex: str = "rowNames", discret: int = 10, period: int = 144, f: object = None) -> pd.DataFrame:
    # src_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv"
    # dst_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_Days.csv"
    # ds = pd.read_csv(src_csv)
    # # col_name = 'lasts'
    # dt_col_name = 'Date Time'
    # data_col_name = 'Real_demand'
    h_period_sr = 60 / discret  # hour in sample resolution, 6
    d_period_sr = period  # day period in sample resolution, 144
    v = ds[data_col_name].values
    n_v = len(v)
    if n_v % d_period_sr != 0:
        msg = "Time series size is {} and it is not multiply of {} period".format(n_v, d_period_sr)
        msg2log(None, msg, f)
        return None

    col_names = ["{}:{}".format(int(i / h_period_sr), int((i % h_period_sr) * 10)) for i in range(d_period_sr)]
    v_dict = {col_names[i]: [] for i in range(d_period_sr)}

    d_dict = {outRowIndex: [pd.to_datetime(ds[dt_col_name][i], dayfirst=True).date().strftime('%Y-%m-%d') \
                            for i in range(0, n_v, d_period_sr)]}

    for i in range(n_v):
        v_dict[col_names[i % 144]].append(v[i])

    d = {**d_dict, **v_dict}
    ds1 = pd.DataFrame(d)
    # ds1.to_csv(dst_csv)
    return ds1


def ts2matrix_Y(ds: pd.DataFrame = None, dt_col_name: str = 'Date Time', data_col_name: str = 'Real_demand',
                outRowIndex: str = "rowNames", discret: int = 10, period: int = 144, f: object = None) -> pd.DataFrame:
    # src_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv"
    # dst_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_TimeStamps.csv"
    # ds = pd.read_csv(src_csv)
    # # col_name = 'lasts'
    # dt_col_name = 'Date Time'
    # data_col_name = 'Real_demand'
    #
    #

    v = ds[data_col_name].values
    n_v = len(v)

    h_period_sr = 60 / discret  # hour in sample resolution, 6
    d_period_sr = period  # day period in sample resolution, 144
    v = ds[data_col_name].values
    n_v = len(v)
    if n_v % d_period_sr != 0:
        msg = "Time series size is {} and it is not multiply of {} period".format(n_v, d_period_sr)
        msg2log(None, msg, f)
        return None

    col_names = [pd.to_datetime(ds[dt_col_name][i], dayfirst=True).date().strftime('%Y-%m-%d') for i in
                 range(0, n_v, d_period_sr)]
    row_names = ["{}:{}".format(int(i / h_period_sr), int((i % h_period_sr) * 10)) for i in range(d_period_sr)]
    v_dict = {}

    d_dict = {outRowIndex: row_names}
    n_start = 0
    n_series = d_period_sr  # obsefvation points
    n_features = 366  # days
    for icol in col_names:
        v_dict[icol] = v[n_start:n_start + n_series].tolist()
        n_start = n_start + n_series

    d = {**d_dict, **v_dict}
    ds1 = pd.DataFrame(d)
    # ds1.to_csv(dst_csv)
    return ds1


def ts2matrix_statest(ds: pd.DataFrame = None, rowIndexName: str = "rowNames", f: object = None):
    # dst_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016-Est.csv"
    # src_csv="~/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_TimeStamps.csv"
    # ds = pd.read_csv(src_csv)
    # col_name = 'lasts'
    dt_col_name = rowIndexName

    d_dst = {}

    vts = ds[rowIndexName].values.tolist()
    vts.append('ave')
    vts.append('std')
    vts.append('min')
    vts.append('max')
    d_dst[rowIndexName] = vts
    for col in ds.columns:
        if 'Unnamed' in col or rowIndexName in col: continue
        v = ds[col].values.tolist()
        a = np.array(v)
        ave = a.mean()
        std = a.std()
        minv = a.min()
        maxv = a.max()
        v.append(round(ave, 3))
        v.append(round(std, 3))
        v.append(round(minv, 2))
        v.append(round(maxv, 2))
        d_dst[col] = v

    ds1 = pd.DataFrame(d_dst)
    n_ds1 = len(ds1)

    ave = []
    std = []
    minv = []
    maxv = []
    for i in range(n_ds1):

        v = []
        for col in ds1.columns[1:]:
            v.append(ds1[col][i])
        a = np.array(v)
        ave.append(round(a.mean(), 3))
        std.append(round(a.std(), 3))
        minv.append(round(a.min(), 2))
        maxv.append(round(a.max(), 2))
    ds1['ave'] = ave
    ds1['std'] = std
    ds1['min'] = minv
    ds1['max'] = maxv

    # ds1.to_csv(dst_csv)
    return ds1

def main(argc,argv):
    args, message0 = parser()

    title = args.cl_title

    csv_source    = args.cl_dset
    dt_col_name   = args.cl_timestamp  # "Date Time"
    data_col_name = args.cl_ts
    discret       = int(args.cl_discret)
    direction     = args.cl_direction
    period        = int(args.cl_period)
    outRowIndexName = args.cl_outrowind
    scale           = args.cl_scale
    title           = args.cl_title
    GlobalConst.setVerbose(args.cl_verbose)

    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(title, date_time)

    listLogSet(str(folder_for_logging))  # A logs are creating
    msg2log(None, message0, D_LOGS["clargs"])
    msg2log(None, message1, D_LOGS["timeexec"])
    fc = D_LOGS["control"]

    folder_for_logging = str(Path(os.path.realpath(fc.name)).parent)
    msg=""
    try:
        df = pd.read_csv(csv_source)
        n = len(df)
        dt_start=df[dt_col_name][0]
    except:
        n=0
        msg = "O-o-ops! I got an unexpected error - reason  {}\n".format(sys.exc_info())
    finally:
        if len(msg) > 0:
            msg2log(main.__name__, msg, D_LOGS["except"])
            log2All()
            closeLogs()
            return -2
        param=(csv_source, dt_col_name, data_col_name, discret, dt_start, n, direction, period, outRowIndexName, scale,
               title,  folder_for_logging)
        nret, msg=logParams(param)
        msg2log(None,logParams(param),D_LOGS["main"])
        if n==0 or nret!=0:
            closeLogs()
            return -1
    """ scaling """
    del df
    df=None
    if 'NO' not in scale:
        folder_csv=Path(csv_source).parent
        file_stem=Path(csv_source).stem
        bkp_csv=str(Path(folder_csv)/Path("backup_{}".format(file_stem)).with_suffix(".csv"))

        scaled_data_col_name =tsScale(source_csv=csv_source,bkp_csv=bkp_csv, data_col_name=data_col_name,scale=scale,
                                      f=D_LOGS['block'])
        if len(scaled_data_col_name)>0:
            data_col_name=scaled_data_col_name
    if direction == "x" or direction =="X":
        ts2matrix(source_csv=csv_source, dt_col_name=dt_col_name, data_col_name=data_col_name,
              outRowIndex=outRowIndexName, discret=10, period=144, direction='x', title="X_direction",
              f=D_LOGS['control'])
        log2All()
    if direction == "y" or direction == "Y":
        ts2matrix(source_csv=csv_source, dt_col_name=dt_col_name, data_col_name=data_col_name,
              outRowIndex=outRowIndexName, discret=10, period=144, direction='y', title="Y_direction",
              f=D_LOGS['control'])
        log2All()

    message = "Time execution logging stoped at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))
    msg2log(None, message, D_LOGS["timeexec"])
    closeLogs()  # The loga are closing


    return 0

def tsScale(source_csv:str=None, bkp_csv:str=None,data_col_name:str=None, scale:str='no',f:object=None)->int:

    scaled_data_col_name=""
    df = pd.read_csv(source_csv)

    v=df[data_col_name].values
    vmean=v.mean()
    vstd = v.std()
    vmin=v.min()
    vmax=v.max()
    n=len(v)
    message=f""" 
Source Time Series: {data_col_name}
Time Series length: {n}
Min               : {vmin}
Max               : {vmax}
Mean              : {vmean}
Std               : {vstd}
"""
    msg2log(None,message,f)
    #'center', 'standard', 'minmax', 'maxabs'
    if 'center' in scale:
        v=v-vmean
    elif 'standard' in scale:
        v=v-vmean
        v=v/vstd
    elif 'minmax' in scale:
        v=(v-vmin)/(vmax-vmin)
    elif 'maxabs' in scale:
        v=2.0*(v-vmin)/(vmax-vmin) -1.0
    else:
        msg2log(None,"{} no valid scale parameter".format(scale),f)
        return scaled_data_col_name

    v=np.round(v,4)
    df.to_csv(bkp_csv, index=False)
    msg2log(None,"Backup of the source dataset saved in {}".format(bkp_csv),f)
    scaled_data_col_name="{}_{}".format(scale,data_col_name)
    df[scaled_data_col_name]=v.tolist()

    msg2log(None, "{} - scaled time series column name is {} ".format(scale, scaled_data_col_name), f)
    df.to_csv(source_csv,index=False)
    msg2log(None,"Source dataset updated in {}".format(source_csv),f)
    return scaled_data_col_name

    




if __name__ == "__main__":
    pass
    # src_csv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016.csv"
    #
    # with open("loglog.log", 'w+') as ff:
    #     ts2matrix(source_csv=src_csv, dt_col_name="Date Time", data_col_name='Real_demand',
    #               outRowIndex="rowNames", discret=10, period=144, direction='x', title="X_direction", f=ff)
    #     ts2matrix(source_csv=src_csv, dt_col_name="Date Time", data_col_name='Real_demand',
    #               outRowIndex="rowNames", discret=10, period=144, direction='y', title="Y_direction", f=ff)
    # pass
    #
    # nret =main(len(sys.argv),sys.argv)
    #
    # sys.exit(nret)