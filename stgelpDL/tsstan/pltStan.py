#!/usr/bin/python3

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tsstan.stan import l_seas_prhouse, l_seas_slplant


def setPlot():
    large = 22; med = 16; small = 12
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")
    # %matplotlib inline --- this for Jupiter notebook




    # Version
    print(mpl.__version__)
    print(sns.__version__)

def plotAll(name,data_list,data_col_name,file_png_path, f=None):
    # Draw Plot
    fig, ((ax1, ax2),(ax3,_)) = plt.subplots(2, 2,  figsize=(16, 6), dpi=80)
    # plot_acf(df.traffic.tolist(), ax=ax1, lags=50)
    # plot_pacf(df.traffic.tolist(), ax=ax2, lags=20)


    plot_acf(data_list, ax=ax1, lags=12)
    plot_pacf(data_list, ax=ax2, lags=6)
    ax3.plot(data_list)

    # Decorate
    # lighten the borders
    ax1.spines["top"].set_alpha(.3);
    ax2.spines["top"].set_alpha(.3)
    ax3.spines["top"].set_alpha(.3)
    ax1.spines["bottom"].set_alpha(.3);
    ax2.spines["bottom"].set_alpha(.3)
    ax3.spines["bottom"].set_alpha(.3)
    ax1.spines["right"].set_alpha(.3);
    ax2.spines["right"].set_alpha(.3)
    ax3.spines["right"].set_alpha(.3)
    ax1.spines["left"].set_alpha(.3);
    ax2.spines["left"].set_alpha(.3)
    ax3.spines["left"].set_alpha(.3)

    # font size of tick labels
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    ax3.tick_params(axis='both', labelsize=12)
    # legend and titles
    ax1.legend(["ACF"])
    ax2.legend(["PACF"])
    ax3.legend(["Day's data"])
    fig.suptitle(name + " (" + data_col_name + ")")
    # plt.show()
    file_png = Path(Path(file_png_path) / Path(name + "_" + data_col_name + "_ACF_PACF.png"))
    plt.savefig(file_png)
    plt.close("all")


def plotPACF(name,l_d_names, df, data_column_name,file_png_path, f=None):
    # Import Data
    if len(l_d_names)== 0:
        return

    d_item=l_d_names.pop(0)
    [(seasonaly_period, dd_item)] = d_item.items()
    [(data_col_name, lmbd)] = dd_item.items()

    # Draw Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=80)
    # plot_acf(df.traffic.tolist(), ax=ax1, lags=50)
    # plot_pacf(df.traffic.tolist(), ax=ax2, lags=20)

    clearlist = [x for x in df[data_col_name].tolist() if not np.isnan(x)]
    plot_acf(clearlist, ax=ax1, lags=100)
    plot_pacf(clearlist, ax=ax2, lags=50)

    # Decorate
    # lighten the borders
    ax1.spines["top"].set_alpha(.3);
    ax2.spines["top"].set_alpha(.3)
    ax1.spines["bottom"].set_alpha(.3);
    ax2.spines["bottom"].set_alpha(.3)
    ax1.spines["right"].set_alpha(.3);
    ax2.spines["right"].set_alpha(.3)
    ax1.spines["left"].set_alpha(.3);
    ax2.spines["left"].set_alpha(.3)

    # font size of tick labels
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    # legend and titles
    ax1.legend(["ACF"])
    ax2.legend(["PACF"])
    fig.suptitle(name+" ("+ data_col_name +")")
    # plt.show()
    file_png = Path( Path(file_png_path)/ Path(name + "_" + data_col_name+"_ACF_PACF.png"))
    plt.savefig(file_png)
    plt.close("all")
    plotPACF(name, l_d_names, df, None,file_png_path,f)



if __name__ == "__main__":
    pass
    setPlot()
    file_png_path="Logs/predict"
    f=None
    csv_source = "~/LaLaguna/stgelpDL/dataLaLaguna/PrivateHouseElectricityConsumption_21012020_seas.csv"
    df = pd.read_csv(csv_source)
    name="PrivateHouse"
    data_column_name=None
    l_d_names=l_seas_prhouse

    plotPACF(name,l_d_names, df, data_column_name,file_png_path,f)

    csv_source = "~/LaLaguna/stgelpDL/dataLaLaguna/SolarPlantPowerGen_21012020_seas.csv"
    data_col_name = None
    df = pd.read_csv(csv_source)
    name = "SolarPlant"
    data_column_name = None
    l_d_names = l_seas_slplant

    plotPACF(name, l_d_names, df, data_column_name, file_png_path, f)

    pass