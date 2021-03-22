#!/usr/bin/python3

# !/usr/bin/python3
""" This package contains  an implementation of a hidden Markov model (hmm).

"""

import copy
import os
import sys
from datetime import datetime
from math import isnan
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tpb
from hmm.fwd import fwd_bkw, viterbi, probNormDist
from matplotlib import pyplot as plt

from hmm.HMM import hmm_gmix
from predictor.cfg import PATH_DATASET_REPOSITORY, DT_DSET, DISCRET, LOG_FILE_NAME, \
    PROGRAMMED_NAME, DEMAND_NAME, CONTROL_PATH, PREDICT_PATH, PATH_REPOSITORY, MODE_IMBALANCE, IMBALANCE_NAME, \
    TS_DURATION_DAYS, SEGMENT_SIZE, RCPOWER_DSET_AUTO
from predictor.control import ControlPlane
from predictor.utility import msg2log, PlotPrintManager, cSFMT, logDictArima

LEARN_HMM = 0
PREDICT_HMM = 1


def input4FwdBkw(states: tuple, observation_labels: np.array, ts_array: np.array, pai: np.array,
                 transDist: np.array, emisDist: np.array, f: object) -> (dict, dict, dict):
    """

    :param states:
    :param observation_labels:
    :param ts_array:
    :param pai:
    :param transDist:
    :param emisDist:
    :param f:
    :return:
    """

    d_start_probability = {}
    d_transDist = {}
    d_emisDist = {}
    for i in range(len(states)):

        d_start_probability[states[i]] = pai[i]

        d_transDist[states[i]] = {}
        for j in range(len(states)):
            d_transDist[states[i]][states[j]] = transDist[i][j]

        d_emisDist[states[i]] = {}
        for k in range(len(observation_labels)):
            d_emisDist[states[i]][observation_labels[k]] = probNormDist(ts_array[k], emisDist[i][0], emisDist[i][1])

    logDictArima(d_start_probability, 4, f)
    logDictArima(d_emisDist, 4, f)
    logDictArima(d_transDist, 4, f)

    return (d_start_probability, d_transDist, d_emisDist)


""" Gets all files (csv-datasets ) from archieve-directory
"""


def listCSV(path_to_repository: Path, parent_dataset_folder: str, child_dataset_folder: str) -> (str, list):
    """
    For example, mypath = str(Path(PATH_DATASET_REPOSITORY / "Demanda_el_huerro" / "Archieve"))

    :param path_to_repository:
    :param parent_dataset_folder:
    :param child_dataset_folder:
    :return:
    """

    if child_dataset_folder is None or child_dataset_folder == "":
        mypath = str(Path(PATH_DATASET_REPOSITORY / parent_dataset_folder))
    else:
        mypath = str(Path(PATH_DATASET_REPOSITORY / parent_dataset_folder / child_dataset_folder))
    # mypath = str(Path(path_to_repository / "Demanda_el_huerro" / "Archieve"))
    fileList = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    return (mypath, fileList)


def predictDataset(cp: ControlPlane, source_folder: str, source_file: str) -> (np.array, np.array, np.array):
    cp.csv_path = Path(Path(source_folder) / Path(source_file))

    (states_aux, d_transitions, d_count, ret_dict, farray, obs_labels, a_sign_aux) = parseDataset(cp, 0)
    observations = copy.copy(farray)  # farray[:6])
    observation_labels = copy.copy(obs_labels)

    (states, hidden_states) = statesRecoding(states_aux, a_sign_aux, cp.fc)

    return (observations, observation_labels, hidden_states)


def predictPath(cp: ControlPlane, source_folder: str, source_file: str):
    (observations, observation_labels, hidden_states) = predictDataset(cp, source_folder, source_file)

    hmm_model = hmm_gmix(2, [0, 1], cp.fc)

    pathHMMName = Path(Path(cp.path_repository) / Path("hmm") / Path("continuos_obs"))

    fileName = Path(pathHMMName / "hmm.json")
    if False == fileName.is_file():
        return

    hmm_model.loadModel(fileName)

    (pai, transitDist, emisDist) = hmm_model.getModel()

    drive_HMM(cp, pai, transitDist, emisDist, observations, observation_labels, None)
    drive_HMM(cp, pai, transitDist, emisDist, observations, observation_labels, hidden_states)

    return


def trainDataset(cp: ControlPlane, source_folder: str, LISTCSV: list) -> (
list, np.array, list, np.array, np.array, np.array):
    """

    :param cp:
    :param source_folder:
    :param LISTCSV:
    :return:
    """

    pass
    dict_csv_ind = {}
    dict_parsed_csv = {}
    csv_ind = 0
    d_total_count = patternCountDict()

    concat_list = []
    states_concat_list = []
    observations = None
    observation_labels = None
    states = None
    a_sign = None
    for item in LISTCSV:
        CSV_PATH = Path(item)
        dict_csv_ind[item] = csv_ind
        cp.csv_path = Path(source_folder / CSV_PATH)

        (states_aux, d_transitions, d_count, ret_dict, farray, obs_labels, a_sign_aux) = parseDataset(cp, csv_ind)
        concat_list.append(farray)
        dict_parsed_csv[csv_ind] = (d_transitions, d_count, ret_dict)

        d_total_count["--"] += d_count["--"]
        d_total_count["-0"] += d_count["-0"]
        d_total_count["-+"] += d_count["-+"]
        d_total_count["0-"] += d_count["0-"]
        d_total_count["00"] += d_count["00"]
        d_total_count["0+"] += d_count["0+"]
        d_total_count["+-"] += d_count["+-"]
        d_total_count["+0"] += d_count["+0"]
        d_total_count["++"] += d_count["++"]

        d_total_count["Start-"] += d_count["Start-"]
        d_total_count["Start0"] += d_count["Start0"]
        d_total_count["Start+"] += d_count["Start+"]

        logTransitions(item, d_count, cp.fp)
        msg2log(trainDataset.__name__, "csv-file: {}\n".format(item), cp.fc)
        logDictArima(dict_parsed_csv, indent=8, f=cp.fc)
        csv_ind += 1
        observations = copy.copy(farray)  # farray[:6])
        observation_labels = copy.copy(obs_labels)
        states = copy.copy(states_aux)
        a_sign = copy.copy(a_sign_aux)
        (states, states_set) = statesRecoding(states, a_sign, cp.fc)
        states_concat_list.append(states_set)

    logTransitions("All datasets", d_total_count, cp.fp)

    return (concat_list, states, states_concat_list, states_set, observations, observation_labels)


def statesRecoding(states_in: np.array, a_sign: np.array, f: object = None) -> (np.array, np.array):
    """

    :param states_in:
    :param a_sign:
    :param f:
    :return:
    """
    states = np.zeros((states_in.shape), dtype=int)
    states_train = np.zeros((a_sign.shape), dtype=int)
    for i in range(len(states_in)):
        states[i] = i

    if f is not None:
        msg = 'The states recoding as in tensorflow_probability.hmm'
        f.write(msg)
        for i in range(len(states_in)):
            msg = '{} => {}\n'.format(states_in[i], states[i])
            f.write(msg)
        msg = '\n## Old  New state name\n'
        f.write(msg)
    for i in range(len(a_sign)):
        for j in range(len(states_in)):
            if a_sign[i] == states_in[j]:
                states_train[i] = states[j]
                break

        if f is not None:
            msg = '{}: {} => {}\n'.format(i, a_sign[i], states_train[i])
            f.write(msg)
    return (states, states_train)


def trainPath(cp, mypath, LISTCSV):
    (concat_list, states, states_concat_list, states_set, observations, observation_labels) = \
        trainDataset(cp, mypath, LISTCSV)

    (pai, transDist, emisDist) = trainHMMprob(concat_list, states_concat_list, cp)

    hmm_model = hmm_gmix(len(states), states, cp.fc)
    hmm_model.setInitialDist(pai)
    hmm_model.setTransitDist(transDist)
    hmm_model.setEmisDist(emisDist)
    pathHMMName = Path(Path(cp.path_repository) / Path("hmm") / Path("continuos_obs"))
    pathHMMName.mkdir(parents=True, exist_ok=True)
    fileName = Path(pathHMMName / "hmm.json")
    hmm_model.dumpModel(fileName)

    msg2log(None, "Test sequence: Imbalance and 'Hidden' states\n", cp.fp)
    test_cutof = 64
    for i in range(len(observations[test_cutof:])):
        s = '{} {}'.format(observations[i], states_set[i])
        msg2log(None, s, cp.fp)

    drive_HMM(cp, pai, transDist, emisDist, observations[test_cutof:], observation_labels[test_cutof:],
              states_set[test_cutof:])
    drive_HMM(cp, pai, transDist, emisDist, observations, observation_labels, states_set)

    aux_states = [str(states[i]) for i in range(len(states))]
    del states
    states = tuple(aux_states)
    msg2log(main.__name__, "Prepare an input data for Forward-backward and Viterbi  algorithms\n", cp.fp)

    (d_start_prob, d_transDist, d_emisDist) = input4FwdBkw(states, observation_labels[test_cutof:],
                                                           observations[test_cutof:], pai, transDist, emisDist, cp.fc)

    msg2log(main.__name__, "\nForward-backward algorithm\n", cp.fp)
    fwd, bkw, posterior = fwd_bkw(observation_labels[test_cutof:], states, d_start_prob, d_transDist, d_emisDist,
                                  states[len(states) - 1], cp.fp)
    msg2log(main.__name__, "\nAposterior probability for each timestamp\n", cp.fp)
    logDictArima(posterior, 4, cp.fp)
    print(fwd)
    print(bkw)

    logFwdBkwDict(fwd, bkw, observation_labels[test_cutof:], cp.fp)

    msg2log(main.__name__, "\nViterbi algorithm\n", cp.fp)
    opt, max_prob, V = viterbi(observation_labels[test_cutof:], states, d_start_prob, d_transDist, d_emisDist, cp.fp)

    logViterbi(V, cp.fp)

    cp.fp.write('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

    plotViterbiPath("fwd", observation_labels[test_cutof:], opt, states_set[test_cutof:], cp)

    return


def main():
    """

    :return:
    """
    workmode=LEARN_HMM
    # workmode = PREDICT_HMM
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    with open("execution_time.log", 'w') as fel:
        fel.write("Time execution logging started at {}\n\n".format(datetime.now().strftime(cSFMT)))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # TODO make path
    if workmode == LEARN_HMM:
        (mypath, LISTCSV) = listCSV(PATH_DATASET_REPOSITORY, "Demanda_el_huerro", "Archieve")
        folder_for_control_logging = Path(dir_path) / "Logs" / CONTROL_PATH / date_time
        folder_for_predict_logging = Path(dir_path) / "Logs" / PREDICT_PATH / date_time
    elif workmode == PREDICT_HMM:
        (mypath, LISTCSV) = listCSV(PATH_DATASET_REPOSITORY, "Demanda_el_huerro", "Predict")
        folder_for_control_logging = Path(dir_path) / "HMM_Logs" / CONTROL_PATH / date_time
        folder_for_predict_logging = Path(dir_path) / "HMM_Logs" / PREDICT_PATH / date_time
    else:
        sys.exit(1)

    CSV_PATH = Path(PATH_DATASET_REPOSITORY / "Demanda_el_huerro" / "Imbalance.csv")
    RCPOWER_DSET = RCPOWER_DSET_AUTO.replace(' ', '_')

    # folder_for_control_logging = Path(dir_path) / "Logs" / CONTROL_PATH / date_time
    # folder_for_predict_logging = Path(dir_path) / "Logs" / PREDICT_PATH / date_time
    folder_for_rt_datasets = Path(PATH_DATASET_REPOSITORY)
    Path(folder_for_control_logging).mkdir(parents=True, exist_ok=True)
    Path(folder_for_predict_logging).mkdir(parents=True, exist_ok=True)
    PlotPrintManager.set_Logfolders(folder_for_control_logging, folder_for_predict_logging)

    suffics = ".log"
    sRCPOWER_DSET = RCPOWER_DSET
    file_for_predict_logging = Path(folder_for_predict_logging, sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    file_for_control_logging = Path(folder_for_control_logging, sRCPOWER_DSET + "_" + Path(__file__).stem).with_suffix(
        suffics)
    fp = open(file_for_predict_logging, 'w+')
    fc = open(file_for_control_logging, 'w+')

    cp = ControlPlane()
    cp.actual_mode = "Hidden Markov Model"
    cp.csv_path = CSV_PATH
    cp.rcpower_dset = RCPOWER_DSET
    cp.log_file_name = LOG_FILE_NAME
    cp.rcpower_dset_auto = RCPOWER_DSET_AUTO
    cp.dt_dset = DT_DSET
    cp.discret = DISCRET
    cp.path_repository = str(PATH_REPOSITORY)
    cp.folder_control_log = folder_for_control_logging
    cp.folder_predict_log = folder_for_predict_logging
    cp.fc = fc
    cp.fp = fp

    ControlPlane.set_modeImbalance(MODE_IMBALANCE)
    ControlPlane.set_modeImbalanceNames((IMBALANCE_NAME, PROGRAMMED_NAME, DEMAND_NAME))
    ControlPlane.set_ts_duration_days(TS_DURATION_DAYS)
    ControlPlane.set_psd_segment_size(SEGMENT_SIZE)

    title1 = "Short Term Green Energy Load Predictor{} ".format(cp.actual_mode)
    title2 = "started at"
    msg = "{}\n".format(date_time)
    msg2log("{} (Control Plane) {} ".format(title1, title2), msg, fc)

    msg2log("{} (Predict Plane) {} ".format(title1, title2), msg, fp)

    if workmode == LEARN_HMM:
        trainPath(cp, mypath, LISTCSV)
    elif workmode == PREDICT_HMM:
        predictPath(cp, mypath, LISTCSV[1])
    else:
        pass

    title1 = "Short Term Green Energy Load Predictor "
    title2 = " finished at "
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    msg = "{}\n".format(date_time)
    msg2log("{} Control Plane {}".format(title1, title2), msg, fc)

    msg2log("{} Predict Plane {}".format(title1, title2), msg, fp)

    fc.close()
    fp.close()
    with open("execution_time.log", 'a') as fel:
        fel.write("Time execution logging finished at {}\n\n".format(datetime.now().strftime(cSFMT)))

    return 0


def logFwdBkwDict(fwd: list, bkw: list, observations: tuple, f: object = None) -> None:
    if f is None: return
    if type(observations) == list:
        observations = tuple(observations)
    elif type(observations) == tuple:
        pass
    else:
        print("\n\nargument 'observations' must be list or tuple\n\n")
        return
    f.write('\n\nForward algorithm list\nFrom first timestamp to last timestamp\n\n')
    for i, observation_i in enumerate(observations):
        item = fwd[i]

        s = "{} => (".format(observation_i)
        for key, val in item.items():
            s = s + "{} : {},".format(key, val)
        s = s + ')\n'
        f.write(s)

    f.write('\n\nBackward algorithm list\n From  timestamp after last to first timestamp\n\n')
    for i, observation_i_plus in enumerate(reversed(observations[1:] + (None,))):
        item = bkw[i]
        s = "\n{} => ".format(observation_i_plus)
        for key, val in item.items():
            s = s + " {} : {},".format(key, val)
        s = s + ')\n'
        f.write(s)

    return


def logViterbi(V, f=None):
    if f is None: return
    f.write("Viterbi algorithm")
    s = " ".join(("%12d" % i) for i in range(len(V)))
    f.write("\n{}\n".format(s))
    for state in V[0]:
        s = "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)
        f.write("\State is {}:  {}\n".format(state, s))

    return


def patternCountDict() -> dict:
    pass
    d_pattern_count = {}
    d_pattern_count["--"] = 0
    d_pattern_count["-0"] = 0
    d_pattern_count["-+"] = 0
    d_pattern_count["0-"] = 0
    d_pattern_count["00"] = 0
    d_pattern_count["0+"] = 0
    d_pattern_count["+-"] = 0
    d_pattern_count["+0"] = 0
    d_pattern_count["++"] = 0
    d_pattern_count["Start-"] = 0
    d_pattern_count["Start0"] = 0
    d_pattern_count["Start+"] = 0

    return copy.copy(d_pattern_count)


def logTransitions(title: str, d_total_count: dict, f: object) -> None:
    """

    :param title:
    :param d_total_count:
    :param f:
    :return:
    """
    totalMinus = d_total_count["--"] + d_total_count["-0"] + d_total_count["-+"]
    totalZero = d_total_count["0-"] + d_total_count["00"] + d_total_count["0+"]
    totalPlus = d_total_count["+-"] + d_total_count["+0"] + d_total_count["++"]

    message = f"""   Dataset Estimation   :{title} 
                     Transitions 
              '-'->'-': {d_total_count['--']}
              '-'->'0': {d_total_count['-0']}
              '-'->'+': {d_total_count['-+']}
              '0'->'-': {d_total_count['0-']}
              '0'->'0': {d_total_count['00']}
              '0'->'+': {d_total_count['0+']}
              '+'->'-': {d_total_count['+-']}
              '+'->'0': {d_total_count['+0']}
              '+'->'+': {d_total_count['++']}

                      Probabilites
              '-'->'-': {d_total_count['--'] / totalMinus}
              '-'->'0': {d_total_count['-0'] / totalMinus}
              '-'->'+': {d_total_count['-+'] / totalMinus}
              '0'->'-': {d_total_count['0-'] / totalZero}
              '0'->'0': {d_total_count['00'] / totalZero}
              '0'->'+': {d_total_count['0+'] / totalZero}
              '+'->'-': {d_total_count['+-'] / totalPlus}
              '+'->'0': {d_total_count['+0'] / totalPlus}
              '+'->'+': {d_total_count['++'] / totalPlus}

        """
    msg2log(main.__name__, message, f)

    return


def readDataset(csv_path: Path, dt_dset: str, programmed_dset: str, rcpower_dset: str, imbalance_dset: str,
                normalized_imbalance: str, f: object) -> (pd.DataFrame, dict, object):
    """

    :param csv_path:
    :param dt_dset:
    :param programmed_dset:
    :param rcpower_dset:
    :param imbalance_dset:
    :param normalized_imbalance:
    :param f:
    :return:
    """

    df = pd.read_csv(csv_path)
    df.head()

    buff = df.to_csv()
    msg2log(readDataset.__name__, "\nDataset from {}".format(csv_path), f)
    msg2log(readDataset.__name__, buff, f)

    a = df.values
    (n, m) = a.shape
    cols = df.columns.values.tolist()
    del df

    cols.append(imbalance_dset)
    ndf = pd.DataFrame(columns=[cols[0], cols[1], cols[2], cols[3], cols[4]])
    # print("Empty Dataframe ", ndf, sep='\n')
    ts_size = 0
    for i in range(n):
        if isnan(a[i][1]):
            break

        ndf = ndf.append({cols[0]: a[i][0], cols[1]: a[i][1], cols[2]: a[i][2], cols[3]: a[i][3],
                          cols[4]: (float(a[i][3]) - float(a[i][1]))},
                         ignore_index=True)
        ts_size += 1

    last_time = ndf[cols[0]].max()
    first_time = ndf[cols[0]].min()

    max_val = ndf[cols[4]].max()
    min_val = ndf[cols[4]].min()
    ndf[dt_dset] = pd.to_datetime(ndf[dt_dset], dayfirst=True)
    mean_val = ndf[imbalance_dset].mean()
    std_val = ndf[imbalance_dset].std()
    first_value = ndf[imbalance_dset].values[0]

    ndf.info()
    ndf.head(10)

    # buff = ndf.to_csv()
    # msg2log(readDataset.__name__, "\nDataset. [Imbalance]=[Programmed_demand]-[Real_demand] parameter added.\n", f)
    # msg2log(readDataset.__name__, buff, f)

    farray = copy.copy(ndf[imbalance_dset].values)

    farray4norm = copy.copy(ndf[imbalance_dset].values)
    for i in range(len(farray4norm)):
        farray4norm[i] = (farray4norm[i] - mean_val) / std_val

    ndf[normalized_imbalance] = farray4norm.tolist()
    min_norm_val = ndf[normalized_imbalance].min()
    max_norm_val = ndf[normalized_imbalance].max()
    buff = ndf.to_csv()

    message = f"""       Dataset
[Imbalance]=[Programmed_demand]-[Real_demand] time series added.
[Normalized_Imalance]                         time series added.
   - mean :               {mean_val}
   - standart derivation : {std_val}
    :{buff}           
    """
    msg2log(readDataset.__name__, message, f)
    # msg2log(readDataset.__name__, "\nNormalized  Dataset mean = {} std = {}".format(mean_val, std_val), f)
    # msg2log(readDataset.__name__, buff, f)

    message = f"""  Dataset       : {csv_path}
            [ProgrammedMinusReal] properties
            First time            : {first_time}
            Last time             : {last_time}
            First value           : {first_value}
            Min. value            : {min_val}
            Max. value            : {max_val}
            Mean value            : {mean_val}
            Std value             : {std_val}
            Min (normalized) value: {min_norm_val}
            Max (normalized) value: {max_norm_val}
    """
    msg2log(readDataset.__name__, message, f)
    ret_dict = {}

    ret_dict['first_value'] = first_value
    ret_dict['first_time'] = first_time
    ret_dict['last_time'] = last_time
    ret_dict['min_val'] = min_val
    ret_dict['max_val'] = max_val
    ret_dict['mean_val'] = mean_val
    ret_dict['std_val'] = std_val
    ret_dict['min_norm_val'] = min_norm_val
    ret_dict['max_norm_val'] = max_norm_val

    # return (ndf, first_time,last_time, min_val, max_val, mean_val, std_val, farray, min_norm_val, max_norm_val)
    return (ndf, ret_dict, farray)


def plotViterbiPath(pref, observations, viterbi_path, hidden_sequence, cp):
    """

    :param df:
    :param cp:
    :return:
    """

    suffics = ".png"
    if hidden_sequence is not None:
        file_name = "{}_viterbi_sequence_vs_hidden_sequence".format(pref)
    else:
        file_name = "{}_viterbi_sequence".format(pref)
    file_png = file_name + ".png"
    vit_png = Path(cp.folder_predict_log / file_png)
    # if sys.platform == "win32":
    #     file_png = file_name + ".png"
    #     vit_png = Path(cp.folder_predict_log / file_png)
    # elif sys.platform == "linux":
    #     vit_png = Path(cp.folder_predict_log / file_name).with_suffics(suffics)
    # else:
    #     vit_png = Path(cp.folder_predict_log / file_name).with_suffics(suffics)
    try:
        # plt.plot(observations, viterbi_path, label="Viterbi path")
        # plt.plot(observations, hidden_sequence, label="Hidden path")
        plt.plot(viterbi_path, label="Viterbi path")

        if hidden_sequence is not None:
            plt.plot(hidden_sequence, label="Hidden path")

        else:
            pass
        numfig = plt.gcf().number
        fig = plt.figure(num=numfig)
        fig.set_size_inches(18.5, 10.5)
        if hidden_sequence is not None:
            fig.suptitle("Viterby optimal path vs hidden path for dataset\n{}".format(cp.csv_path), fontsize=24)
        else:
            fig.suptitle("Viterby optimal path for dataset\n{}".format(cp.csv_path), fontsize=24)
        plt.ylabel("States", fontsize=18)
        plt.xlabel("Observation timestamps", fontsize=18)
        plt.legend()
        plt.savefig(vit_png)

    except:
        pass
    finally:
        plt.close("all")
    return


def plotDF(df, cp):
    """

    :param df:
    :param cp:
    :return:
    """

    suffics = ".png"
    file_name = str(cp.csv_path.name)
    file_png = file_name + ".png"
    df_png = Path(cp.folder_control_log / file_png)
    # if sys.platform == "win32":
    #     file_png = file_name + ".png"
    #     df_png = Path(cp.folder_control_log / file_png)
    # elif sys.platform == "linux":
    #     df_png = Path(cp.folder_control_log / file_name+suffics)
    # else:
    #     df_png = Path(cp.folder_control_log / file_name+suffics)
    try:
        df.plot(x=df.columns.values[0], y=df.columns.values[1:], kind='line')
        numfig = plt.gcf().number
        fig = plt.figure(num=numfig)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle("DF with normalized Imbalance {}".format(cp.csv_path), fontsize=24)

        plt.savefig(df_png)
        message = f"""
        Data Frame               : {cp.csv_path}
        Plot file                : {df_png}
        Time series              : {df.columns}

        """
        msg2log(plotDF.__name__, message, cp.fc)
    except:
        pass
    finally:
        plt.close("all")


def trainHMMprob(concat_list: list, states_concat_list: list, cp: ControlPlane) -> (np.array, np.matrix, np.matrix):
    """

    :param concat_list:
    :param states_concat_list:
    :param cp:
    :return:
    """

    trainArray = np.concatenate(concat_list)
    ar_state = np.concatenate(states_concat_list)
    mean = np.mean(trainArray)
    std = np.std(trainArray)

    states, counts = np.unique(ar_state, return_counts=True)

    emisDist = emissionDistribution(trainArray, ar_state, states, counts, cp.fc)

    transDist = transitionsDist(ar_state, states, cp.fc)

    pai = np.zeros((states.shape), dtype=float)
    for i in range(len(states)):
        pai[i] = counts[i] / len(trainArray)

    logDist(pai, "Initial Distribution", cp.fc)

    plotArray(trainArray, "Train Time Series for hmm", "train_HMM", cp)
    del trainArray
    plotArray(ar_state[:64], "Train Hidden States for hmm (first 64 items)", "hiddenStates_HMM_first", cp)
    if (len(ar_state) - 64) >= 0:
        plotArray(ar_state[len(ar_state) - 64:], "Train Hidden States for hmm (last items)", "hiddenStates_HMM_last",
                  cp)
    del ar_state

    return (pai, transDist, emisDist)


def plotArray(arr, title, file_name, cp):
    suffics = ".png"


    file_png = file_name + ".png"
    fl_png = Path(cp.folder_control_log / file_png)

    try:
        plt.plot(arr)
        numfig = plt.gcf().number
        fig = plt.figure(num=numfig)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle("{}".format(title), fontsize=24)

        plt.savefig(fl_png)
        message = f"""
          
           Plot file                : {fl_png}
           Time series              : {title}

           """
        msg2log(plotArray.__name__, message, cp.fc)
    except:
        pass
    finally:
        plt.close("all")
    return


# initialDist obsolited
def initialDist(concat_list: list, states: np.array, valBound: float, f: object = None) -> np.array:
    pass
    n_samples = len(concat_list)
    shp = len(states)
    pai = np.zeros((shp), dtype=float)
    aux = np.zeros((n_samples))
    for i in range(n_samples):
        val = concat_list[i][0]
        if val < (-valBound):
            aux[i] = - 1
        elif val >= (valBound):
            aux[i] = 1
        else:
            pass
    _, counts = np.unique(aux, return_counts=True)
    for i in range(shp):
        pai[i] = counts[i] / float(n_samples)
    logDist(pai, "Initial Distribution", f)
    return pai


def emissionDistribution(trainArray: np.array, ar_state: np.array, states: np.array, counts: np.array,
                         f: object) -> np.array:
    pass
    emisDist = np.zeros((len(states), 2), dtype=float)
    (n_size,) = ar_state.shape
    for i in range(len(states)):
        a = np.zeros((counts[i],), dtype=float)
        j = 0
        for k in range(n_size):
            if ar_state[k] == states[i]:
                a[j] = trainArray[k]
                k += 1
                j += 1
        (nn,)=a.shape
        if nn == 1:
            emisDist[i][0] = a[0]
            emisDist[i][1] = 1e-04
        else:
            emisDist[i][0] = a.mean()
            emisDist[i][1] = max(1e-02, a.std())

        del a
    print(emisDist)
    logDist(emisDist, "Emission Distribution", f)
    return emisDist


def logDist(arDist: np.array, title: str, f: object = None) -> None:
    if f is None:
        return
    f.write("\n{}\n".format(title))
    shp = arDist.shape
    if len(shp) == 1:

        b = arDist.reshape((-1, 1))
        (n_row, n_col) = b.shape
        for i in range(n_row):
            s = ""
            for j in range(n_col):
                s = s + "  " + "{}".format(b[i][j])
            s = s + '\n'
            f.write(s)
        return
    elif len(shp) == 2:
        (n_row, n_col) = shp
    else:
        return

    for i in range(n_row):
        s = ""
        for j in range(n_col):
            s = s + "  " + "{}".format(arDist[i][j])
        s = s + '\n'
        f.write(s)
    return

def imprLogDist(arDist: np.array, row_header:list, col_header:list, title: str, f: object = None) -> None:
    if f is None:
        return



    f.write("\n{}\n".format(title))
    shp = arDist.shape
    if len(shp) == 1:

        b = arDist.reshape((-1, 1))
        (n_row, n_col) = b.shape
        auxLogDist(b, n_row, n_col, row_header, col_header, title, f)

    elif len(shp) == 2:
        (n_row, n_col) = shp
        auxLogDist(arDist, n_row, n_col, row_header, col_header, title, f)
    else:
        msg2log(None,"Incorrect array shape {}".format(shp),f)

    return

def auxLogDist(arDist: np.array, n_row:int, n_col:int,  row_header:list, col_header:list, title: str, f: object = None):
    if row_header is None or not row_header:
        row_header = [str(i) for i in range(n_row)]
    if col_header is None or not col_header:
        col_header = [str(i) for i in range(n_col)]
    row_header = [str(i) for i in row_header]
    col_header = [str(i) for i in col_header]
    wspace = ' '
    s = "{:<10s}".format(wspace)

    for i in col_header:
        s = s + "{:^11s}".format(i)
    f.write("{}\n".format(s))
    for i in range(n_row):
        s = "{:<10s}".format(row_header[i])
        for j in range(n_col):
            if isinstance(arDist[i][j], int):
                s1 = "{:>10d}".format(arDist[i][j])
            elif isinstance(arDist[i][j], float):
                s1 = "{:>10.4f}".format(arDist[i][j])
            else:
                s1 = "{:^10s}".format(arDist[i][j])
            s = s + "  " + s1
        f.write("{}\n".format(s))


def transitionsDist(a_sign: np.array, states: np.array, f: object = None) -> np.matrix:
    # states,counts=np.unique(a_sign,return_counts=True)
    _, denominators = np.unique(a_sign[:-1], return_counts=True)
    n_size = len(states)
    shp = (n_size, n_size)
    tranDist = np.zeros(shp, dtype=float)
    for i in range(len(states)):
        denominator = denominators[i]
        for j in range(len(states)):
            numerator = 0
            for k in range(len(a_sign) - 1):
                if a_sign[k] == states[i] and a_sign[k + 1] == states[j]:
                    numerator += 1
            tranDist[i][j] = float(numerator) / float(denominator)
    logDist(tranDist, "Transitions Distribution", f)
    return tranDist


def parseDataset(cp: ControlPlane, csv_ind: int) -> (np.array, dict, dict, dict, np.array, np.array, np.array):
    """

    :param cp:
    :param csv_ind:
    :return: (states,d_transitions, d_count, property_dict, farray,observation_labels)
    """

    normalized_imbalance = "Normalized_Imbalance"
    signed_imbalance = "Signed_Imbalance"

    # read dataset
    (imbalance_dset, programmed_dset, demand_dset) = ControlPlane.get_modeImbalanceNames()

    # (df, first_time, last_time, min_val, max_val, mean, std,farray, min_norm_val, max_norm_val) = readDataset(
    #     cp.csv_path, cp.dt_dset, programmed_dset,cp.rcpower_dset, imbalance_dset, normalized_imbalance, cp.fc)

    (df, property_dict, farray) = readDataset(
        cp.csv_path, cp.dt_dset, programmed_dset, cp.rcpower_dset, imbalance_dset, normalized_imbalance, cp.fc)

    plotDF(df, cp)
    (df, states, d_transitions, d_count, a_sign) = calcTransitions(cp, df, farray, property_dict['std_val'],
                                                                   signed_imbalance, cp.fc)

    file_df = "expanded" + str(Path(cp.csv_path).name)
    updated_df = Path(cp.folder_control_log / file_df)
    df.to_csv(updated_df)

    pass
    observation_labels = [df.values[i][0].strftime('%Y-%m-%d %H:%M') for i in range(df.values.shape[0])]
    observation_labels = observation_labels[:len(farray)]
    # return (dict_processed_csv,d_count)
    return (states, d_transitions, d_count, property_dict, farray, observation_labels, a_sign)


""" Creates the sign array they items belong {-1,0,+1} by rule

    -1 , if x[i]<=-0.1*std
     0,  if -0.1*std<=x[i]<=0.1*std
    +1,  if x[i]>0.1*std

    Create the transmissions array which start with 0 and if x[

"""


def calcTransitiosMatrix(a_sign: np.array) -> (np.array, np.matrix):
    states, counts = np.unique(a_sign, return_counts=True)
    _, denominators = np.unique(a_sign[:-1], return_counts=True)
    n_size = len(states)
    shp = (n_size, n_size)
    p_t = np.zeros(shp, dtype=float)
    for i in range(len(states)):
        denominator = denominators[i]
        for j in range(len(states)):
            numerator = 0
            for k in range(len(a_sign) - 1):
                if a_sign[k] == states[i] and a_sign[k + 1] == states[j]:
                    numerator += 1
            p_t[i][j] = float(numerator) / float(denominator)
    return (states, p_t)


def boundBalance(std: float) -> float:
    return 0.1 * std


#
def calcTransitions(cp: ControlPlane, df: pd.DataFrame, x: np.array, std: float, signed_imbalance: str, f: object) -> (
pd.DataFrame, np.array, dict, dict, np.array):
    """

    :param cp:
    :param df:
    :param x:
    :param std:
    :param signed_imbalance:
    :param f:
    :return: (df, states, d_transitions, d_count),
    where f - pd.DataFrame
        states -vector of states, each state is string variable
        d_transitions -dictionary,
        d_count  - dictionary
    """
    valBB = boundBalance(std)
    a_sign = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        if x[i] < -valBB:
            a_sign[i] = -1
        elif x[i] >= valBB:
            a_sign[i] = 1
        else:
            pass
    (states, p_t) = calcTransitiosMatrix(a_sign)

    df[signed_imbalance] = a_sign
    d_transitions = {}
    d_count = patternCountDict()

    d_count['Std-'] = 0.0
    d_count['Mean-'] = 0.0
    d_count['Len-'] = 0
    d_count['Hist-'] = []
    d_count['Std0'] = 0.0
    d_count['Mean0'] = 0.0
    d_count['Len0'] = 0
    d_count['Hist0'] = []
    d_count['Std+'] = 0.0
    d_count['Mean+'] = 0.0
    d_count['Len+'] = 0
    d_count['Hist+'] = []

    for i in range(len(x) - 1):
        if x[i] < -valBB and x[i + 1] < -valBB:
            d_count["--"] += 1
            d_transitions[i] = '(-) -> (-)'
        elif x[i] < -valBB and x[i + 1] >= -valBB and x[i + 1] <= valBB:
            d_count["-0"] += 1
            d_transitions[i] = '(-) -> (0))'
        elif x[i] < -valBB and x[i + 1] > valBB:
            d_count["-+"] += 1
            d_transitions[i] = '(-) -> (+))'
        elif x[i] >= -valBB and x[i] <= valBB and x[i + 1] < -valBB:
            d_count["0-"] += 1
            d_transitions[i] = '(0) -> (-)'
        elif x[i] >= -valBB and x[i] <= valBB and x[i + 1] >= -valBB and x[i + 1] <= valBB:
            d_count["00"] += 1
            d_transitions[i] = '(0) -> (0)'
        elif x[i] >= -valBB and x[i] <= valBB and x[i + 1] > valBB:
            d_count["0+"] += 1
            d_transitions[i] = '(0) -> (+)'
        elif x[i] > valBB and x[i + 1] < -valBB:
            d_count["+-"] += 1
            d_transitions[i] = '(+) -> (-)'
        elif x[i] > valBB and x[i + 1] >= -valBB and x[i + 1] <= valBB:
            d_count["+0"] += 1
            d_transitions[i] = '(+) -> (0)'
        elif x[i] > valBB and x[i + 1] > valBB:
            d_count["++"] += 1
            d_transitions[i] = '(+) -> (+)'

    if x[0] > valBB:
        d_count["Start+"] = 1
    elif x[0] >= - valBB and x[0] <= valBB:
        d_count["Start0"] = 1
    else:
        d_count["Start-"] = 1

    bins = 10
    d_count['Std-'] = np.std([x[i] for i in range(len(x)) if x[i] < -valBB])
    d_count['Mean-'] = np.mean([x[i] for i in range(len(x)) if x[i] < -valBB])
    d_count['Len-'] = len([x[i] for i in range(len(x)) if x[i] < -valBB])
    d_count['Hist-'] = np.histogram([x[i] for i in range(len(x)) if x[i] < -valBB], bins)

    d_count['Std0'] = np.std([x[i] for i in range(len(x)) if x[i] >= -valBB and x[i] <= valBB])
    d_count['Mean0'] = np.mean([x[i] for i in range(len(x)) if x[i] >= -valBB and x[i] <= valBB])
    d_count['Len0'] = len([x[i] for i in range(len(x)) if x[i] >= -valBB and x[i] <= valBB])
    d_count['Hist0'] = np.histogram([x[i] for i in range(len(x)) if x[i] >= -valBB and x[i] <= valBB], bins)

    d_count['Std+'] = np.std([x[i] for i in range(len(x)) if x[i] > valBB])
    d_count['Mean+'] = np.mean([x[i] for i in range(len(x)) if x[i] > valBB])
    d_count['Len+'] = len([x[i] for i in range(len(x)) if x[i] > valBB])
    d_count['Hist+'] = np.histogram([x[i] for i in range(len(x)) if x[i] > valBB], bins)

    file_name = Path(cp.csv_path).name
    file_name_plus = file_name.replace('.', '_') + ".PlusHist.png"
    file_name_minus = file_name.replace('.', '_') + ".MinusHist.png"
    file_name_zero = file_name.replace('.', '_') + ".ZeroHist.png"
    file_plus_png = Path(cp.folder_control_log / file_name_plus)
    file_minus_png = Path(cp.folder_control_log / file_name_minus)
    file_zero_png = Path(cp.folder_control_log / file_name_zero)

    logHistogram("Histogram (+) for {}".format(file_name), file_plus_png,
                 [x[i] for i in range(len(x)) if x[i] > valBB], bins)
    logHistogram("Histogram (0) for {}".format(file_name), file_zero_png,
                 [x[i] for i in range(len(x)) if x[i] >= -valBB and x[i] <= valBB], bins)
    logHistogram("Histogram (-) for {}".format(file_name), file_minus_png,
                 [x[i] for i in range(len(x)) if x[i] < -valBB], bins)

    buff = df.to_csv()
    msg2log(calcTransitions.__name__, "\nExpanded   Dataset ", f)
    msg2log(calcTransitions.__name__, buff, f)
    for i in range(len(states)):
        states[i] = str(states[i])
    return (df, states, d_transitions, d_count, a_sign)


def logHistogram(title, file_png, arr, bins):
    plt.hist(arr, bins)
    plt.title(title)

    plt.savefig(file_png)
    # plt.show()
    plt.close("all")
    return


def drive_HMM(cp: ControlPlane, pai: np.array, transitDist: np.matrix, emisDist: np.matrix, observations: np.array,
              observation_labels: np.array, states_set: np.array)->(np.array, str, object):
    """

    :rtype: object
    :param cp:
    :param pai:
    :param transitDist:
    :param emisDist:
    :param observations:
    :param observation_labels:
    :param states_set:
    :return:
    """

    tfd = tpb.distributions

    logDist(pai, "Initial Distribution", cp.fp)
    logDist(transitDist, "Transition Distribution", cp.fp)
    logDist(emisDist, "Emission Distribution", cp.fp)

    pai = tf.convert_to_tensor(pai, dtype=tf.float64)
    transitDist = tf.convert_to_tensor(transitDist, dtype=tf.float64)

    initial_distribution = tfd.Categorical(probs=pai)

    # transition_distributions = tfd.Categorical(probs=[[0.7,0.3],
    #                                                  [0.2,0.8]])

    transition_distribution = tfd.Categorical(probs=transitDist)

    mean_list = emisDist[:, 0].tolist()
    std_list = emisDist[:, 1].tolist()
    # mean_list = [emisDist[0][0],emisDist[1][0],emisDist[2][0]]
    # std_list = [emisDist[0][1], emisDist[1][1], emisDist[2][1]]
    for i in range(len(std_list)):
        if std_list[i] < 1e-06:
            std_list[i] = 1e-06

    mean_list = tf.convert_to_tensor(mean_list, dtype=tf.float64)
    std_list = tf.convert_to_tensor(std_list, dtype=tf.float64)

    # observation_distribution = tfd.Normal(loc=[0.5,15], scale=[5.0,10])
    observation_distribution = tfd.Normal(loc=mean_list, scale=std_list)

    model = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=len(observations))

    observations_tenzor = tf.convert_to_tensor(observations.tolist(), dtype=tf.float64)

    post_mode = model.posterior_mode(observations_tenzor)
    msg2log(drive_HMM.__name__, "Posterior mode\n\n{}\n".format(post_mode), cp.fp)
    post_marg = model.posterior_marginals(observations_tenzor)
    msg2log(drive_HMM.__name__, "{}\n\n{}\n".format(post_marg.name, post_marg.logits), cp.fp)
    mean_value = model.mean()
    msg2log(drive_HMM.__name__, "mean \n\n{}\n".format(mean_value), cp.fp)
    log_probability = model.log_prob(observations_tenzor)
    msg2log(drive_HMM.__name__, "Log probability \n\n{}\n".format(log_probability), cp.fp)

    plotViterbiPath(str(len(observations)), observation_labels, post_mode.numpy(), states_set, cp)
        
    return post_mode.numpy(),post_marg.name, post_marg.logits      






if __name__ == "__main__":
    main()
