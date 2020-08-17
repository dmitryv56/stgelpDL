#!/usr/bin/python3
import sys
import numpy as np
"""
print to log
"""

"""
tsBoundaries2log prints and logs time series boundaries characterisics (min,max,len)
"""


def tsBoundaries2log(title, df, dt_dset, rcpower_dset, f=None):
    """

    :param title: - title for print
    :param df:    - pandas DataFrame object
    :param dt_dset: - Data/Time column name
    :param rcpower_dset: - time series column name
    :param f:
    :return:
    """
    # For example,title ='Number of rows and columns after removing missing values'
    print('\n{}: {}'.format(title, df.shape))
    print('The time series length: {}\n'.format(len(df[dt_dset])))
    print('The time series starts from: {}\n'.format(df[dt_dset].min()))
    print('The time series ends on: {}\n\n'.format(df[dt_dset].max()))
    print('The minimal value of the time series: {}\n\n'.format(df[rcpower_dset].min()))
    print('The maximum value of the time series: {}\n\n'.format(df[rcpower_dset].max()))
    if f is not None:
        f.write('\n{}: {}\n'.format(title, df.shape))
        f.write('The time series length: {}\n'.format(len(df[dt_dset])))
        f.write('The time series starts from: {}\n'.format(df[dt_dset].min()))
        f.write('The time series ends on: {}\n'.format(df[dt_dset].max()))
        f.write('The minimal value of the time series: {}\n'.format(df[rcpower_dset].min()))
        f.write('The maximum value of the time series: {}\n\n'.format(df[rcpower_dset].max()))

    return


def tsSubset2log(dt_dset, rcpower_dset, df_train, df_val=None, df_test=None, f=None):
    pass
    print('Train dates: {} to {}'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
    if f is not None:
        f.write("\nTrain dataset\n")
        f.write('Train dates: {} to {}\n\n'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
        for i in range(len(df_train)):
            f.write('{} {}\n'.format(df_train[dt_dset][i], df_train[rcpower_dset][i]))

    if df_val is not None:

        print('Validation dates: {} to {}'.format(df_val[dt_dset].min(), df_val[dt_dset].max()))
        if f is not None:
            f.write("\nValidation dataset\n")
            f.write(
                'Validation  dates: {} to {}\n\n'.format(df_val[dt_dset].min(), df_val[dt_dset].max()))
            for i in range(len(df_train), len(df_train) + len(df_val)):
                f.write('{} {}\n'.format(df_val[dt_dset][i], df_val[rcpower_dset][i]))

    if df_test is not None:

        print('Test dates: {} to {}'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
        f.write("\nTest dataset\n")
        f.write('Test  dates: {} to {}\n\n'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
        start = len(df_train) if df_val is None else len(df_train) + len(df_val)
        stop = len(df_train) + len(df_test) if df_val is None else len(df_train) + len(df_val) + len(df_test)
        for i in range(start, stop):
            f.write('{} {}\n'.format(df_test[dt_dset][i], df_test[rcpower_dset][i]))
    return


def chunkarray2log(title, nparray, width=8, f=None):
    # scaled train data
    if f is not None:

        f.write("\n{}\n".format(title))

        for i in range(len(nparray)):
            if i % width == 0:
                f.write('\n{} : {}'.format(i, nparray[i]))
            else:
                f.write(' {} '.format(nparray[i]))
    return


"""
Supervised learning data to log
"""


def svld2log(X, y, print_weight, f=None):
    """

    :param X:
    :param y:
    :param print_weight:
    :param f:
    :return:
    """

    if (f is None):
        return

    for i in range(X.shape[0]):
        k = 0
        line = 0
        f.write("\nRow {}: ".format(i))
        for j in range(X.shape[1]):
            f.write(" {}".format(X[i][j]))
            k = k + 1
            k = k % print_weight
            if k == 0 or k == X.shape[1]:

                if line == 0:
                    f.write(" |  {} \n      ".format(y[i]))
                    line = 1
                else:
                    f.write("\n      ")
    return


def dataset_properties2log(csv_path, dt_dset, rcpower_dset, discret, test_cut_off, val_cut_off, n_steps, n_features, \
                           n_epochs, f=None):
    pass
    if f is not None:
        f.write(
            "====================================================================================================")
        f.write("\nDataset Properties\ncsv_path: {}\ndt_dset: {}\nrcpower_dset: {}\ndiscret: {}\n".format(csv_path,
                                                                                                          dt_dset,
                                                                                                          rcpower_dset,
                                                                                                          discret))
        f.write(
            "\n\nDataset Cut off Properties\ncut of for test sequence: {} minutes\ncut off for validation sequence: {} minutes\n".format(
                test_cut_off, val_cut_off))

        f.write("\n\nTraining Properties\n time steps: {},\nfeatures: {}\n,epochs: {}\n".format(n_steps, n_features,
                                                                                                n_epochs))
        f.write(
            "====================================================================================================\n\n")
    return


def msg2log(funcname, msg, f=None):
    print("\n{}: {}".format(funcname, msg))
    if f is not None:
        f.write("\n{}: {}\n".format(funcname, msg))

def vector_logging(title, seq, print_weigth, f=None):
    if f is None:
        return
    f.write("{}\n".format(title))
    k=0
    line = 0
    f.write("{}: ".format(line))
    for i in range(len(seq)):
        f.write(" {}".format(seq[i]))
        k = k + 1
        k = k % print_weigth
        if k == 0:
            line=line+1
            f.write("\n{}: ".format(line))

    return



# ##############################################charting################################################################

# preallocate empty array and assign slice (https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array)
def shift(arr, num, fill_value=np.nan):
    shift_arr = np.empty_like(arr)
    if num > 0:
        shift_arr[:num] = fill_value
        shift_arr[num:] = arr[:-num]
    elif num < 0:
        shift_arr[num:] = fill_value
        shift_arr[:num] = arr[-num:]
    else:
        shift_arr[:] = arr
    return shift_arr