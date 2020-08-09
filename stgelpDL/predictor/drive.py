#! /usr/bin/python3
import sys
from predictor.control import controlPlane
from predictor.dataset import dataset
from predictor.NNmodel import MLP, CNN, LSTM
import copy
import matplotlib.pyplot as plt
import numpy as np
from predictor.api import show_autocorr, chart_MAE, chart_MSE,  prepareDataset, d_models_assembly, fit_models
from predictor.api import save_modeles_in_repository, deserialize_lst_trained_models, get_list_trained_models
from predictor.api import predict_model, chart_predict
from predictor.utility import msg2log
from pathlib import Path



def drive_auto(cp, ds):
    pass


def drive_train(cp, ds):
    pass
    """
    1.check list of modeles
    2.create models from template
    3.update model parameters (n_steps)
    4.compile models
    5.train modeles 
    6. save modeles 
    """
    d_models = {}

    for keyType, valueList in cp.all_models.items():
        print('{}->{}'.format(keyType, valueList))  # MLP->[(0,'mlp_1'),(1,'mlp_2)], CNN->[(2, 'univar_cnn')]
        # LSTM->[(3, 'vanilla_lstm'), (4, 'stacked_lstm'), (5, 'bidir_lstm'), (6, 'cnn_lstm')]

        status = d_models_assembly(d_models, keyType, valueList, cp, ds )


    print(d_models)

    if cp.fc is not None:
        cp.fc.write("\n   Actual Neuron Net Models\n")
        for k, v in d_models.items():
            cp.fc.write("{} - > {}\n".format(k, v))


    #  fit
    histories = fit_models(d_models, cp, ds)

    # save modeles
    save_modeles_in_repository(d_models, cp)

    return



def drive_predict(cp, ds):
    #check if model exists
    #check if history for forecat exists (dataset)
    # load models
    # predict
    # predict analysis

    ds.data_for_predict = cp.n_steps
    ds.predict_date = None
    predict_history = copy.copy(ds.data_for_predict)

    dict_model = get_list_trained_models(cp)
    n_predict=4
    dict_predict = predict_model(dict_model, cp, ds, n_predict)


    chart_predict(dict_predict, n_predict, cp, ds, "{} forecasting".format(cp.rcpower_dset), cp.rcpower_dset)


    return




def drive_control(cp, ds):
    pass