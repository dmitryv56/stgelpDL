#! /usr/bin/python3

import copy
from predictor.api import d_models_assembly, fit_models, save_modeles_in_repository,  get_list_trained_models
from predictor.api import predict_model, chart_predict, tbl_predict
from predictor.utility import exec_time

@exec_time
def drive_auto(cp, ds):
    pass

"""
1.check list of modeles
2.create models from template
3.update model parameters (n_steps)
4.compile models
5.train modeles 
6. save modeles 
"""
@exec_time
def drive_train(cp, ds):
    """
    :param cp:  ControlPlane object
    :param ds:  dataset object
    :return:
    """

    d_models = {}

    for keyType, valueList in cp.all_models.items():
        print('{}->{}'.format(keyType, valueList))  # MLP->[(0,'mlp_1'),(1,'mlp_2)], CNN->[(2, 'univar_cnn')]
        # LSTM->[(3, 'vanilla_lstm'), (4, 'stacked_lstm'), (5, 'bidir_lstm'), (6, 'cnn_lstm')]

        status = d_models_assembly(d_models, keyType, valueList, cp, ds )


    print(d_models)

    if cp.fc is not None:
        cp.fc.write("\n   Actual Neuron Net and Statistical Time Series Models\n")
        for k, v in d_models.items():
            cp.fc.write("{} - > {}\n".format(k, v))


    #  fit
    histories = fit_models(d_models, cp, ds)

    # save modeles
    save_modeles_in_repository(d_models, cp)

    return


"""
1. check if model exists
2. check if history for forecat exists (dataset)
3. load models
4. predict
5. predict analysis

"""
@exec_time
def drive_predict(cp, ds):
    """

    :param cp: ControlPlane object
    :param ds: dataset object
    :return:
    """

    ds.data_for_predict = cp.n_steps
    ds.predict_date = None
    predict_history = copy.copy(ds.data_for_predict)

    dict_model = get_list_trained_models(cp)
    n_predict=4
    dict_predict = predict_model(dict_model, cp, ds, n_predict)


    chart_predict(dict_predict, n_predict, cp, ds, "{} Predict".format(cp.rcpower_dset), cp.rcpower_dset)

    tbl_predict(dict_predict, n_predict, cp, ds, "{} Predict".format(cp.rcpower_dset))

    return

def drive_control(cp, ds):
    pass