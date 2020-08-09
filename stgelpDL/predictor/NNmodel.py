#!/usr/bin/python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional,TimeDistributed,Flatten

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import metrics, models
from tensorflow.keras.models import  save_model, load_model
from predictor.predictor import Predictor
from predictor.utility import msg2log
import copy


class NNmodel(Predictor):
    _param = ()
    _param_fit = ()
    model = None

    def __init__(self, nameM, typeM, n_steps, n_epochs, f=None):
        pass
        super().__init__(nameM, typeM, n_steps, n_epochs, f)

    # getter/setter
    def set_param(self, val):
        type(self)._param = copy.deepcopy(val)

    def get_param(self):
        return type(self)._param

    param = property(get_param, set_param)

    def set_param_fit(self, val):
        type(self)._param_fit = copy.copy(val)

    def get_param_fit(self):
        return type(self)._param_fit

    param_fit = property(get_param_fit, set_param_fit)

    # def set_model_from_template(self,func, *args):
    #     self.model = func(*args)
    def set_model_from_template(self, func):
        self.model = func()
        self.model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
        self.model.summary()

        if self.f is not None:
            self.model.summary(print_fn=lambda x: self.f.write(x + '\n'))

        return

    def set_model_from_saved(self, path_2_saved_model):

        old_model = models.load_model(path_2_saved_model)
        old_model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
        old_model.summary()
        if self.f is not None:
            old_model.summary(print_fn=lambda x: self.f.write(x + '\n'))
        self.model = old_model
        return old_model

    def updOneLeInSavedModel(self, old_model, layer_number, key, value):
        pass

        model_config = old_model.get_config()
        if self.f is not None:
            self.f.write('\n The model configuration\n')
            self.f.write(model_config)
        for lr in range(len(model_config['layers'])):
            print(model_config['layers'][lr])
            print(model_config['layers'][lr]['config'])
            if self.f is not None:
                self.f.write(model_config['layers'][lr])
                self.f.write(model_config['layers'][lr]['config'])

        # model_config['layers'][0]['config']['batch_input_shape'] = (None, 36, 1)
        model_config['layers'][layer_number]['config'][key] = value
        print(model_config['layers'][layer_number])
        print(model_config['layers'][layer_number]['config'])

        if self.f is not None:
            self.f.write('\nUpdated layer\n')
            self.f.write(model_config['layers'][layer_number])
            self.f.write(model_config['layers'][layer_number]['config'])

        self.model = models.Sequential.from_config(model_config)
        self.model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
        self.model.summary()

        if self.f is not None:
            self.write('\n New model \n')
            self.model.summary(print_fn=lambda x: self.f.write(x + '\n'))

        return

    """
    updates key:value in the layer 'config' dictionary
    """

    def updConfigSavedModel(self, old_model,
                            list_updates):  # [(layer_number, key, value),(layer_number, key, value)...]
        """

        :param old_model: tensorflow.keras Sequential model that was loaded from saved model
        :param list_updates: list contains a tuples (layer_number, key, new_value)
        :return:
        """

        model_config = old_model.get_config()
        if self.f is not None:
            self.f.write('\n The model configuration\n')
            self.f.write(model_config)
        for lr in range(len(model_config['layers'])):
            print(model_config['layers'][lr])
            print(model_config['layers'][lr]['config'])
            if self.f is not None:
                self.f.write(model_config['layers'][lr])
                self.f.write(model_config['layers'][lr]['config'])

        for tuple_item in list_updates:
            layer_number, key, value = tuple_item

            if layer_number >= len(model_config['layers']):
                msg2log(self.updConfigSavedModel.__name__, "Layer {} dont exist".format(layer_number), self.f)
                continue
            if key not in model_config['layers'][layer_number]['config']:
                msg2log(self.updConfigSavedModel.__name__,
                        "Key {} dont exist on {} layerr config".format(key, layer_number), self.f)
                continue

            # model_config['layers'][0]['config']['batch_input_shape'] = (None, 36, 1)
            model_config['layers'][layer_number]['config'][key] = value
            print(model_config['layers'][layer_number])
            print(model_config['layers'][layer_number]['config'])

            if self.f is not None:
                self.f.write('\nUpdated layer\n')
                self.f.write(model_config['layers'][layer_number])
                self.f.write(model_config['layers'][layer_number]['config'])

        self.model = models.Sequential.from_config(model_config)
        self.model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
        self.model.summary()

        if self.f is not None:
            self.write('\n New model \n')
            self.model.summary(print_fn=lambda x: self.f.write(x + '\n'))

        return

    def fit_model(self):
        X, y, X_val, y_val, n_steps, n_features, n_epochs, logfolder, f = self.param_fit
        history = self.model.fit(X, y, epochs=n_epochs, verbose=0, validation_data=(X_val, y_val), )
        print(history.history)
        if f is not None:
            f.write("\n\nTraining history for model {}\n{}".format(self.model.name, history.history))

        return history

    def predict_one_step(self, vec_data):
        pass
        print("{} {}".format(self.__class__.__name__, self.predict_one_step.__name__))
        print(self.predict_one_step.__name__)
        xx_ = vec_data.reshape((1, vec_data.shape[0], 1))
        y_pred = self.model.predict(xx_)
        return y_pred
########################################################################################################################
########################################################################################################################
class MLP(NNmodel):
    pass

    def __init__(self, nameM, typeM, n_steps, n_epochs, f=None):
        super().__init__(nameM, typeM, n_steps, n_epochs, f)

    def myprint(self):
        print("kuku MLP")

    # def mlp_1(self, param):  # n_steps, n_features = 1,hidden_neyron_number=100, dropout_factor=0.2
    def mlp_1(self):
        # define model
        n_steps, n_features, hidden_neyron_number, dropout_factor = self.param
        model = Sequential(name=self.mlp_1.__name__)
        # model.add(tf.keras.Input(shape=( n_steps,1)))
        model.add(Dense(hidden_neyron_number, activation='relu', input_dim=n_steps, name='Layer_0'))

        model.add(layers.Dropout(dropout_factor, name='Layer_1'))
        model.add(Dense(32, name='Layer_2'))
        model.add(layers.Dropout(dropout_factor, name='Layer_3'))
        model.add(Dense(1, name='Layer_4'))
        return model

    def mlp_2(self):  # n_steps, n_features = 1,hidden_neyron_number=100, dropout_factor=0.2
        # define model
        n_steps, n_features, hidden_neyron_number, dropout_factor = self.param
        model = Sequential(name=self.mlp_2.__name__)
        # model.add(tf.keras.Input(shape=( n_steps,1)))
        model.add(Dense(hidden_neyron_number, activation='relu', input_dim=n_steps, name='Layer_0'))

        model.add(layers.Dropout(dropout_factor, name='Layer_1'))
        model.add(Dense(32, name='Layer_2'))
        model.add(Dense(16, name='Layer_3'))

        model.add(Dense(1, name='Layer_4'))

        return model

    ####################################################################################################################
    ####################################################################################################################
    def predict_one_step(self, vec_data):
        pass
        print("{} {}".format(self.__class__.__name__, self.predict_one_step.__name__))
        xx_ = vec_data.reshape((1, vec_data.shape[0]))
        y_pred = self.model.predict(xx_)
        return y_pred

class LSTM(NNmodel):
    pass

    def __init__(self, nameM, typeM, n_steps, n_epochs, f=None):
        super().__init__(nameM, typeM, n_steps, n_epochs, f)

    def vanilla_lstm(self):  # (units, n_steps, n_features) ):
        units, n_steps, n_features = self.param
        model = None
        model = Sequential()  # name=self.vanilla_lstm.__name__)
        # model.add( LSTM( units,  activation='relu', input_shape=(n_steps, n_features), name='Layer_0'))
        model.add(tf.keras.layers.LSTM(units, activation='relu', input_shape=(n_steps, n_features), name='Layer_0'))
        model.add(Dense(1, name='Layer_1'))
        try:
            model_name = self.vanilla_lstm.__name__
        except:
            print("cant set model._name")

        return model

    def stacked_lstm(self):
        units, n_steps, n_features = self.param
        model = None
        model = Sequential()  # name=self.stacked_lstm.__name__)
        model.add(tf.keras.layers.LSTM(units, activation='relu', return_sequences=True, input_shape=(n_steps, n_features),
                                     name='Layer_0'))
        model.add(tf.keras.layers.LSTM(units, activation='relu', name='Layer_1'))
        model.add(Dense(1, name='Layer_2'))
        try:
            model_name = self.stacked_lstm.__name__
        except:
            print("cant set model._name")

        return model

    def bidir_lstm(self):
        units, n_steps, n_features = self.param
        model = None
        model = Sequential()  # name=self.bidir_lstm.__name__)
        model.add(Bidirectional(tf.keras.layers.LSTM(units, activation='relu'), input_shape=(n_steps, n_features),
                                    name='Layer_0'))
        model.add(Dense(1, name='Layer_1'))
        try:
            model_name = self.bidir_lstm.__name__
        except:
            print("cant set model._name")

        return model

    def cnn_lstm(self):
        units, n_steps, n_features = self.param
        model = None
        model = Sequential()  # name=self.cnn_lstm.__name__)
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                      input_shape=(None, n_steps / 2, n_features), name='Layer_0'))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2), name='Layer_1'))
        model.add(TimeDistributed(Flatten(), name='Layer_2'))
        model.add(tf.keras.layers.LSTM(units, activation='relu', name='Layer_3'))
        model.add(Dense(1, name='Layer_4'))
        try:
            model_name = self.cnn_lstm.__name__
        except:
            print("cant set model._name")

        return model


########################################################################################################################
########################################################################################################################


class CNN(NNmodel):
    pass

    def __init__(self, nameM, typeM, n_steps, n_epochs, f=None):
        super().__init__(nameM, typeM, n_steps, n_epochs, f)

    def univar_cnn(self):
        n_steps, n_features = self.param
        model = Sequential(name=self.univar_cnn.__name__)
        model.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features), name='Layer_0'))
        model.add(MaxPooling1D(pool_size=2, name='Layer_1'))
        model.add(Flatten(name='Layer_2'))
        model.add(Dense(50, activation='relu', name='Layer_3'))
        model.add(Dense(1, name='Layer_4'))

        return model

