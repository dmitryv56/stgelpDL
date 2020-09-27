#!/usr/bin/python3
""" This predictor module contains an implementation of Predictor class that is the base class for NNmodel and
Statmodel classes.

"""

import os
import sys
from pathlib import Path

from tensorflow.keras.models import Sequential, save_model, load_model
from pickle import dump, load


class Predictor():
    _count = -1
    _features = 1
    _steps = 0
    _epochs = 0
    _path2modelRepository = 0
    _timeseries_name = None


    def __init__(self, nameModel, typeModel, steps, epochs, f=None):
        self.__class__._count += 1
        self.id = self.__class__._count
        self.nameModel = nameModel
        self.typeModel = typeModel
        self.set_steps = steps
        self.set_epochs = epochs
        self.f = f
        self.scaler = None
        pass

    def __str__(self):
        return 'id :' + str(self.get_id()) + ' model: ' + self.nameModel + "  type: " + self.typeModel + \
               "\nmodel repository: " + self.path2modelrepository + ": time series :" + self.timeseries_name

    # ============ setter/getter ==============================
    def get_id(self):
        return self.id

    def get_features(self):
        return type(self)._features

    def set_features(self, val):
        type(self)._features = val

    n_features = property(get_features, set_features)

    def get_steps(self):
        return type(self)._steps

    def set_steps(self, val):
        type(self)._steps = val

    n_steps = property(get_steps, set_steps)

    def get_epochs(self):
        return type(self)._epochs

    def set_epochs(self, val):
        type(self)._epochs = val

    n_epochs = property(get_epochs, set_epochs)

    """ 
     repository methods 
    """

    def get_modelRepository(self):
        return type(self)._path2modelRepository

    def set_modelRepository(self, path2repos):
        type(self)._path2modelRepository = path2repos

    path2modelrepository = property(get_modelRepository, set_modelRepository)

    def get_timeseries_name(self):
        return type(self)._timeseries_name

    def set_timeseries_name(self,val):
        type(self)._timeseries_name = val

    timeseries_name = property(get_timeseries_name, set_timeseries_name)

    def set_model_repository(self):

        """
        Model get save/ load
        The trained model saved to 'model Repository'. The model subfolder name is formated  like as
        <model_type>_<model_name>_<dataset_name>
        """
        pass
    """
    The modes is saved in the folder <self.path2modelrepository>/<self.timeseries_name>/<self.typeModel>/<self.nameModel>
    """
    def save_model_wrapper(self):
        pass


        model_folder = Path(self.path2modelrepository) / self.timeseries_name / self.typeModel / self.nameModel
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        save_model(self.model, model_folder)
        scaler_saved_file = Path(model_folder) / "scaler.pkl"
        if self.scaler is not None:
            with open(scaler_saved_file, 'wb') as fp:
                dump(self.scaler, fp)

        if self.f is not None:
            self.f.write("Model saved in {} ".format(model_folder))
            currDir = Path(model_folder)
            if not sys.platform == "win32":
            # WindowsPath is not iterable
            #     for currFile in currDir:
            #         self.f.write("  {}\n".format(currFile))
                pass

        return
    """
    TODO
    """
    def load_model_wrapper(self, ):
        pass
        status = False

        model_folder = Path(self.path2modelrepository) / self.timeseries_name/self.typeModel/self.nameModel
        if not Path(model_folder).exists():
            if self.f is not None:
                self.f.write("No found saved {} model \n".format(model_folder))
            return status, None, None
        scaler_file = Path(model_folder) / "scaler.pkl"

        self.model = load_model(model_folder, compile=True)
        self.model.summary()
        if self.f is not None:
            self.model.summary(print_fn=lambda x: self.f.write(x + '\n'))

        # load MinMaxScaler that was saved with model
        with open(scaler_file, 'rb') as fp:
            scaler = load(fp)

        status = True
        return status, scaler
