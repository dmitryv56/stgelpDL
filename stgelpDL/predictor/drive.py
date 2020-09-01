#! /usr/bin/python3

import copy
from predictor.api import d_models_assembly, fit_models, save_modeles_in_repository,  get_list_trained_models
from predictor.api import predict_model, chart_predict, tbl_predict
from predictor.utility import exec_time
from predictor.demandwidget import DemandWidget
from abc import ABC, abstractmethod
from datetime import datetime, timedelta as td
from time import sleep
from predictor.utility import cSFMT
from predictor.control import ControlPlane
from predictor.dataset import Dataset
import os
from pathlib import Path
from predictor.api import prepareDataset

#
class ISubject(ABC):
    """
    interface to Subject
    """
    @abstractmethod
    def attach(self, observer)->None:
        """
        attaches the observers to the subject
        :param observer:
        :return:
        """
        pass

    @abstractmethod
    def detach(self,observer)->None:
        """
        detaches the observers from the subject
        :param observer:
        :return:
        """
        pass

    @abstractmethod
    def notify(self, dct : object)->None:
        """
        notifies all observers
        :type dct: object
        :param dct: for parameters passing
        :return:
        """
        pass

class UpdateProvider(ISubject):
    """
        _state = 0 -initial state, Control plane should be run
        _state = 1 - dataset created. The Train Plane should be run
        _state = 2 - deep learning finished. The Predict should be run
        _state = 3 - deep learning finished. The  re-train should be run
    """
    _state: int = None
    _observers = []
    _changed   = False

    def __init__(self, f):
        _observers = []
        self.f = f
        self.state = 0  # initial for control plane
        return

    # getter/setter

    def set_state(self,val):
        type(self)._state=val

    def get_state(self):
        return type(self)._state
    state = property(get_state, set_state)

    def set_changed(self,val):
        type(self)._changed=val

    def get_changed(self):
        return type(self)._changed
    changed = property(get_changed, set_changed)

    def attach(self,observer)->None:
        msg ='{} UpdateProvider: attached an observer {}'.format(datetime.now().strftime(cSFMT),observer.__str__())
        self._observers.append(observer)
        print(msg)
        if self.f is not None:
            self.f.write('{}\n'.format(msg))
        return

    def detach(self,observer)->None:
        """

        :type observer: object
        """
        msg = '{} UpdateProvider: detached from observer {}'.format(datetime.now().strftime(cSFMT), observer.__str__())

        self._observers.remove(observer)
        print(msg)
        if self.f is not None:
            self.f.write('{}\n'.format(msg))
        return

    def notify(self, dct: object)-> None:
        msg = '{} UpdateProvider: notifying observers..'.format(datetime.now().strftime(cSFMT))
        print(msg)
        if self.f is not None:
            self.f.write('\n{}\n'.format(msg))

        for observer in self._observers:
            observer.update(self, dct)
            msg = '{} UpdateProvider: notifying observers.. The observer {} has notification'.format(
                datetime.now().strftime(cSFMT), observer.__str__())

            print(msg)
            if self.f is not None:
                self.f.write('\n{}\n'.format(msg))

        return

class UpdateChecker(UpdateProvider):

    def __init__(self,f=None):
        super().__init__(f)

    def isChanged(self): return self.changed

    def setChanged(self): self.changed = 1

    def clearChanged(self): self.changed = 0

    def drive(self,cp: ControlPlane, ds: Dataset)->None:
        pass
        # self._state = randrange(0, 3)
        scaled_data = False
        start_time = "2020-08-30 00:00:00"
        end_time = "2020-09-03 00:00:00"
        dwdg = DemandWidget(scaled_data,start_time,end_time,'hour',None, None, self.f)
        dwdg.set_url()

        print(dwdg.url)

        requested_widget = dwdg.getDemandRT(None)
        print("Requested widget has type {}".format(type(requested_widget)))
        dwdg.plot_ts(os.getcwd(), False)
        dwdg.autocorr_show(os.getcwd(), False)
        dct= {'DataFrame': dwdg, 'ControlPlane': cp, 'Dataset': ds}
        msg = '{} {} :My state has just changed to {}'.format(datetime.now().strftime(cSFMT),
                                                              type(self).__name__, self._state)
        print(msg)
        if self.f is not None:
            self.f.write('\n{}\n'.format(msg))
        self.notify(dct)

        return

class IObserver(ABC):
    """
    observer's interface
    """
    @abstractmethod
    def update(self,subject: ISubject, dct)->None:
        pass

class ControlPlaneObserver(IObserver):

    def __init__(self,f=None):
        self.f=f

    def update(self,subject,dct)->None:
        msg = '{} {} : Reached to the event.'.format(datetime.now().strftime(cSFMT), self.__str__())
        if subject._state == 0:
            print(msg)
            if self.f is not None:
                self.f.write(msg)

        self.UpdateControlPlane(dct)
        subject._state = 1
        subject.setChanged()
        return

    def UpdateControlPlane(self,dct):

        mdwg=dct['DataFrame']
        cp = dct['ControlPlane']
        ds=dct['Dataset']

        cp.dt_dset = mdwg.names[0]
        cp.rcpower_dset = mdwg.names[1]
        suffics = ".csv"

        file_csv = Path(cp.folder_control_log, cp.rcpower_dset + "_" +
                                      Path(__file__).stem).with_suffix(suffics)


        cp.csv_path =mdwg.to_csv(file_csv)
        ds = Dataset(cp.csv_path, cp.dt_dset, cp.rcpower_dset, cp.discret, cp.fc)  # create dataset
        print("Dataset created")
        if self.f is not None:
            self.f.write("Dataset created\n")
        prepareDataset(cp, ds, cp.fc)

        cp.ts_analysis(ds)
        cp.state=1



class TrainPlaneObserver(IObserver):
    def __init__(self, f=None):
        self.f = f

    def update(self, subject, dct) -> None:
        msg = '{} {} : Reached to the event. The training models is carried out'.format(
            datetime.now().strftime(cSFMT), self.__str__())
        if subject._state == 1:
            print(msg)
            if self.f is not None:
                self.f.write(msg)
            self.TrainModels( dct)

        if  subject._state ==3:
            print(msg)
            if self.f is not None:
                self.f.write(msg)
            self.UpdateModels( dct)
        subject._state = 2
        subject.setChanged()

    def TrainModels(self,dct)->None:
        # mdwg = dct['DataFrame']
        cp = dct['ControlPlane']
        ds = dct['Dataset']
        if self.f is not None:
            self.f.write("\nThe deep learning started...\n\n")
        drive_train(cp, ds)

        if self.f is not None:
            self.f.write("\nThe deep learning finished.\n\n")

    def UpdateModels(self,dct)->None:
        pass


class PredictPlaneObserver(IObserver):
    def __init__(self, f=None):
        self.f = f

    def update(self, subject, dct) -> None:
        msg = '{} {} : Reached to the event. The predict by models   is carried out'.format(
            datetime.now().strftime(cSFMT), self.__str__())
        if subject._state == 2:
            print(msg)
            if self.f is not None:
                self.f.write(msg)
            self.PredictByModels(dct)

        subject._state = 3
        subject.setChanged()

    def PredictByModels(self,dct)->None:
        pass
        cp = dct['ControlPlane']
        ds = dct['Dataset']
        if self.f is not None:
            self.f.write("\nThe prediction started...\n\n")
        drive_predict(cp, ds)

        if self.f is not None:
            self.f.write("\nThe prediction finished.\n\n")

@exec_time
def drive_auto(cp, ds):
    pass


    subject = UpdateChecker(cp.fa)

    observer_a = ControlPlaneObserver(cp.fa)
    subject.attach(observer_a)

    observer_b = TrainPlaneObserver(cp.fa)
    subject.attach(observer_b)

    observer_c = PredictPlaneObserver(cp.fa)
    subject.attach(observer_c)

    start_time=datetime.now()
    if cp.fc  is not None:
        cp.fc.write("\nFirst time to run UpdateChecker {}\n".format(start_time.strftime(cSFMT)))
    nrun = 1
    while 1:
        subject.drive(cp,ds)
        nrun+=1
        curr_time=datetime.now()

        while  curr_time < start_time + td(minutes=2):

            deltat = (start_time + td(seconds=2*60) - curr_time)
            sleep_in_sec =deltat.seconds
            if cp.fc is not None:
                cp.fc.write("\n\nCurrent time is {} . Wait {} seconds to {}th run of  UpdateChecker run\n".format(
                        curr_time.strftime(cSFMT), sleep_in_sec, nrun))

            sleep(sleep_in_sec)
            curr_time = datetime.now()
            continue
        start_time = curr_time
        if cp.fc is not None:
            cp.fc.write("\n\nNext time to run UpdateChecker {}\n".format(start_time.strftime(cSFMT)))

        continue

    subject.detach(observer_a)


    subject.detach(observer_b)
    subject.detach(observer_c)

    return

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