#! /usr/bin/python3
'''
This module

'''

import copy
from predictor.api import d_models_assembly, fit_models, save_modeles_in_repository,  get_list_trained_models
from predictor.api import predict_model, chart_predict, tbl_predict
from predictor.utility import exec_time, msg2log
from predictor.demandwidget import DemandWidget
from abc import ABC, abstractmethod
from datetime import datetime, timedelta as td
from time import sleep
from predictor.utility import cSFMT,incDateStr,decDateStr
from predictor.control import ControlPlane

from predictor.dataset import Dataset
import os
from pathlib import Path
from shutil import copyfile
from predictor.api import prepareDataset


'''
State - machine for UpdateChecker 
'''
SM_CP_CREATE_DATASET = 0
SM_TP_MODEL_TRAINING = 1
SM_CP_UPDATE_DATASET = 2
SM_PP_PREDICTING     = 3
SM_TP_MODEL_UPDATING = 4
SM_CP_DATA_WAITING   = 5
SM_INVALID_STATE     = 6

sm_dict = {SM_CP_CREATE_DATASET: 'CP sends first "GET"-request, receives data and creates a dataset',
           SM_TP_MODEL_TRAINING: 'TP trains NN ans STS models and saves them in the repository',
           SM_CP_UPDATE_DATASET: 'CP sends "GET"-request to add new data to the dataset',
           SM_PP_PREDICTING:     'PP loads model from repository and performs predicting',
           SM_TP_MODEL_UPDATING: 'TP updates the models and saves them in the repository',
           SM_CP_DATA_WAITING:   'CP waits when data will be available',
           SM_INVALID_STATE:     'Invalid State...'}

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
    The UpdateProvider ( or Observable in terms of Observer-pattern) implements the simple state-machine with
    following states and transfers:

        _state = 0 (SM_CP_CREATE_DATASET) - initial state, Control Plane sends the request, receives data and creates
                    a dataset.
        _state = 1 (SM_TP_MODEL_TRAINING) - dataset created. The Train Plane estimates weights of NN (deep learning) and
                    identifies orders of models and parameters   Statistical Models of the Time Series (STS)
        _state = 2 (SM_CP_UPDATE_DATASET) - deep learning finished. The Control Plan sends the request to receive new
                    data and update the dataset.
        _state = 3 (SM_PP_PREDICTING) - Predict Plane performs the forecast
        _state = 4 (SM_TP_MODEL_UPDATING) - Train Plane updates models. The models are updated no more than once an
                    hour, that is, after 6 predictions, since the sampling rate is 10 minutes.
        _state = 5 (SM_CP_DATA_WAITING)  - no new data, wait 10 minutes.

        The transfer sequences are
            0->1->3->2

                |-(no data)---->5
            2-(6 times)-------->3
                |-(each 7th)--->4

            4->3
            5->2
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
        msg2log(self.attach.__name__,msg, self.f)

        return

    def detach(self,observer)->None:
        """

        :type observer: object
        """
        msg = '{} UpdateProvider: detached from observer {}'.format(datetime.now().strftime(cSFMT), observer.__str__())

        self._observers.remove(observer)
        msg2log(self.detach.__name__,msg, self.f)

        return

    def notify(self, dct: object)-> None:
        msg = '{} UpdateProvider: notifying observers..'.format(datetime.now().strftime(cSFMT))
        msg2log(self.notify.__name__,msg, self.f)


        for observer in self._observers:
            observer.update(self, dct)
            msg = '{} UpdateProvider: notifying observers.. The observer {} has notification'.format(
                datetime.now().strftime(cSFMT), observer.__str__())
            msg2log(self.notify.__name__, msg, self.f)

        return

class UpdateChecker(UpdateProvider):

    def __init__(self,f=None):
        super().__init__(f)
        self.dct = {'DataFrame': None, 'ControlPlane': None, 'Dataset': None}

    def isChanged(self): return self.changed

    def setChanged(self): self.changed = 1

    def clearChanged(self): self.changed = 0

    # state - machine functions
    def _sm_CP_CREATE_DATASET(self):
        pass
        self.notify(self.dct)

    def _sm_TP_MODEL_TRAINING(self):
        self.notify(self.dct)

    def _sm_CP_UPDATE_DATASET(self):
        self.notify(self.dct)

    def _sm_PP_PREDICTING(self):

        ds=self.dct['Dataset']
        cp=self.dct['ControlPlane']
        if ds.df is None:
            message=f"""
                The predicting state was got from Descriptor at the program starting.
                The state is : {cp.drtDescriptor['state']}
                Description :  {cp.drtDescriptor['misc']}
                Is need to create the dataset from csv-file : {cp.drtDescriptor['csvDataset']}
            """
            msg2log(self._sm_PP_PREDICTING.__name__, message,self.f)
            ds.csv_path=cp.drtDescriptor['csvDataset']
            prepareDataset(cp, ds, self.f)
            cp.ts_analysis(ds)
            self.dct['ControlPlane']=cp
            self.dct['Dataset']=ds

        self.notify(self.dct)

    def _sm_TP_MODEL_UPDATING(self):
        self.notify(self.dct)

    def _sm_CP_DATA_WAITING(self):
        pass

    def drive(self,cp: ControlPlane, ds: Dataset)->None:
        pass
        # self._state = randrange(0, 3)
        message = f"""
        State-Machine:
        current tme  : {datetime.now().strftime(cSFMT)}
        current state: {self.state}
        description  : {sm_dict[self.state]}
        Transfer to next state...
        """
        msg2log(self.drive.__name__, message, cp.fa)
        self.dct['ControlPlane'] = cp
        self.dct['Dataset']      = ds

        statusDescriptor = self.getDescriptor(cp)
        if statusDescriptor<0:
            cp.drtDescriptor={}
            cp.drtDescriptor['state'] = SM_CP_CREATE_DATASET
            self.state = cp.drtDescriptor['state']

        self.state=cp.drtDescriptor['state']
        sdtn = datetime.now().strftime(cSFMT)
        msg='{}: State -Machine: {}\n'.format(sdtn,sm_dict[self.state])
        msg2log(self.drive.__name__, msg,self.f)

        #     state machine
        if self.state== SM_CP_CREATE_DATASET:
            msg2log(self.drive.__name__, '{} :{}\n'.format(self.state, sm_dict[self.state]), cp.fa)
            self._sm_CP_CREATE_DATASET()

        elif self.state == SM_TP_MODEL_TRAINING :
            msg2log(self.drive.__name__, '{} :{}\n'.format(self.state, sm_dict[self.state]), cp.fa)
            self._sm_TP_MODEL_TRAINING()

        elif self.state == SM_CP_UPDATE_DATASET:
            msg2log(self.drive.__name__, '{} :{}\n'.format(self.state, sm_dict[self.state]), cp.fa)
            self._sm_CP_UPDATE_DATASET()

        elif self.state == SM_PP_PREDICTING:
            msg2log(self.drive.__name__, '{} :{}\n'.format(self.state, sm_dict[self.state]), cp.fa)
            self._sm_PP_PREDICTING()

        elif self.state == SM_TP_MODEL_UPDATING:
            msg2log(self.drive.__name__, '{} :{}\n'.format(self.state, sm_dict[self.state]), cp.fa)
            self._sm_TP_MODEL_UPDATING()

        elif self.state == SM_CP_DATA_WAITING:
            msg2log(self.drive.__name__, '{} :{}\n'.format(self.state, sm_dict[self.state]), cp.fa)
            self._sm_CP_DATA_WAITING()

        else:
            msg="Invalid state of state-machine"
            msg2log(self.drive.__names__, msg, self.f)
            return

        return

    def getDescriptor(self,cp):
        '''
        This method retrieves a descriptor in supposed path. The folder name and descriptor file received for
        configuration  ( cp.folder_descriptor and cp.file_descriptor).

        :param cp:
        :return: 0 - success
                1- descriptor folder not found. The empty folder  created. The descriptor state sets to STATE_START.
                2 - no descriptor into folder. The descriptor state sets to STATE_START
                -1- descriptor folder can not be loaded.
        '''
        supposed_path_to_descriptor_folder = Path(cp.folder_descriptor)
        supposed_path_to_descriptor        = Path(cp.folder_descriptor) / Path(cp.file_descriptor)
        if (not os.path.exists(supposed_path_to_descriptor_folder)) or (not os.path.isdir(supposed_path_to_descriptor_folder)):
            msg = "No descriptors folder {}".format(supposed_path_to_descriptor_folder)
            msg2log(self.getDescriptor.__name__, msg, self.f)

            Path(supposed_path_to_descriptor_folder).mkdir(parents=True, exist_ok=True)

            msg = "The descriptors folder created {}".format(supposed_path_to_descriptor_folder)
            msg2log(self.getDescriptor.__name__, msg, self.f)

            cp.drtDescriptor['state']=SM_CP_CREATE_DATASET
            cp.drtDescriptor['misc'] = sm_dict[SM_CP_CREATE_DATASET]
            return 1
        elif (not os.path.exists(supposed_path_to_descriptor)) or (not os.path.isfile(supposed_path_to_descriptor)):
                msg = "No descriptors into folder {}".format(supposed_path_to_descriptor_folder)
                msg2log(self.getDescriptor.__name__, msg, self.f)
                cp.drtDescriptor['state'] = SM_CP_CREATE_DATASET
                cp.drtDescriptor['misc'] = sm_dict[SM_CP_CREATE_DATASET]
                return 2
        else:
            if  not cp.load_descriptor():
                msg = "The descriptors cannot be loaded {}".format(supposed_path_to_descriptor)
                msg2log(self.getDescriptor.__name__, msg, self.f)
                raise NameError(msg)
                return -1
        return 0

    def parseDescriptor(self,cp):
        '''
        This method returns the state for Observable
        :param cp:
        :return:
        '''

        if cp.drtDescriptor['state']   == SM_CP_CREATE_DATASET:
            pass
        elif cp.drtDescriptor['state'] == SM_TP_MODEL_TRAINING:
            pass
        elif cp.drtDescriptor['state'] == SM_CP_UPDATE_DATASET:
            pass
        elif cp.drtDescriptor['state'] == SM_PP_PREDICTING:
            pass
        elif cp.drtDescriptor['state'] == SM_TP_MODEL_UPDATING:
            pass
        elif cp.drtDescriptor['state'] == SM_CP_DATA_WAITING:
            pass
        else:
            pass

        return

    def setDescriptor(self,cp):
        cp.save_descriptor()
        return






class IObserver(ABC):
    """
    observer's interface
    """
    @abstractmethod
    def update(self,subject: ISubject, dct)->None:
        pass





class ControlPlaneObserver(IObserver):
    '''
    This concrete Observer class listens a notification from Observable UpdateProvider.
    A dataset being created in the real time is saved in folowing folder:
    <cp.folder_rt_datasets>/<type_id>, where type_id gets from the header of the  requested widget
    '''

    def __init__(self,f=None):
        self.f=f
        self.wait_index = 0
        self.wait_max_index = 2

    def update(self,subject,dct)->None:
        msg = '{} {} : Reached to the event.'.format(datetime.now().strftime(cSFMT), self.__str__())
        msg2log(self.update.__name__, msg, self.f)
        if subject.state == SM_CP_CREATE_DATASET:
            self.createDataset(dct)
            self.updateControlPlane(dct)
            cp=dct['ControlPlane']
            dmwdg=dct['DataFrame']
            ds=dct['Dataset']
            cp.drtDescriptor["csvDataset"] = cp.csv_path
            cp.drtDescriptor['typeID'] = dmwdg.type_id
            cp.drtDescriptor['title'] = dmwdg.title
            cp.drtDescriptor['lastTime'] = dmwdg.last_time
            cp.drtDescriptor['state'] = SM_CP_CREATE_DATASET
            cp.drtDescriptor['misc'] = sm_dict[SM_CP_CREATE_DATASET]

            cp.state = SM_TP_MODEL_TRAINING
            cp.save_descriptor()
            dct['ControlPlane'] = cp

            subject.state = SM_TP_MODEL_TRAINING

        elif subject.state == SM_CP_UPDATE_DATASET:
            status = self.updateDataset( dct)
            cp = dct['ControlPlane']
            cp.drtDescriptor['modelRepository'] = Path(cp.path_repository) / Path(cp.rcpower_dset)
            cp.drtDescriptor['state'] = SM_CP_UPDATE_DATASET
            cp.drtDescriptor['misc'] = sm_dict[SM_CP_UPDATE_DATASET]

            if status == 0:
                self.wait_index = 0
                dmwdg = dct['DataFrame']
                self.updateControlPlane(dct)
                cp.drtDescriptor['lastTime'] = dmwdg.last_time
                cp.state = SM_PP_PREDICTING
                subject.state = SM_PP_PREDICTING
            else:
                cp.state = SM_CP_DATA_WAITING
                subject.state = SM_CP_DATA_WAITING

            dct['ControlPlane'] = cp
            cp.save_descriptor()

        elif subject.state ==  SM_CP_DATA_WAITING:
            cp = dct['ControlPlane']
            sleep(cp.discret * 30)
            self.wait_index+=1
            if self.wait_index > self.wait_max_index:
                msg="Can not get an update data for time series after {} attempts.\n".format(self.wait_max_index)
                msg2log(self.update.__name__, msg,self.f)
                subject.state=SM_INVALID_STATE
                return


            subject.state=SM_CP_UPDATE_DATASET


        # subject.setChanged()
        return

    def createDataset(self,dct):
        pass
        scaled_data = False
        start_time = "2020-08-30 00:00:00"
        end_time_t = datetime.now()
        end_time = end_time_t.strftime(cSFMT)
        start_time_t = end_time_t - td(days=3)
        start_time = start_time_t.strftime(cSFMT)

        dmwdg = DemandWidget(scaled_data, start_time, end_time, 'hour', None, None, self.f)
        dmwdg.set_url()

        print(dmwdg.url)

        requested_widget = dmwdg.getDemandRT(None)
        print("Requested widget has type {}".format(type(requested_widget)))

        dmwdg.plot_ts(os.getcwd(), False)
        dmwdg.autocorr_show(os.getcwd(), False)

        dct['DataFrame'] = dmwdg  # 'ControlPlane': cp, 'Dataset': ds]

        return

    def updateDataset(self,dct)->int:
        nRet = 1

        cp = dct['ControlPlane']

        cp.csv_path =cp.drtDescriptor["csvDataset"]

        scaled_data= False
        start_time = incDateStr(cp.drtDescriptor['lastTime'], minutes=cp.discret)
        end_time = datetime.now().strftime(cSFMT)
        dmwdg = DemandWidget(scaled_data, start_time, end_time, 'hour', None, None, self.f)
        dmwdg.set_url()
        print(dmwdg.url)

        requested_widget = dmwdg.getDemandRT(None)
        if requested_widget is None:
            nRet = -1

        if dmwdg.ts_size> 0:
            df_new = dmwdg.concat_with_df_from_csv( cp.drtDescriptor["csvDataset"] )
            bak_csv_str0 =str(cp.drtDescriptor["csvDataset"]).replace('.csv', '.' +
                            cp.drtDescriptor['lastTime'].replace(' ','_') +'.bak')
            bak_csv_str1 = bak_csv_str0.replace(':','_')
            bak_csv_str = bak_csv_str1.replace('+', '_')
            bak_csv_file=Path(bak_csv_str)

            copyfile(Path(cp.drtDescriptor["csvDataset"]), bak_csv_file)

            df_new.to_csv(Path(cp.drtDescriptor["csvDataset"]))
            dmwdg.df=df_new
            dmwdg.ts_size=len(df_new)
            dct['DataFrame']=dmwdg
            dct['ControlPlane'] = cp
            nRet=0

        return nRet

    def updateControlPlane(self,dct):

        dmwdg=dct['DataFrame']
        cp = dct['ControlPlane']
        ds=dct['Dataset']

        cp.dt_dset = dmwdg.names[0]
        cp.rcpower_dset = dmwdg.names[1]
        suffics = ".csv"

        # file_csv = Path(cp.folder_control_log, cp.rcpower_dset + "_" +
        #                               Path(__file__).stem).with_suffix(suffics)

        dataset_folder = Path(cp.folder_rt_datasets) / str(dmwdg.one_word_title)
        Path(dataset_folder).mkdir(parents=True, exist_ok=True)

        file_csv = Path(dataset_folder, cp.rcpower_dset + "_" + Path(__file__).stem).with_suffix(suffics)

        cp.csv_path =dmwdg.to_csv(file_csv)
        ds = Dataset(cp.csv_path, cp.dt_dset, cp.rcpower_dset, cp.discret, cp.fc)  # create dataset
        msg ="Dataset (csv-file) created"
        msg2log(self.updateControlPlane.__name__,msg, self.f)
        prepareDataset(cp, ds, cp.fc)
        dct["Dataset"]=ds
        cp.ts_analysis(ds)

        return

class TrainPlaneObserver(IObserver):
    '''
    This concrete Observer class listens a notification from Observable UpdateProvider.
    '''
    def __init__(self, f=None):
        self.f = f

    def update(self, subject, dct) -> None:
        msg = '{} {} : Reached to the event. The training models is carried out'.format(
            datetime.now().strftime(cSFMT), self.__str__())
        if subject._state == SM_TP_MODEL_TRAINING:
            msg2log(self.update.__name__, msg, self.f)
            self.TrainModels( dct)

        if  subject._state ==SM_TP_MODEL_UPDATING:
            msg2log(self.update.__name__, msg, self.f)
            self.UpdateModels( dct)

        subject._state = 2
        subject.setChanged()

    def TrainModels(self,dct)->None:
        # dmwdg = dct['DataFrame']
        cp = dct['ControlPlane']
        ds = dct['Dataset']
        msg ='\nThe deep learning started...\n\n'
        msg2log(self.TrainModels.__name__,msg, self.f)

        drive_train(cp, ds)
        cp.drtDescriptor['modelRepository'] = Path(cp.path_repository) / Path(cp.rcpower_dset)
        cp.drtDescriptor['state'] = SM_PP_PREDICTING  # next state
        cp.drtDescriptor['misc'] = sm_dict[SM_PP_PREDICTING]


        cp.save_descriptor()
        msg = "\nThe deep learning finished.\n\n"
        msg2log(self.TrainModels.__name__, msg, self.f)
        dct['ControlPlane']=cp

    def UpdateModels(self,dct)->None:
        cp = dct['ControlPlane']
        ds = dct['Dataset']
        msg = '\nDP and STS model re-estimation  started ...\n\n'
        msg2log(self.TrainModels.__name__, msg, self.f)

        drive_train(cp, ds)
        cp.drtDescriptor['modelRepository'] = Path(cp.path_repository) / Path(cp.rcpower_dset)
        cp.drtDescriptor['state'] = SM_PP_PREDICTING  # next state
        cp.drtDescriptor['misc'] = sm_dict[SM_PP_PREDICTING]

        cp.save_descriptor()
        msg = "\nDP and STS model re-estimation finished.\n\n"
        msg2log(self.TrainModels.__name__, msg, self.f)
        dct['ControlPlane'] = cp


class PredictPlaneObserver(IObserver):
    '''
    This concrete Observer class listens a notification from Observable UpdateProvider.
    '''
    def __init__(self, f=None):
        self.f = f
        self.predict_index =0
        self.predict_index_max = 6

    def update(self, subject, dct) -> None:
        msg = '{} {} : Reached to the event. The predict by models   is carried out'.format(
            datetime.now().strftime(cSFMT), self.__str__())
        if subject._state == 2:
            msg2log(self.update.__name__, msg, self.f)
            self.PredictByModels(dct)

        subject._state = 3
        subject.setChanged()

    def PredictByModels(self,dct)->None:
        pass
        cp = dct['ControlPlane']
        ds = dct['Dataset']

        msg2log(self.PredictByModels.__name__, '\nThe prediction started...\n', self.f)

        drive_predict(cp, ds)
        self.predict_index+=1
        cp.drtDescriptor['state'] = SM_CP_UPDATE_DATASET  # next state
        cp.drtDescriptor['musc'] = sm_dict[SM_CP_UPDATE_DATASET]
        if self.predict_index>=self.predict_index_max:
            cp.drtDescriptor['state'] = SM_TP_MODEL_UPDATING  # next state
            cp.drtDescriptor['musc'] = sm_dict[SM_TP_MODEL_UPDATING]

        cp.save_descriptor()
        msg2log(self.PredictByModels.__name__, '\nThe prediction finished.\n', self.f)
        return

@exec_time
def drive_auto(cp, ds):
    '''
    This drive_auto() function manages data processing flow in auto real-time mode.

    :param cp: ControlPlan object (class implementation
    :param ds: Dataset object

    :return:
    '''
    pass


    subject = UpdateChecker(cp.fa)
    subject.setState = 0

    observer_a = ControlPlaneObserver(cp.fa)
    subject.attach(observer_a)

    observer_b = TrainPlaneObserver(cp.fa)
    subject.attach(observer_b)

    observer_c = PredictPlaneObserver(cp.fa)
    subject.attach(observer_c)

    start_time=datetime.now()
    msg="\nFirst time to run UpdateChecker {}\n".format(start_time.strftime(cSFMT))
    msg2log(drive_auto.__name__, msg, cp.fa)

    nrun = 1
    while 1:
        subject.drive(cp,ds)
        nrun+=1
        curr_time=datetime.now()

        while  curr_time < start_time + td(minutes=5):

            deltat = (start_time + td(seconds=5*60) - curr_time)
            sleep_in_sec =deltat.seconds
            msg="\n\nCurrent time is {} . Wait {} seconds to {}th run of  UpdateChecker run\n".format(
                        curr_time.strftime(cSFMT), sleep_in_sec, nrun)
            msg2log(drive_auto.__name__, msg, cp.fa)

            sleep(sleep_in_sec)
            curr_time = datetime.now()
            continue
        start_time = curr_time
        msg="\n\nNext time to run UpdateChecker {}\n".format(start_time.strftime(cSFMT))
        msg2log(drive_auto.__name__, msg, cp.fa)

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
    if ds is None or ds.rcpower is None or len(ds.rcpower)==0:
        msg2log(drive_predict.__name__," The dataset is empty now. The predict step is skipping", cp.fa)
        return
    ds.data_for_predict = cp.n_steps
    ds.predict_date = None
    predict_history = copy.copy(ds.data_for_predict)

    dict_model = get_list_trained_models(cp)
    n_predict=4
    dict_predict = predict_model(dict_model, cp, ds, n_predict)


    # title = '{} (Short Term Predict on {} steps)'.format(cp.rcpower_dset, n_predict)
    title='Short Term Predict'
    chart_predict(dict_predict, n_predict, cp, ds, title, cp.rcpower_dset)

    tbl_predict(dict_predict, n_predict, cp, ds, title)

    cp.forecast_number_step=cp.forecast_number_step+1

    return

def drive_control(cp, ds):
    pass