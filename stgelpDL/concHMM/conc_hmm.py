#!/usr/bin/env python3

""" HMM base class."""

""" DataHelper class implements a following tasks:
 - read data from dataset;
 - form a train and a test sequences.
An initalization parameters are following:
 - dataset (file csv);
 - observations-timeseries column name in the dataset ;
 - timestamps - timestamp labels column in the dataset;
 - exogenious - column names in the dataset for exogenious;
 - endogenious - column names in the dataset for endogenious.
You have two ways define 'timeseries' in the dataset. First, explicity select by name passed in 'observations' 
parameter. The  endogenious list should be empty. Second, the imbalance that is a substract 
endogenious[0] - endogenious[1]. The 'observations' parameter should be None.
All observations (timeseries ) are used for model learning by using maximum likelyhood estimation methods.
The model M(m,n,S,pi,A,B) where m -number of states; n - number of observation; S-  states set S0,S1,...,Sm-1; 
pi -init probabilities vectop  of m-size; A-transmission matrix m x m ; B -emission matrix m*n.  
   
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from json import dump,load
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from conc_cfg import IMBALANCE, TEST_SEQUENCE_SIZE, EVALUATION_SEQUENCE_SIZE, PATH_ROOT_FOLDER, \
    CLASSIFICATION_FOR_STATE_EXTRACTION, PCA_CLASSIFICATION_FOR_STATE_EXTRACTION, SIMPLE_STATE_EXTRACTION, MIN_STATE, \
    MAX_STATE, EXOGENIOUS_CLASSIFICATION_ONLY
from conc_api import paiMLE, emisMLE, transitionsMLE, getEvalSequenceStartIndex,  logViterbiPath, logMarginPrb, \
    execution_time, getClusters, getPCAClusters
from plt_api import pltViterbi

logger = logging.getLogger(__name__)

""" The DataHelper class implements the functionality for original dataset reading and parsing (ReadData(), 
ClusterData() -methods). This class contains also a methods for labeling each observations by state according by 
harcoded rules ( getStateSequence(),getStateSequenceClusteringData()).
The method dumpStates() serializes the list of state sequences in csv-file 

"""
class DataHelper():
    pass
    def __init__(self,dataset:str=None, observations:str=None,timestamps:str=None,endogenious:list=[],
                 exogenious:list = [] ):
        """

        :param dataset:
        :param observations:
        :param timestamps:
        :param exogenious:
        """
        self._log = logger
        self.dataset = dataset
        self.observations_name =observations
        self.dt_name =timestamps
        self.exogenious=exogenious
        self.endogenious=endogenious
        self.df:pd.DataFrame = None
        self.observations =[]
        self.dt = []
        self.amin = 0.0
        self.amax = 0.0
        self.mean = 0.0
        self.std = 0.0
        self.range =0.0
        self.extracted_state_sequences = {}
        self.extracted_state_names = {}

        file_stem = str(Path(self.dataset).stem)
        file_parent = Path(self.dataset).parent
        file_parent = Path(PATH_ROOT_FOLDER)
        self.dump_dataset = str(Path(file_parent / "Logs" /"dump_state_sequence_{}".format(file_stem)).with_suffix(".csv"))
        self.dump_json = str(Path(file_parent / "Logs" /"dump_state_sequence_{}".format(file_stem)).with_suffix(".json"))
        self.model_json_dumps =str(Path(file_parent / "models"))
        self.file_png_path = str(Path(file_parent / "Logs"))

    def readData(self):

        if Path(self.dataset).exists():
            self.df = pd.read_csv(self.dataset)
        else:
            self._log.error("dataset {} is not exist".format(self.dataset))
            return
        if self.observations_name is None and len(self.endogenious)<2:
            self._log.error("Timeseries name is not correct. Or observations should be pointed on one of columns,")
            self._log.error("or endogenious list should contain two items for 'Imbalance'.")
            self._log.error("Then timeseries should be substract : 'Imbalance' := endogenious[0] -endogenious[1] ")
            return

        self.dt = self.df[self.dt_name].values

        if self.observations_name is None :
            self.observations_name=IMBALANCE
            self.observations=[ round(self.df[self.endogenious[0]].values[i] - self.df[self.endogenious[1]].values[i],4)
                      for i in range(len(self.df))]
            self.observations = np.array(self.observations)
            pass
        else:
            self.observations=self.df[self.observations_name].values
        self.mean = round(np.mean(self.observations), 4)
        self.std = round(np.std(self.observations), 4)
        self.amin = np.amin(self.observations)
        self.amax = np.amax(self.observations)
        self.range = self.amax - self.amin

    def clusterData(self, title:str=None, pca_title:str=None)->(dict,dict,dict,dict):
        pass
        x_list = []
        corr_head=[]
        if EXOGENIOUS_CLASSIFICATION_ONLY==0:
            if self.observations_name==IMBALANCE:
                x_list.append(self.observations.tolist())
                corr_head.append(IMBALANCE)
            else:
                for item in self.endogenious:
                    x_list.append(self.df[item].values)
                    corr_head.append(item)
        for item in self.exogenious:
            x_list.append(self.df[item].values)
            corr_head.append(item)

        X=np.array(x_list).transpose()
        (n,m)=X.shape

        if m>1: # correlation matrix
            R = np.corrcoef(X, rowvar=False)
            (k,k) = R.shape
            msg = "\n               "
            msg0 ="".join("{:<15s} ".format(item) for item in corr_head)
            msg =msg + msg0 +"\n"
            for i in range(k):
                msg1 = "{:<15s}".format(corr_head[i])
                msg2 = "".join("{:^15.4f}     ".format(round(R[i][j], 3)) for j in range(k))
                msg=msg +msg1 + msg2+ "\n"

            self._log.info(msg)
            print(msg)

        if title is None:
            title = "Clusterization"
        cluster_centers , cluster_labels = getClusters(file_png_path=self.file_png_path, title= title, X=X,
                                                        labels=np.array(self.df[self.dt_name].values), max_clusters = 5)

        if pca_title is None:
            pca_title = "PCA_Clusterization"

        pca_cluster_centers = None
        pca_cluster_labels = None
        if m > 2:
            pca_cluster_centers, pca_cluster_labels = getPCAClusters(file_png_path=self.file_png_path,
                                                                 title=pca_title, X=X,
                                                                 labels=np.array(self.df[self.dt_name].values),
                                                                 max_clusters=5)

        return cluster_centers, cluster_labels, pca_cluster_centers, pca_cluster_labels


    def __str__(self):
        msg_=""
        if len(self.observations)>10:
            msg_=msg_ + ''.join("{}: {}\n".format(self.dt[i],self.observations[i]) for i in range(5))
            msg_=msg_ + ''.join("....\n....\n")
            msg_=msg_ + ''.join("{}: {}\n".format(self.dt[i], self.observations[i]) for i in range(len(self.observations)-4,len(self.observations)))
        msg__ = "".join("{}:{}\n".format(k,v) for k,v in self.extracted_state_sequences.items())

        msg = f"""
Dataset     : {self.dataset}
Dump Dataset: {self.dump_dataset}  ( this one contains extracted state sequences )
Dump(json)  : {self.dump_json}
Time Series : {self.observations_name}
Timestamps  : {self.dt_name}
Exogenious  : {self.exogenious}
Data        :
{msg_}

Statistics  :
Mean        : {self.mean}
Std         : {self.std}
Min         : {self.amin}
Max         : {self.amax}
Range       : {self.range}

Extracted state sequences:
{msg__}

"""
        self._log.info(msg)
        print(msg)

    """ This method extracts states as steps in timeseries.
    The bins list is generates according by self.amin and self.range
    The hit of i-th observation in the j-th bin is 'j'-state for observation. 
    """
    def statesfromsteps(self, n_states:int):

        delta = self.range/(float(n_states))
        bins = [[self.amin + delta*i , self.amin + delta *(i+1)] for i in range(n_states)]
        bins[-1][1]=bins[-1][1]+ delta  # not to lose the upper limit value!
        n = len(self.observations)
        state_sequence =[0 for i in range(n)]
        state_names = ["{}<=y<{}".format(round(bins[i][0],2),round(bins[i][1],2)) for i in range(n_states)]

        for i in range(n):
            for j in range(n_states):
                if self.observations[i]>=bins[j][0] and self.observations[i]<bins[j][1]:
                    state_sequence[i] = j
                    break
        check_states, check_counts = np.unique(np.array(state_sequence), return_counts=True)
        self._log.info("Check extracted states for 'states from steps' algorithm\nStates:{}\nHistogram:{}".format(
            check_states, check_counts))

        if n_states!=len(check_states):
            self._log.error("Error state extraction for {} states. Here is losing state!".format(n_states))
            state_sequence = None
            state_names = None
        return state_sequence, state_names

    @execution_time
    def getStateSequence(self,n_states: int = 2) -> (list,list):
        state_sequence = []
        state_names =[]
        (n,)=self.observations.shape
        if n_states ==2:
            state_sequence=[0 if self.observations[i]<0.0 else 1 for i in range(n)]
            state_names=['y<0','y>=0']
        elif n_states == 3:
            for i in range(n):
                st=0
                y = self.observations[i]
                if y>=-0.1*self.range and y <0.1*self.range:
                    st = 1
                elif y >=0.1*self.range:
                    st =2
                else:
                    st =0
                state_sequence.append(st)
            state_names = ['y<-e', 'y>=-e; y<e','y>=e']

        elif n_states == 4:
            for i in range(n):
                st = 0
                y = self.observations[i]
                dy=y
                if i>0:
                    dy=dy -self.observations[i-1]
                if y <0.0 and dy<0:
                    st = 0
                elif y <0.0 and dy>=0.0:
                    st = 1
                elif y>=0.0 and dy<0:
                    st = 2
                elif y>=0.0 and dy>=0.0:
                    st = 3
                else:
                    st = 0
                state_sequence.append(st)

            state_names = ['y<0;dy<0', 'y<0;dy>=0', 'y>=0;dy<0','y>=0;dy>=0']

        elif n_states == 5:
            e =0.1 * self.range
            for i in range(n):
                st = 0
                y = self.observations[i]
                dy = y
                if i > 0:
                    dy = dy - self.observations[i - 1]
                if y < -e and dy < 0:
                    st = 0
                elif y < -e and dy >= 0.0:
                    st = 1
                elif y >=-e   and  y < e:
                    st = 2
                elif y >= e and dy < 0:
                    st = 3
                elif y >= e and dy >= 0.0:
                    st = 4
                else:
                    st = 0
                state_sequence.append(st)
            state_names = ['y<0;dy<0', 'y<0;dy>=0','y~0', 'y>=0;dy<0','y>=0;dy>=0']

        elif n_states >= 6:
            state_sequence, state_names = self.statesfromsteps(n_states)

        else:
            pass
        return state_sequence, state_names

    """ Let y is observations. 2 states: S0 is y[t]<0, S1 is y[t]>0. """

    @execution_time
    def getMLE(self, n_states: int = 2) -> (list, np.array, np.array, np.array, list):
        (n,)=self.observations.shape
        state_sequence, state_names = self.getStateSequence(n_states=n_states)
        states, counts = np.unique(state_sequence, return_counts=True)
        pai = paiMLE(states, counts, n)
        A = transitionsMLE(state_sequence, states )
        B = emisMLE(self.observations, state_sequence, states)

        return state_names, pai, A, B, state_sequence

    @execution_time
    def dumpStateSequences(self):
        pass
        self.readData()
        if self.observations_name == IMBALANCE:
           self.df[IMBALANCE] = self.observations.tolist()

        if SIMPLE_STATE_EXTRACTION == 1:
            for i in range(MIN_STATE,MAX_STATE + 1):
                state_sequence, state_name = self.getStateSequence(n_states=i)
                if state_sequence is None or state_name is None: # to prevent missing states in sequence
                    break

                name="St{}".format(i)
                self.df[name]=state_sequence
                self.extracted_state_sequences[name] = i
                self.extracted_state_names[name] = state_name
                self._log.info("The state sequence for model {} with {} states extracted".format(name,i))

        if CLASSIFICATION_FOR_STATE_EXTRACTION == 1 or PCA_CLASSIFICATION_FOR_STATE_EXTRACTION == 1:
            _, d_cl_sequences, _, d_pca_cl_sequences = self.clusterData()

            if CLASSIFICATION_FOR_STATE_EXTRACTION == 1:
                for key,value in d_cl_sequences.items():
                    name="CSt{}".format(key)
                    self.df[name] = value
                    self.extracted_state_sequences[name] = key
                    self.extracted_state_names[name] = ["S{}".format(i) for i in range(int(key))]
                    self._log.info("The state sequence for model {} with {} states extracted".format(name, key))
                del d_cl_sequences

            if PCA_CLASSIFICATION_FOR_STATE_EXTRACTION == 1:
                for key,value in d_pca_cl_sequences.items():
                    name="PCACSt{}".format(key)
                    self.df[name] = value
                    self.extracted_state_sequences[name] = key
                    self.extracted_state_names[name] = ["S{}".format(i) for i in range(int(key))]
                    self._log.info("The state sequence for model {} with {} states extracted".format(name, key))
                del d_pca_cl_sequences

        self.df.to_csv(self.dump_dataset, index=False)
        self._log.info("The dump is containing extracted state sequences created: {}".format(self.dump_dataset))

        with open(self.dump_json, 'w') as fw:
            dump(self.extracted_state_names, fw)
            self._log.info("The dump is containing the names of state sequences created: {}".format(self.dump_json))

        return

    @execution_time
    def getMLEmodel(self,name_model:str="St2"):
        (n,) = self.observations.shape
        state_sequence = self.df[name_model].values
        states, counts = np.unique(state_sequence, return_counts=True)
        pai = paiMLE(states, counts, n)
        A = transitionsMLE(state_sequence, states)
        B = emisMLE(self.observations, state_sequence, states)

        return [], pai, A, B, state_sequence






class HMModel():
    def __init__(self, name:str="HMM", state_names:list=[],pai:np.array=None,a: np.array = None, b: np.array = None):
        self._log = logger
        self.n_states=len(state_names)
        self.states=[i for i in range(self.n_states)]
        self.names=state_names
        self.pai=pai
        self.a=a
        self.b=b

    def __str__(self):
        msg = f"""
number of states: {self.n_states}
states          : {self.names}
state indexes   : {self.states}
initial (Pai)   : {self.pai}
transitions (A) : 
{self.a}
emissions (B)   : 
{self.b}
"""
        print(msg)
        self._log.info(msg)


class HMMpropagator():
    """
    This class contains the data and methods for Hidden Markov model propogation along timeseries.
    The observation (time series) splits on evaluation ( >90%) and test (<~>10% of timeseries) sequences.
    The test sequence is need only for forecast accuracy estimation. Maybe it is empty.
    """

    def __init__(self,hmm: HMModel = None, observations: np.array = None, timestamps: np.array = None, n_test:int=0):
        pass
        self._log = logging.getLogger(self.__class__.__name__)
        self.hmm = hmm
        self.observations = observations
        self.timestamp = timestamps
        self.tfd_model = None
        self.n_test = n_test
        (self.n,) = self.observations.shape
        self.n=self.n-self.n_test

    @execution_time
    def drive_tfd_model(self)->(np.array,str,np.array):
        pai = tf.convert_to_tensor(self.hmm.pai, dtype=tf.float64)
        transitDist = tf.convert_to_tensor(self.hmm.a, dtype=tf.float64)

        initial_distribution = tfd.Categorical(probs=pai)

        transition_distribution = tfd.Categorical(probs=transitDist)

        mean_list = self.hmm.b[:, 0].tolist()
        std_list =  self.hmm.b[:, 1].tolist()
        mean_list = tf.convert_to_tensor(mean_list, dtype=tf.float64)
        std_list = tf.convert_to_tensor(std_list, dtype=tf.float64)
        observation_distribution = tfd.Normal(loc=mean_list, scale=std_list)

        self.tfd_model = tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=observation_distribution,
            num_steps=self.n)

        observations_tenzor = tf.convert_to_tensor(self.observations.tolist(), dtype=tf.float64)

        post_mode = self.tfd_model.posterior_mode(observations_tenzor)
        msg = "Posterior mode\n\n{}\n".format(post_mode)
        # msg2log(drive_HMM.__name__, msg, cp.fp)
        self._log.info(msg)

        post_marg = self.tfd_model.posterior_marginals(observations_tenzor)
        msg = "{}\n\n{}\n".format(post_marg.name, post_marg.logits)
        # msg2log(drive_HMM.__name__, msg, cp.fp)
        self._log.info(msg)

        mean_value = self.tfd_model.mean()
        msg = "mean \n\n{}\n".format(mean_value)
        # msg2log(drive_HMM.__name__, msg, cp.fp)
        self._log.info(msg)

        log_probability = self.tfd_model.log_prob(observations_tenzor)
        msg = "Log probability \n\n{}\n".format(log_probability)
        # msg2log(drive_HMM.__name__, msg, cp.fp)
        self._log.info(msg)

        # plotViterbiPath(str(len(observations)), observation_labels, post_mode.numpy(), states_set, cp)

        return post_mode.numpy(), post_marg.name, post_marg.logits



    def get_mean(self):
        mean = self.tfd_model.mean()

        return mean

    def get_logprob(self,value):
        pass


@execution_time
def trainPath(dhlp:DataHelper = None, eval_start:int = 0, eval_ar:np.array = None,
              eval_timestamps_ar:np.array = None) -> list:

    report_list = []
    for hmm_model,n_states in dhlp.extracted_state_sequences.items():
        state_names=dhlp.extracted_state_names[hmm_model]
        logger.info("\n\n\n{} model with {} states\n{}\n".format(hmm_model, n_states, state_names))
        _, pai, A, B, state_sequence = dhlp.getMLEmodel(name_model=hmm_model)
        eval_hidden_sequence = [state_sequence[i] for i in range(eval_start,
                                                            eval_start + EVALUATION_SEQUENCE_SIZE + TEST_SEQUENCE_SIZE)]

        message=drive_HMMext(dhlp=dhlp, model_title=hmm_model, n_states=n_states,
                     state_names=state_names, eval_start = eval_start,
                     eval_ar=eval_ar, eval_timestamps_ar=eval_timestamps_ar, pai=pai, A=A, B=B,
                     eval_hidden_sequence=eval_hidden_sequence)
        model_json=str(Path(Path(dhlp.model_json_dumps)/Path(hmm_model)).with_suffix(".json"))
        with open(model_json,'w') as fmodel:
            dump({"states":state_names, "pai":pai.tolist(),"A":A.tolist(), "B":B.tolist()},fmodel)
        logger.info("{} dumped to {}".format(hmm_model,model_json))
        logger.info("\n\n")
        report_list.append(message)

    return report_list

@execution_time
def drive_HMMext(dhlp:DataHelper = None, model_title:str = "HMM", n_states:int = 2, state_names: list =[],
                 eval_start:int = 0, eval_ar: np.array=None, eval_timestamps_ar:np.array = None, pai:np.array =None,
                 A:np.array =None, B:np.array = None, eval_hidden_sequence:list = [])->str:


    if dhlp is None:
        logger.error("Error : DataHelper object missed in call {} arguments".format(drive_HMMext.__name__))
        return
    if eval_ar is None:
        logger.error("Error : Observations missed in call {} arguments".format(drive_HMMext.__name__))
        return
    if eval_timestamps_ar is None:
        logger.error("Error : Observation timestamps missed in call {} arguments".format(drive_HMMext.__name__))
        return

    if pai is None:
        logger.error("Error : Initial probabilities pai- vector missed in call {} arguments".format(
            drive_HMMext.__name__))
        return
    if A is None:
        logger.error("Error : Transmission matrix A missed in call {} arguments".format(drive_HMMext.__name__))
        return
    if B is None:
        logger.error("Error : Emission matrix B missed in call {} arguments".format(drive_HMMext.__name__))
        return
    if len(eval_hidden_sequence) == 0:
        logger.error("Error : Eval.hidden state sequence missed in call {} arguments".format(drive_HMMext.__name__))
        return

    if len(state_names)==0:
        state_names = ["S{}".format(i) for i in range(n_states)]
    hmm = HMModel(name=model_title, state_names=state_names, pai=pai, a=A, b=B)
    hmm.__str__()
    (n,)=eval_ar.shape
    hmm_drive = HMMpropagator(hmm=hmm, observations=eval_ar, timestamps=eval_timestams_ar,
                              n_test=TEST_SEQUENCE_SIZE)
    viterbi_path, aux_name, logits_ar = hmm_drive.drive_tfd_model()

    pltViterbi(file_png_path=dhlp.file_png_path, pref_file_name=model_title,
               subtitle="starting at {}".format(eval_timestams_ar[0]), observations=eval_ar, viterbi_path=viterbi_path,
               hidden_sequence=eval_hidden_sequence)

    logger.info("logits_ar\n{}".format(logits_ar))
    logMarginPrb(model_title, logits_ar, eval_timestams_ar)
    match_cnt,match_rate = logViterbiPath(model_title, viterbi_path, eval_timestams_ar, eval_hidden_sequence)

    message=f"""
***************************************************************************************************************    
HMModel: {model_title} States number: {n_states}
States: {state_names}
Model estimation:
Pai:{pai}
Trnsmission matrix: 
{A}
Emission matrix (mean,std):
{B}


Viterbi path for evaluation data {n} -size starting at {eval_timestamps_ar[0]}
Match count : {match_cnt}  Rate: {match_rate}

***************************************************************************************************************
"""
    logger.info(message)
    return message




@execution_time
def drive_HMM(dhlp:DataHelper = None, model_title:str = "HMM", n_states:int = 2, eval_start:int = 0,
              eval_ar: np.array=None, eval_timestamps_ar:np.array = None):
    """

    :param dhlp:
    :param model_title:
    :param n_states:
    :param eval_start:
    :param eval_ar:
    :param eval_timestams_ar:
    :return:
    """

    if dhlp is None or eval_ar is None or eval_timestamps_ar is None:
        logger.error("Arguments {} are not set correctly".format(drive_HMM.__name__))
        return

    state_names, pai, A, B, state_sequence = dhlp.getMLE(n_states=n_states)
    hidden_sequence = [state_sequence[i] for i in range(eval_start,
                                                         eval_start + EVALUATION_SEQUENCE_SIZE + TEST_SEQUENCE_SIZE)]

    hmm = HMModel(name=model_title, state_names=state_names, pai=pai, a=A, b=B)
    hmm.__str__()

    hmm_drive = HMMpropagator(hmm=hmm, observations=eval_ar, timestamps=eval_timestams_ar,
                               n_test=TEST_SEQUENCE_SIZE)
    viterbi_path, aux_name, logits_ar = hmm_drive.drive_tfd_model()

    pltViterbi(pref_file_name=model_title, subtitle="starting at {}".format(eval_timestams_ar[0]),
               observations=eval_ar, viterbi_path=viterbi_path, hidden_sequence=hidden_sequence)

    logger.info("logits_ar\n{}".format(logits_ar))
    logMarginPrb(model_title, logits_ar, eval_timestams_ar)
    logViterbiPath(model_title, viterbi_path, eval_timestams_ar, hidden_sequence)

    return



if __name__ == "__main__":
    filename = str(Path(PATH_ROOT_FOLDER/"Logs"/"log").with_suffix(".log"))
    logging.basicConfig(filename=filename, filemode='a',
                        level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    dataset = '/home/dmitry/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_20102020_conchmm.csv'
    ts = "Real_demand"
    dt = "Date Time"
    endogenious=['Programmed_demand','Real_demand']
    # exogenious = ['Diesel_Power', 'WindGen_Power','HydroTurbine_Power','Pump_Power']
    exogenious = ['Diesel_Power', 'WindGen_Power', 'Hydrawlic']
    dhlp=DataHelper(dataset=dataset,timestamps=dt,endogenious=endogenious, exogenious=exogenious)
    # dhlp.readData()
    # dhlp.__str__()
    # dhlp.clusterData()

    dhlp.dumpStateSequences()
    dhlp.__str__()
    (n_train,)=dhlp.observations.shape
    eval_start = getEvalSequenceStartIndex(n_train=n_train, n_eval=EVALUATION_SEQUENCE_SIZE, n_test=TEST_SEQUENCE_SIZE)

    eval_ar = np.array([dhlp.observations[i] for i in range(eval_start,
                                                            eval_start+ EVALUATION_SEQUENCE_SIZE + TEST_SEQUENCE_SIZE)])
    eval_timestams_ar = np.array([dhlp.dt[i] for i in range(eval_start,
                                                        eval_start + EVALUATION_SEQUENCE_SIZE + TEST_SEQUENCE_SIZE)])
    test_ar = np.array([dhlp.observations[i] for i in range(eval_start+ EVALUATION_SEQUENCE_SIZE,
                                                         eval_start + EVALUATION_SEQUENCE_SIZE + TEST_SEQUENCE_SIZE )])

    report_list = trainPath(dhlp=dhlp, eval_start=eval_start, eval_ar=eval_ar, eval_timestamps_ar=eval_timestams_ar)

    with open("Report.log", 'w') as frep:
        for item in report_list:
            frep.write(item)




    # drive_HMM(dhlp=dhlp, model_title="HMM2", n_states= 2, eval_start=eval_start,  eval_ar=eval_ar,
    #           eval_timestamps_ar=eval_timestams_ar)
    #
    # drive_HMM(dhlp=dhlp, model_title="HMM3", n_states=3, eval_start=eval_start, eval_ar=eval_ar,
    #           eval_timestamps_ar=eval_timestams_ar)
    #
    # drive_HMM(dhlp=dhlp, model_title="HMM4", n_states=4, eval_start=eval_start, eval_ar=eval_ar,
    #           eval_timestamps_ar=eval_timestams_ar)
    #
    # drive_HMM(dhlp=dhlp, model_title="HMM5", n_states=5, eval_start=eval_start, eval_ar=eval_ar,
    #           eval_timestamps_ar=eval_timestams_ar)
    pass

    # state_names2, pai2, A2, B2, state_sequence2 = dhlp.getMLE(n_states=2)
    # hidden_sequence = [state_sequence2[i] for i in range(eval_start,
    #                                                         eval_start+ EVALUATION_SEQUENCE_SIZE + TEST_SEQUENCE_SIZE)]
    # hmm2 = HMModel(name="HMM2", state_names=state_names2, pai=pai2, a=A2, b=B2)
    # hmm2.__str__()
    #
    # hmm2_drive = HMMpropagator(hmm=hmm2, observations = eval_ar, timestamps=eval_timestams_ar,
    #                            n_test = TEST_SEQUENCE_SIZE)
    # viterbi_path, aux_name, logits_ar = hmm2_drive.drive_tfd_model()
    #
    # pltViterbi(pref_file_name="HMM2", subtitle="starting at {}".format(dhlp.dt[eval_start]),
    #                 observations=eval_ar, viterbi_path=viterbi_path, hidden_sequence=hidden_sequence)
    #
    # logger.info("logits_ar\n{}".format(logits_ar))
    # logMarginPrb("HMM2", logits_ar, eval_timestams_ar)
    # logViterbiPath("HMM2", viterbi_path, eval_timestams_ar, hidden_sequence)
    #
    # state_names3, pai3, A3, B3, state_sequence3 = dhlp.getMLE(n_states=3)
    # hmm3 = HMModel(name="HMM3", state_names=state_names3, pai=pai3, a=A3, b=B3)
    # hmm3.__str__()
    #
    # state_names4, pai4, A4, B4, state_sequence4 = dhlp.getMLE(n_states=4)
    # hmm4 = HMModel(name="HMM4", state_names=state_names4, pai=pai4, a=A4, b=B4)
    # hmm4.__str__()
    #
    # state_names5, pai5, A5, B5, state_sequence5 = dhlp.getMLE(n_states=5)
    # hmm5 = HMModel(name="HMM5", state_names=state_names5, pai=pai5, a=A5, b=B5)
    # hmm5.__str__()
    pass