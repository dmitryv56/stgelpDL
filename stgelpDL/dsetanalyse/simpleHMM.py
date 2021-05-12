#!/usr/bin/python3
import os
import sys

import pandas as pd
from pathlib import Path
import numpy as np
import copy
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions

from dsetanalyse.dsetAnalysis import D_STATES

class simpleHMM:

    def __init__(self, name: str = "HMM", states: list = [], num_steps: int = 0, f: object = None):
        pass
        self.name = name
        self.states = states
        self.num_states = len(states)
        self.num_steps = num_steps
        self.pai = None  # vector {pi}
        self.transitions = None  # matrix { aij}
        self.emission = None # matrix of pairs [mean,std]
        self.model = None
        self.posterior_marginals = None

    def setInit(self, train_states_seq: np.array = None):
        n = m = 0
        un, counts = np.unique(train_states_seq, return_counts=True)
        n, = train_states_seq.shape
        m, = counts.shape
        if m < self.num_states:  # missed states
            a = np.append(train_states_seq, [i for i in range(self.num_states)])
            un, counts = np.unique(a, return_counts=True)
            n, = a.shape
            m, = counts.shape
        del self.pai

        self.pai = np.array([round(counts[i] / n, 6) for i in range(m)]).astype("float32")
        return

    """ Estimate aij bya variant of maximum likelihood estimation
    Nominator: 'expected number of transitions from state i to state j'
    Denominator:'expected number of transitions from state i'
    """

    def setTransition(self, train_states_seq: np.array = None):
        n = m = 0
        a = copy.copy(train_states_seq)
        un, counts = np.unique(a, return_counts=True)
        n, = train_states_seq.shape
        m, = counts.shape
        if m < self.num_states:  # missed states
            a = np.append(a, [i for i in range(self.num_states)])
            un, counts = np.unique(a, return_counts=True)
            n, = a.shape
            m, = counts.shape

        del self.transitions

        self.transitions = np.zeros((m, m), dtype=float).astype("float32")
        for stateI in range(m):
            number_transitions_from_I = counts[un[stateI]]
            for stateJ in range(m):
                number_transitions_from_I_to_J = 0
                for i in range(n - 1):
                    if a[i] == stateI and a[i + 1] == stateJ:
                        number_transitions_from_I_to_J += 1
                self.transitions[stateI, stateJ] = round(number_transitions_from_I_to_J / number_transitions_from_I, 6)
        self.transitions=self.transitions.astype("float32")
        return

    """ Postulate a normal distribution for emission probabiliry Bj(Vk).

    """

    def setEmission(self, train_states_seq: np.array = None, train_observations: np.array = None):
        del self.emission
        self.emission = np.zeros((self.num_states,2), dtype=float)
        n, = train_states_seq.shape
        for i in range(self.num_states):
            self.emission[i,0]=0.0
            self.emission[i,1]=1e-06
        for i in range(self.num_states):

            part_sample = [train_observations[k] for k in range(n) if train_states_seq[k] == i]
            if len(part_sample) > 1:

                self.emission[i, 0] = np.array(part_sample).mean()
                self.emission[i, 1] = np.array(part_sample).std()
        self.emission=self.emission.astype("float32")
        return

    def createModel(self):
        # initial_distribution = tfd.Categorical(probs=[0.1, 0.7, 0.2])
        initial_distribution = tfd.Categorical(probs=self.pai.astype("float32"))

        # pp = np.array([[0.1, 0.6, 0.3],
        #                [0.6, 0.2, 0.2],
        #                [0, 0.2, 0.8]])
        # transition_distribution = tfd.Categorical(probs=pp)
        transition_distribution = tfd.Categorical(probs=self.transitions.astype("float32"))

        # We can model this with:
        list_loc = []
        list_scale = []
        for item in self.emission:  # item is a list [mean,std]
            list_loc.append(item[0])
            list_scale.append(item[1])
        # observation_distribution = tfd.Normal(loc=[-1., 0., 15.], scale=[2., 5., 10.])
        observation_distribution = tfd.Normal(loc=list_loc, scale=list_scale)

        # We can combine these distributions into a single week long
        # hidden Markov model with:
        del self.model
        self.model = tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=observation_distribution,
            num_steps=self.num_steps)

        return


    def fitModel(self, observations: np.array = None):

        if self.model is not None:
            observations=observations.astype("float32")
            log_posterior_marg = self.model.posterior_marginals(observations).logits
            a = np.array(log_posterior_marg)
            del self.posterior_marginals
            self.posterior_marginals = np.exp(a)
        return

    def viterbiPath(self, observations: np.array = None):
        observations = observations.astype("float32")
        posterior_mode = self.model.posterior_mode(observations)
        return posterior_mode

    def predict(self, n_periods: int = 1, in_sample_predict: int = 0) -> dict:
        predict_state = {}
        for k in range(in_sample_predict + 1, 0, -1):
            p = self.posterior_marginals[k-1, :]
            predict_state[k] = []
            for i in range(n_periods):
                pp = self.transitions.dot(p)
                state = np.argmax(pp)
                predict_state[k].append(state)
                p = copy.copy(pp)

        return predict_state


class poissonHMM(simpleHMM):

    def __init__(self,name: str = "HMM", states: list = [], lambdas:list=[],num_steps: int = 0, f: object = None):
        self.lambdas=lambdas
        super().__init__(name=name, states=states, num_steps=num_steps, f=f)

    def setEmission(self, train_states_seq: np.array = None, train_observations: np.array = None):
        del self.emission
        self.emission = np.array(self.lambdas)

        self.emission=self.emission.astype("float32")
        return

    def createModel(self):
        # initial_distribution = tfd.Categorical(probs=[0.1, 0.7, 0.2])
        initial_distribution = tfd.Categorical(probs=self.pai.astype("float32"))

        transition_distribution = tfd.Categorical(probs=self.transitions.astype("float32"))

        # We can model this with:

        observation_distribution = tfd.Poisson(rate=self.emission.tolist())

        # We can combine these distributions into a single week long
        # hidden Markov model with:
        del self.model
        self.model = tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=observation_distribution,
            num_steps=self.num_steps)

        return





def main(src_csv:str=None,repository:Path=None):
    data_col_names = ["Diesel_Power", "Real_demand", "WindGen_Power", "HydroTurbine_Power"]
    discret = 10

    test_size = 144


    dt_col_name = "Date Time"
    data_col_name = "Diesel_Power"
    state_col_name = "State_{}".format(data_col_name)
    # src_csv = "/home/dmitryv/LaLaguna/stgelpDL/dataLaLaguna/Data_ElHierro_2016_Diesel.csv"
    states=list(D_STATES.keys())
    df = pd.read_csv(src_csv)
    n=len(df)
    train_size=n-test_size

    with open("../dev/PowerGainDropMWtMin.log", 'w') as ff:
        for item in data_col_names:
            data_col_name=item
            state_col_name="State_{}".format(data_col_name)
            print("\n\n\n{}\n".format(item))
            train_observations = df[item].values[:-test_size] #-test_size]
            train_states       = df[state_col_name].values[:-test_size]#[:-test_size]
            d_states_new,train_states,states= convertStates(states=states, state_seq=train_states)
            print(d_states_new)
            states=list(d_states_new.keys())

            test_observations  = df[item].values[-test_size:]
            test_states        = df[state_col_name].values[-test_size:]

            shmm= simpleHMM(name = "{}_HMM".format(item), states=states, num_steps=train_size, f=ff)

            shmm.setInit (train_states_seq=train_states)
            shmm.setTransition(train_states_seq=train_states)
            shmm.setEmission(train_states_seq=train_states,train_observations=train_observations)

            shmm.createModel()
            shmm.fitModel(observations=train_observations)
            posterior_mode=shmm.viterbiPath(observations=train_observations)
            d_predict=shmm.predict(144)
            print("predict")
            print(d_predict)
            print("real test states")
            print(test_states)

def convertStates(states:list=None, state_seq:np.array=None)->(dict,np.array,list):
    un,c = np.unique(state_seq, return_counts=True)
    real_states,=un.shape
    number_states=len(states)
    d_states_new = {}
    if real_states==number_states:
        # no convert
        pass
        return d_states_new,state_seq, states

    ss=state_seq.tolist()
    for i in range(real_states):
        if un[i]==i:
            d_states_new[i]=D_STATES[i]
        else:
            d_states_new[i]=D_STATES[un[i]]
            for index, item in enumerate(ss):
                if ss[index]==un[i]:
                    ss[index]=i
    states=[i for i in range(real_states)]
    state_seq=np.array(ss)
    un1, c1 = np.unique(state_seq, return_counts=True)
    return d_states_new,state_seq, states






if __name__=="__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    repository = Path(dir_path) / Path("Repository")
    file_csv="updatedMwtMin.csv"
    src_csv=str(Path(repository)/Path(file_csv))

    main(src_csv=src_csv,repository=repository)

