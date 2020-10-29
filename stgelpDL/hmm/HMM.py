#!/usr/bin/python
""" This module implements the wrapper for HiddenMarkovModel class

"""

import copy
from pickle import dump, load

import numpy as np
import scipy.stats

DISCRETE_OBS = 0
CONTINOUS_OBS = 1

probNormDist = lambda val, mean, std, prob_decision: round(scipy.stats.norm.cdf(val, mean, std), prob_decision)


class hmm():

    def __init__(self, num_states, num_obs, obs_type, states, f=None):
        self.num_states = num_states
        self.num_obs = num_obs
        self.obs_type = obs_type
        self.states = np.array(states)
        self.pai = np.zeros((self.num_states), dtype=float)
        self.transitDist = np.zeros((self.num_states, self.num_states), dtype=float)
        self.observations = None  # for discrete observations
        self.emisDist = None
        self.B = np.zeros((self.num_states, self.num_obs), dtype=float)
        self.d_pai = {}
        self.d_transitDist = {}
        self.d_emisDist = {}
        self.model = {}

        self.forwardprob = 0.0

        return

    def getModel(self):
        return (self.pai, self.transitDist, self.emisDist)

    def dumpModel(self, path2file=None):
        self.model['obs_type'] = self.obs_type
        self.model['num_states'] = self.num_states
        self.model['num_obs'] = self.num_obs
        self.model['states'] = self.states
        self.model['d_pai'] = self.d_pai
        self.model['d_transitDist'] = self.d_transitDist
        self.model['d_emisDist'] = self.d_emisDist
        self.model['observations'] = self.observations

        with open(path2file, 'wb') as fp:
            dump(self.model, fp)

        return

    def loadModel(self, path2file):
        del self.model

        with open(path2file, 'rb') as fp:
            self.model = load(fp)
            self.obs_type = self.model["obs_type"]
            self.num_states = self.model["num_states"]
            self.num_obs = self.model["num_obs"]
            self.states = copy.copy(self.model["states"])
            self.d_pai = copy.copy(self.model["d_pai"])
            self.d_transitDist = copy.copy(self.model["d_transitDist"])
            self.d_emisDist = copy.copy(self.model["d_emisDist"])
            self.observations = copy.copy(self.model["observations"])
            self.dict2initDist()
            self.dict2transitDist()
            self.dict2emisDist()
        return

    def setObs(self, obs):

        self.observations = np.array(obs)

        self.num_obs = self.observations.shape[0]
        return

    def setTransitDist(self, transitDist):
        self.transitDist = copy.copy(transitDist)
        for i in range(len(self.states)):
            d = {}
            for j in range(len(self.states)):
                d[self.states[j]] = self.transitDist[i][j]
            self.d_transitDist[self.states[i]] = copy.copy(d)
            del d
        return

    def setInitialDist(self, initDist):
        self.pai = copy.copy(initDist)
        self.initDist2dict()
        return

    def initDist2dict(self):
        for i in range(len(self.pai)):
            self.d_pai[self.states[i]] = self.pai[i]
        return

    def dict2initDist(self):
        self.pai = np.zeros(self.num_states, dtype=float)
        for key, val in self.d_pai.items():
            self.pai[int(key)] = val
        return

    def transitDist2dict(self):
        for i in range(len(self.states)):
            d = {}
            for j in range(len(self.states)):
                d[self.states[j]] = self.transitDist[i][j]
            self.d_transitDist[self.states[i]] = copy.copy(d)
            del d
        return

    def dict2transitDist(self):
        self.transitDist = np.zeros((self.num_states, self.num_states), dtype=float)
        for i, d in self.d_transitDist.items():
            for j, val in d.items():
                self.transitDist[int(i)][int(j)] = val
        return

    def setEmisDist(self, emisDist):
        self.emisDist = copy.copy(emisDist)
        self.emisDist2dict()

    def fillObsProbMatrix(self):
        self.bObs = copy.copy(self.emisDist)

    """ Forward, backward, viterbi and etc"""

    def forward(self) -> tuple(float, np.array):
        pass
        self.alfa = np.zeros((self.num_states, self.num_obs), dtype=float)

        # initialization recursion step
        for j in range(self.num_states):
            self.alfa[j][0] = self.pai[j] * self.bObs[j][0]
        # recursion step
        for t in range(1, self.bObs.shape[1]):
            for s in range(self.transitDist.shape[0]):
                part_sum = 0.0
                for s1 in range(self.transitDist.shape[0]):
                    part_sum += self.alfa[s1][t - 1] * self.transitDist[s1][s]
                self.alfa[s][t] += part_sum * self.bObs[s][t]

        self.forwardprob = 0.0
        self.forwardprob += sum(self.alfa[s][self.alfa.shape[1] - 1] for s in (range(self.alfa.shape[0])))

        return (self.forwardprob, self.alfa)

    def viterbi(self) -> tuple(float, np.array, np.array):

        vi = np.zeros((self.states.shape[0], self.bObs.shape[1]))
        bt = np.zeros((self.states.shape[0], self.bObs.shape[1]), dtype=int)

        # initialization step
        for s in range(self.states.shape[0]):
            vi[s][0] += self.pai[s] * self.bObs[s][0]
            bt[s][0] = 0
        # recursion step
        for t in range(1, self.bObs.shape[1]):
            for j in range(self.states.shape[0]):
                part_item = 0.0
                for i in range(self.states.shape[0]):
                    part_item = vi[i][t - 1] * self.transitDist[i][j] * self.bObs[j][t]
                    if (part_item > vi[j][t]):
                        vi[j][t] = part_item
                        bt[j][t] = i
        # termination state
        bestPathProb = 0.0
        bestpathptr = 0
        for i in range(self.states.shape[0]):
            if vi[i][self.bObs.shape[1] - 1] > bestPathProb:
                bestPathProb = vi[i][self.bObs.shape[1] - 1]
                bestpathptr = i
        # reverse
        viterbi_list = []
        viterbi_list.append(bestpathptr)
        for t in range(vi.shape[1] - 1, 0, -1):
            reverseptr = 0
            bestProb = vi[0][t]
            for i in range(1, self.states.shape[0]):
                if vi[i][t] > bestProb:
                    reverseptr = i
            viterbi_list.insert(0, reverseptr)
        viterbi_path = np.array(viterbi_list)
        return (bestPathProb, vi, viterbi_path)


""" This class implements an emission distribution matrix for discrete observations.
There is an finite set of possible observations O={o1,o2,...,oT]
"""


class hmm_dobs(hmm):

    def __init__(self, num_states, num_obs, states, f=None):
        super().__init__(num_states, num_obs, DISCRETE_OBS, states, f=None)

    def emisDist2dict(self):
        for j in range(self.states.shape[0]):
            d = {}
            for t in range(self.observations.shape[0]):
                d[self.states[j]] = self.emisDist[j][t]
            self.d_emisDist[self.states[j]] = copy.copy(d)
            del d

    def dict2emisDist(self):
        self.emisDist = np.zeros((self.states.shape[0], self.observations.shape[0]), dtype=float)
        for i, d in self.d_emisDist.items():
            for j, val in d.items():
                self.emisDist[int(i)][int(j)] = val
        return


""" This class implements the emission distribution matrix for continuouos observations.
The observations for each state are described by some parameterized probability distribution
"""


class hmm_cobs(hmm):

    def __init__(self, num_states, num_obs, states, f=None):
        super().__init__(num_states, num_obs, CONTINOUS_OBS, states, f=None)


""" This class implements the emission distribution matrix for continuouos observations.
The observations for each state are described by normal (Gauss) probability distribution.
The loc (mean) and scale(standard derivation) are defined normal distribution. 
"""


class hmm_gmix(hmm_cobs):
    _num_dist_params = 2
    _ind_loc_param = 0
    _ind_scale_param = 1

    def __init__(self, num_states, states, f=None):
        """

        :param num_states:
        :param states:
        :param f:
        """
        super().__init__(num_states, 0, states, f=None)
        return

    # def setEmisDist(self,emisDist):
    #     self.emisDist=copy.copy(emisDist)
    #     self.emisDist2dict()
    #
    #     return

    def emisDist2dict(self):
        for i in range(self.num_states):
            d = {}
            d['loc'] = self.emisDist[i][hmm_gmix._ind_loc_param]
            d['scale'] = self.emisDist[i][hmm_gmix._ind_scale_param]
            self.d_emisDist[self.states[i]] = copy.copy(d)
            del d

    def dict2emisDist(self):
        self.emisDist = np.zeros((self.num_states, hmm_gmix._num_dist_params), dtype=float)
        for i, val in self.d_emisDist.items():
            self.emisDist[int(i)][hmm_gmix._ind_loc_param] = val['loc']
            self.emisDist[int(i)][hmm_gmix._ind_scale_param] = val['scale']

    @staticmethod
    def getNumGaussParams():
        return hmm_gmix._num_dist_params

    @staticmethod
    def getIndexLocParam():
        return hmm_gmix._ind_loc_param

    @staticmethod
    def getIndexScaleParam():
        return hmm_gmix._ind_scale_param

    def fillObsProbMatrix(self):
        del self.bObs
        self.bObs = np.zeros((self.states.shape[0], self.observations.shape[0]))
        round_decision = 6
        for s in range(self.states.shape[0]):
            for t in range(self.observations.shape[0]):
                self.bObs[s][t] = probNormDist(self.observations[t], self.emisDist[s][hmm_gmix._ind_loc_param],
                                               self.emisDist[s][hmm_gmix._ind_scale_param], round_decision)
        return


""" test for discrete emision Distributions
A hidden Markov model for relating numbers of ice creams eaten by Jason (the
observations) to the weather (H or C, the hidden variables) - D. Jurascky & J.M. Martin Speech and 
Language Processing, 2019
HMM:
Two states Cold -0 , Hot -1
P(C->C) a[0,0] = 0.5
P(C->H) a{1,0] = 0.5
P(H->H) a(1,1) = 0.6 
P(H->C) a[1,0] = 0.4
P(3|c)=0.1 P(1|C)=
"""


def test_discreteEmission():
    pass
    states = np.array([0, 1])
    #
    observations = np.array([3.0, 1.0, 3.0])

    pai = np.array([0.2, 0.8])
    transitDist = np.array([[0.5, 0.5], [0.4, 0.6]])
    emisDist = np.array([[0.1, 0.5, 0.1], [0.4, 0.2, 0.4]])

    dhmm = hmm_dobs(2, 3, states)
    dhmm.setObs(observations)
    dhmm.setInitialDist(pai)
    dhmm.setEmisDist(emisDist)
    dhmm.setTransitDist(transitDist)
    dhmm.fillObsProbMatrix()
    prob, alfa = dhmm.forward()

    dhmm.viterbi()

    s1 = dhmm.dumpModel("a.json")
    dhmm.loadModel(s1, "a.json")


def test_continuousEmission():
    pai = np.array([0.4, 0.5, 0.1])
    transitDist = np.array([[0.1, 0.7, 0.2], [0.5, 0.1, 0.4], [0.1, 0.8, 0.1]])
    emisDist = np.array([[-1.0, 2.0], [0.0, 1.0], [1.0, 2.0]])

    ahmm = hmm_gmix(3, [0, 1, 2])

    ahmm.setInitialDist(pai)
    ahmm.setEmisDist(emisDist)
    ahmm.setTransitDist(transitDist)
    ahmm.setObs([1.0, -5.0, 0.1, 0.9, -0.09, 4.0, 6.0, 2.5])
    ahmm.fillObsProbMatrix()
    prob, alfa = ahmm.forward()

    s1 = ahmm.dumpModel("a.json")
    ahmm.loadModel(s1, "a.json")


if __name__ == "__main__":
    test_discreteEmission()

    pass
