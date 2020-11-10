# 1/usr/bin/python

import scipy.stats

scipy.stats.norm(loc=100, scale=12)
# where loc is the mean and scale is the std dev
# if you wish to pull out a random number from your distribution
print(scipy.stats.norm.rvs(loc=100, scale=12))

# To find the probability that the variable has a value LESS than or equal
# let's say 113, you'd use CDF cumulative Density Function
print(scipy.stats.norm.cdf(113, 100, 12))
# Output: 0.86066975255037792
# or 86.07% probability

# To find the probability that the variable has a value GREATER than or
# equal to let's say 125, you'd use SF Survival Function
print(scipy.stats.norm.sf(125, 100, 12))
# Output: 0.018610425189886332
# or 1.86%

# To find the variate for which the probability is given, let's say the
# value which needed to provide a 98% probability, you'd use the
# PPF Percent Point Function
print(scipy.stats.norm.ppf(.98, 100, 12))
# Output: 124.64498692758187
pass
print(round(scipy.stats.norm.cdf(-0.2, 0.0, 0.05 / 3), 3))
print(round(scipy.stats.norm.cdf(0.0, 0.0, 0.05 / 3), 3))
print(round(scipy.stats.norm.cdf(0.4, 0.0, 0.05 / 3), 3))
if False:
    states = ('Healthy', 'Fever')
    end_state = 'E'

    observations = ('normal', 'cold', 'dizzy')

    start_probability = {'Healthy': 0.6, 'Fever': 0.4}

    transition_probability = {
        'Healthy': {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
        'Fever': {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
    }

    emission_probability = {
        'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
        'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
    }
"""              
Emission Distribution
  -0.19700000000000004  0.12436237373096415
  0.0  0.0
  0.22069693769799367  0.13989866607629026

Transitions Distribution
  0.53625  0.215  0.24875
  0.34243697478991597  0.26260504201680673  0.3949579831932773
  0.21987315010570824  0.18921775898520085  0.5909090909090909  

Initial Distribution
  0.0
  0.07692307692307693
  0.9230769230769231

"""

states = ('-', '0', '+')
end_state = '+'

observations = ('16:00', '16:10', '16:20', '16:30', '16:40', '16:50', '17:00', '17:10')
start_probability = {'-': 0.0, '0': 0.069, '+': 0.9231}
start_probability = {'-': 0.33, '0': 0.33, '+': 1 - 0.66}

transition_probability = {
    '-': {'-': 0.5362, '0': 0.215, '+': 0.24875},
    '0': {'-': 0.34243697478991597, '0': 0.26260504201680673, '+': 0.3949579831932773},
    '+': {'-': 0.21987315010570824, '0': 0.18921775898520085, '+': 0.5909090909090909},

}
probNormDist = lambda val, mean, std: round(scipy.stats.norm.cdf(val, mean, std), 4)

emission_probability = {
    '-': {'16:00': probNormDist(-0.2, -0.197, 0.1244),
          '16:10': probNormDist(-0.2, -0.197, 0.1244),
          '16:20': probNormDist(0.0, -0.197, 0.1244),
          '16:30': probNormDist(-0.2, -0.197, 0.1244),
          '16:40': probNormDist(-0.1, -0.197, 0.1244),
          '16:50': probNormDist(0.0, -0.197, 0.1244),
          '17:00': probNormDist(0.0, -0.197, 0.1244),
          '17:10': probNormDist(0.4, -0.197, 0.1244)
          },
    '0': {'16:00': probNormDist(-0.2, 0.0, 0.05 / 3),
          '16:10': probNormDist(-0.2, 0.0, 0.05 / 3),
          '16:20': probNormDist(0.0, 0.0, 0.05 / 3),
          '16:30': probNormDist(-0.2, 0.0, 0.05 / 3),
          '16:40': probNormDist(-0.1, 0.0, 0.05 / 3),
          '16:50': probNormDist(0.0, 0.0, 0.05 / 3),
          '17:00': probNormDist(0.0, 0.0, 0.05 / 3),
          '17:10': probNormDist(0.4, 0.0, 0.05 / 3),
          },
    '+': {'16:00': probNormDist(-0.2, 0.221, 0.1399),
          '16:10': probNormDist(-0.2, 0.221, 0.1399),
          '16:20': probNormDist(0.0, 0.221, 0.1399),
          '16:30': probNormDist(-0.2, 0.221, 0.1399),
          '16:40': probNormDist(-0.1, 0.221, 0.1399),
          '16:50': probNormDist(0.0, 0.221, 0.1399),
          '17:00': probNormDist(0.0, 0.221, 0.1399),
          '17:10': probNormDist(0.4, 0.221, 0.1399)
          }
}


def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st, f=None):
    """Forward–backward algorithm."""
    # Forward part of the algorithm
    observations = tuple(observations)
    fwd = []
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k] * trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)

    # Backward part of the algorithm
    bkw = []
    for i, observation_i_plus in enumerate(reversed(observations[1:] + (None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)

        bkw.insert(0, b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)
    message = f"""
              Forward path : {p_fwd}
              Backward_path:  {p_bkw}
    """
    print(message)
    if f is not None:
        f.write(message)
    # Merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    # assert p_fwd == p_bkw
    return fwd, bkw, posterior


def example():
    return fwd_bkw(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability,
                   end_state)


def example1():
    return viterbi(list(observations),
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)


def viterbi(obs, states, start_p, trans_p, emit_p, f=None):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)
    return opt, max_prob, V


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


if __name__ == '__main__':
    pass
    for line in example():
        print(*line)
    pass
    print('======================================')
    example1()
    pass