__author__ = 'clemens'


def viterbi(obs, states, start_p, trans_p, emit_p):
    v = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        v[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        v.append({})
        newpath = {}

        for y in states:
            (prob, state) = max((v[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            v[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath
    n = 0           # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t

    # return np.prod(v_array.shape), dd_mt_ident.complexity(), dd_ev_ident.complexity(), dd_mt_rmf3.complexity(), dd_ev_rmf3.complexity(), dd_ev_gf3.complexity(), dd_ev_gf3.complexity()
    return v


# Don't study this, it just prints a table of the steps.
def print_dptable(V):
    s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
    for y in V[0]:
        s += "%.5s: " % y
        s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
        s += "\n"
    print(s)

# states = ('Healthy', 'Fever')
#
# observations = ('normal', 'cold', 'dizzy')
#
# start_probability = {'Healthy': 0.6, 'Fever': 0.4}
#
# transition_probability = {
#    'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
#    'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
#    }
#
# emission_probability = {
#    'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
#    'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
#    }


def create_random_hmm(no_states, no_obs_states, sparsity=1.0, random_seed=0):
    import numpy as np
    np.random.seed(random_seed)
    states = np.array(['State '+str(i) for i in range(no_states)])
    observations = np.array(['Obs '+str(i) for i in range(no_obs_states)])
    start_probability_distribution = np.random.random((no_states,))
    start_probability_distribution /= np.sum(start_probability_distribution)
    start_probability = {}
    for i in range(len(states)):
        start_probability[states[i]] = start_probability_distribution[i]
    np.random.shuffle(states)
    np.random.shuffle(observations)
    transition_probability = {}
    emission_probability = {}
    for state in states:
        transition_probability[state] = {}
        emission_probability[state] = {}
        for state2 in states:
            transition_probability[state][state2] = 0
        for obs in observations:
            emission_probability[state][obs] = 0
    for state in range(len(states)):
        sum_of_transitions = 0
        number_of_transitions = 1
        encountered_goals = []
        next_state = state + 1 if state != len(states)-1 else 0
        while not (sum_of_transitions >= 0.95 or number_of_transitions >= sparsity*no_states):
            transition_goal = np.random.randint(no_states)
            if not (transition_goal == state+1 or transition_goal in encountered_goals):
                transition_probability[states[state]][states[transition_goal]] = np.random.uniform(low=0.0,
                                                                                           high=1.0-sum_of_transitions)
                sum_of_transitions += transition_probability[states[state]][states[transition_goal]]
                number_of_transitions += 1
                encountered_goals.append(transition_goal)
        transition_probability[states[state]][states[next_state]] = 1.0 - sum_of_transitions
        # the emissions
        sum_of_obs_goals = 0.0
        number_of_obs_transitions = 1
        encountered_observation_goals = []
        while not (sum_of_obs_goals >= 0.95 or number_of_obs_transitions >= sparsity*no_obs_states):
            transition_observation_goal = np.random.randint(no_obs_states)
            if not (transition_observation_goal == state or transition_observation_goal in encountered_observation_goals):
                emission_probability[states[state]][observations[transition_observation_goal]] = np.random.uniform(low=0.0,
                                                                                           high=1.0-sum_of_obs_goals)
                sum_of_obs_goals += emission_probability[states[state]][observations[transition_observation_goal]]
                number_of_obs_transitions += 1
                encountered_observation_goals.append(transition_observation_goal)
        emission_probability[states[state]][observations[np.mod(state, no_obs_states)]] = 1.0 - sum_of_obs_goals

    return observations, states, start_probability, transition_probability, emission_probability