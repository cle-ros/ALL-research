__author__ = 'clemens'

"""
This is the main test file for the decision diagram package.
"""


from pyDD.div.viterbi import *


def example():
    states = ('Healthy', 'Fever')
    observations = ('normal', 'cold', 'dizzy')
    start_probability = {'Healthy': 0.6, 'Fever': 0.4}
    transition_probability = {
       'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
       'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
       }
    emission_probability = {
       'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
       'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
       }

    return viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)


def test_hmm_generator():
    observations, states, start_probability, transition_probability, emission_probability = create_random_hmm(30, 30, 0.5, 0)
    # print create_random_hmm(6, 6, 0.5, 0)


def test_dds_on_hmms():
    import numpy as np
    result = np.array([[], []])
    for i in range(10,110,1):
        observations, states, start_probability, transition_probability, emission_probability = create_random_hmm(i, 16, 0.7, 1)
        # print observations
        # print states
        # print start_probability
        # print transition_probability
        # print emission_probability
        # mt_i, ev_i, mt_r, ev_r, mt_g, ev_g = viterbi(observations, states, start_probability, transition_probability, emission_probability)
        res = viterbi(observations, states, start_probability, transition_probability, emission_probability)
        result = np.hstack((result, np.array(res)[None].T))
    from pyDD.file_interaction.store_results import store_results
    store_results(result, 'aa2dd', 'div')


def some_more_kronecker_tests():
    import numpy as np
    from pyDD.basis import matrices, kronecker
    b = []
    r = []
    for i in range(1, 4, 1):
        b.append(kronecker.expansion(matrices.rmf3, i))
        r.append(np.sum(b[-1], axis=1))
    print r


if __name__ == "__main__":
    # some_more_kronecker_tests()
    test_dds_on_hmms()
    # test_hmm_generator()
    # print(example())
