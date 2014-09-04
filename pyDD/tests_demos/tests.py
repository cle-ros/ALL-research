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


def test_viterbi():
    # print_dptable(v)
    from pyDD.diagram.diagram import AAxEVDD
    # from pyDD.basis import matrices as exp_bases
    import numpy as np
    result = []
    diagram_ev = AAxEVDD(2)
    for k in range(2, 10, 1):
        diagram_ev.base = k
        tmp_result = []
        for i in range(1000, 1000000, 50000):
        # for i in range(10, 12, 1):
            observations, states, start_probability, transition_probability, emission_probability = create_random_hmm(i, 10, 0.8, 1)
            v = viterbi(observations, states, start_probability, transition_probability, emission_probability)
            # diagram_ev = tern.AAEV3DD()
            # diagram_mt = tern.MT3DD()
            v_array = []
            for element in v:
                tmp = []
                tmp += element.values()
                v_array.append(tmp)
            v_array = np.array(v_array)
            dd_ev_ident = diagram_ev.create(v_array, 0)
            tmp_result.append(dd_ev_ident.complexity())
        result.append(tmp_result)

    from pyDD.file_interaction.store_results import store_results
    store_results(result, 'aa2dd', 'div')
    # dd_mt_ident = diagram_mt.create(v_array, 0)
    # dd_ev_rmf3  = diagram_ev.create(v_array, 0, kron_exp=exp_bases.rmf3)
    # dd_mt_rmf3  = diagram_mt.create(v_array, 0, kron_exp=exp_bases.rmf3)
    # dd_ev_gf3   = diagram_ev.create(v_array, 0, kron_exp=exp_bases.gf3)
    # dd_mt_gf3   = diagram_mt.create(v_array, 0, kron_exp=exp_bases.gf3)
    # print 'Complexity matrix:       ' + str(np.prod(v_array.shape))
    # print 'Complexity dd mt ident:  ' + str(dd_mt_ident.complexity())
    # print 'Complexity dd ev ident:  ' + str(dd_ev_ident.complexity())
    # print 'Complexity dd mt rmf3:   ' + str(dd_mt_rmf3.complexity())
    # print 'Complexity dd ev rmf3:   ' + str(dd_ev_rmf3.complexity())


def test_basis_power():
    import numpy as np
    result = np.array([[], []])

                # print observations
                # print states
                # print start_probability
                # print transition_probability
                # print emission_probability
                # mt_i, ev_i, mt_r, ev_r, mt_g, ev_g = viterbi(observations, states, start_probability, transition_probability, emission_probability)




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
    test_viterbi()
    # some_more_kronecker_tests()
    # test_dds_on_hmms()
    # test_hmm_generator()
    # print(example())
