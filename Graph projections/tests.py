# -*- coding: utf-8 -*-
"""
@author: Clemens Rosenbaum (cgbr@cs.umass.edu)
"""

import numpy as np
import config
import pylab as pl



def test_multiplication(number, max_size, sparsity):
    """
    This function just computes many, many diagram multiplications and checks them
    against the "correct" value.
    :param number:
    :param max_size:
    :param sparsity:
    :return:
    """
    misses = []
    from diagram_initialization import initialize_diagram, expand_matrix_exponential
    from diagram_computations import multiply_diagram

    for i in range(number):
        # creating the sizes of the matrices
        size_first = [np.random.random_integers(2, max_size, 1)[0], np.random.random_integers(2, max_size, 1)[0]]
        size_second = [size_first[1], np.random.random_integers(2, max_size, 1)[0]]
        # creating the random matrices
        mat1 = np.random.rand(size_first[0], size_first[1]) * 2 - 1
        mat2 = np.random.rand(size_second[0], size_second[1]) * 2 - 1
        # adding sparsity
        spars1 = np.round(np.random.rand(size_first[0], size_first[1]) * 0.5 / sparsity)
        spars2 = np.round(np.random.rand(size_second[0], size_second[1]) * 0.5 / sparsity)
        mat1 = mat1 * spars1
        mat2 = mat2 * spars2
        # creating the diagrams
        diag1 = BNode('x')
        diag2 = BNode('y')
        initialize_diagram(diag1, mat1, 0)
        initialize_diagram(diag2, mat2, 0)
        # computing the multiplication
        result = multiply_diagram(diag1, diag2, mat1.shape[1])
        result_mat = result.to_matrix(mat1.shape[0], False)
        if not (result_mat == [0]).all():
            reference_result = expand_matrix_exponential(np.dot(mat1, mat2), np.log2(result_mat.shape), 0, 2)
            diff = np.max(np.abs(reference_result - result_mat))
            if not diff < 1e-10:
                import code;

                code.interact(local=dict(locals().items() + globals().items()))
                problematic_mats = [mat1, mat2]
                misses.append(problematic_mats)
        else:
            if np.max(np.abs(np.dot(mat1, mat2))) > 0:
                problematic_mats = [mat1, mat2]
                misses.append([problematic_mats, [result_mat, reference_result]])
    return misses


def run_tests():
    import numpy as np
    from diagram_initialization import initialize_diagram
    from diagram_computations import transpose_diagram, multiply_diagram
    #mat1 = np.random.random_integers(0,5,[3,3])
    #mat2 = np.random.random_integers(-5,0,[3,3])
    mat1 = np.array([[1, 2, 0], [0, 3, 0], [0, 4, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 5, 0]])
    mat2 = np.array([[1, -2, 0], [2, -3, 0], [3, -4, -1]])
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([2.0, 3.0, 4.0])
    diag1 = BNode('x')
    diag2 = BNode('y')
    vecDiag1 = BNode('z')
    vecDiag2 = BNode('a')
    initialize_diagram(vecDiag1, vec1, 0)
    initialize_diagram(vecDiag2, vec2, 0)
    initialize_diagram(diag1, mat1, 0)
    initialize_diagram(diag2, mat2, 0)
    #  print 'buh'
    diag3 = multiply_diagram(diag1, diag2, 3)
    # diag3 = elementwise_multiply_diagrams(diag1, diag2)
    print('lala')
    print diag1.to_matrix(3, False)
    print transpose_diagram(diag2, 3).to_matrix(3, False)
    print('correct result')
    print np.dot(mat1, mat2)
    print (diag3.to_matrix(7, True))
    print np.max(np.abs(np.dot(mat1, mat2) - diag3.to_matrix(7, True)))
    print('hi')
    print(diag3.d)
    print diag3.to_matrix()
    import code;

    code.interact(local=dict(locals().items() + globals().items()))
    a = BNode('hallo', 'x1')
    b = BNode('hallo1', 'x2', p=a)
    c = BNode('hallo1', 'x2', p=b)
    d = BNode('hallo1', 'x2', n=b)


def run_tests2():
    a = np.array([[0.5, 0, 0, 0], [0, 0, 0, 0]])
    diag1 = BNode('x')
    from diagram_initialization import initialize_diagram

    initialize_diagram(diag1, a, 0)
    print diag1.to_matrix(2, False)


def test_new_implementation():
    from diagram import MTBDD

    mat1 = np.array([[1, 2, 0], [0, 3, 0], [0, 4, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 5, 0]])
    mtbdd = MTBDD()
    node = mtbdd.create(mat1, 0)
    print node.to_matrix(7)
    print node.get_subdiagrams(5)


def test_reduction():
    from diagram_binary import MTBDD, EVBDD

    mat1 = np.array([[1, 2, 0], [0, 1, 0], [2, 4, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 5, 0]], dtype=float)
    print mat1
    print('MTBDD:')
    mtbdd = MTBDD()
    node = mtbdd.create(mat1, 0)
    # import code; code.interact(local=dict(locals().items() + globals().items()))
    # print node.to_matrix(7)
    # print node.leaves
    node1 = mtbdd.create(mat1, 0)
    reduced = mtbdd.reduce(node1)
    # import code; code.interact(local=dict(locals().items() + globals().items()))
    red = reduced.leaves
    print node.to_matrix(7)
    print len(node.leaves)
    print reduced.to_matrix(7)
    print len(red)
    print('EVBDD:')
    mtbdd = EVBDD()
    node = mtbdd.create(mat1, 0)
    node1 = mtbdd.create(mat1, 0)
    reduced = mtbdd.reduce(node1)
    red = reduced.leaves
    print node.to_matrix(7)
    print len(node.leaves)
    print reduced.to_matrix(7)
    print len(red)


def test_addition():
    from diagram import MTBDD, EVBDD

    mat1 = np.array([[1, 7, 0], [0, -1, 0], [2, 8, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 15, 0]], dtype=float)
    print mat1 + mat1
    print('MTBDD:')
    mtbdd = MTBDD()
    node1 = mtbdd.create(mat1, 0)
    node2 = mtbdd.create(mat1, 0)
    from diagram_computations import add_diagrams

    node3 = add_diagrams(node1, node2)
    print node3.to_matrix(7, True)
    print('EVBDD:')
    evbdd = EVBDD()
    node4 = evbdd.create(mat1, 0)
    node5 = evbdd.create(mat1, 0)
    from diagram_computations import diagram_sum

    node6 = add_diagrams(node4, node5)
    print node6.to_matrix(7, True)
    print('summing the diagrams:')
    print mat1
    print diagram_sum(node1)
    print diagram_sum(node4)
    print np.sum(mat1)
    print node4.nodes
    node6.plot('ev-add')
    node3.plot('mt-add')


def test_multiplication():
    from diagram import MTBDD, EVBDD

    mat1 = np.array([[1, 7, 0], [0, -1, 0], [2, 8, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 15, 0]], dtype=float)
    print mat1
    print 'Scalar Multiplication: ~~~~~~~~~~~~~~'
    print mat1 * 2.5
    print('MTBDD:')
    from diagram_computations import scalar_multiply_diagram, elementwise_multiply_diagrams_evbdd, \
        elementwise_multiply_diagrams_mtbdd, multiply_by_column_vector, multiply_diagram

    mtbdd = MTBDD()
    node1 = mtbdd.create(mat1, 0)
    node2 = scalar_multiply_diagram(node1, 2.5)
    print node2.to_matrix(7, True)
    print('EVBDD:')
    evbdd = EVBDD()
    node4 = evbdd.create(mat1, 0)
    node5 = scalar_multiply_diagram(node4, 2.5)
    print node5.to_matrix(7, True)
    print 'Elementwise Multiplication: ~~~~~~~~~'
    print np.multiply(mat1, mat1)
    print('MTBDD:')
    node3 = elementwise_multiply_diagrams_mtbdd(node1, node1)
    # node3.plot('mtbdd')
    print node3.to_matrix(7, True)
    print('EVBDD:')
    node6 = elementwise_multiply_diagrams_evbdd(node4, node4)
    print node6.to_matrix(7, True)
    # node6.plot('evbdd')
    vec1 = np.array([1.0, 2.0, 3.0])
    vnode_mt = mtbdd.create(vec1, 0)
    vnode_ev = evbdd.create(vec1, 0)
    print('Column-vector multiplication')
    print('Reference')
    print(np.dot(mat1, vec1))
    print('MTBDD')
    node7 = multiply_by_column_vector(node1, vnode_mt)
    print(node7.to_matrix(7, True))
    print('EVBDD')
    node8 = multiply_by_column_vector(node4, vnode_ev)
    print(node8.to_matrix(7, True))
    print('Matrix-multiplication')
    mat2 = np.array([[1, -2, 0], [2, -3, 0], [3, -4, -1]])
    print('Reference')
    print(np.dot(mat1, mat2))
    print('MTBDD')
    node11 = mtbdd.create(mat2, 0)
    node9 = multiply_diagram(node1, node11, mat1.shape[0])
    print(node9.to_matrix(3, True))
    print('EVBDD')
    node12 = evbdd.create(mat2, 0)
    node10 = multiply_diagram(node4, node12, mat1.shape[0])
    print(node10.to_matrix(7, True))


def general_tests():
    from diagram import MTBDD, EVBDD

    mtbdd = MTBDD()
    evbdd = EVBDD()
    mat1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    node1 = mtbdd.create(mat1, 0)
    node4 = evbdd.create(mat1, 0)
    node1.plot('mt-st-r')
    node4.plot('ev-st-r')
    node1 = mtbdd.create(mat1, 0, False)
    node4 = evbdd.create(mat1, 0, False)
    node1.plot('mt-st-nr')
    node4.plot('ev-st-nr')


def store_results(test_type, diagram_type, episode, value):
    # print('storing '+test_type + ' with '+diagram_type+' in episode '+str(episode)+' and the value '+str(value))
    if episode == -1 and config.unknown_episode:
        config.last_episode += 1
        iepisode = config.last_episode
        config.unknown_episode = False
    else:
        iepisode = config.last_episode
    try:
        config.glob_result[test_type][diagram_type][iepisode] = np.hstack((config.glob_result[test_type][diagram_type][iepisode], np.array(value)))
    except KeyError:
        config.glob_result[test_type][diagram_type][iepisode] = np.array([])
        config.glob_result[test_type][diagram_type][iepisode] = np.hstack((config.glob_result[test_type][diagram_type][iepisode], np.array(value)))


def test_power_method(iterations, episodes, max_size):
    for ep in range(episodes):
        # initialization
        vec_size = np.random.random_integers(10, high=max_size)
        vec = np.random.rand(vec_size, 1)
        mat = np.random.rand(vec_size, vec_size)
        for it in range(iterations):
            # multiplying
            vec = np.dot(mat, vec)
            # normalizing
            vec /= np.linalg.norm(vec)
            # creating the BDD representations
            from diagram import MTBDD, EVBDD
            mtbdd = MTBDD()
            evbdd = EVBDD()
            node_mt = mtbdd.create(np.round(config.precision_elements/np.max(vec)*vec, config.precision_round), 0)
            node_ev = evbdd.create(np.round(config.precision_elements/np.max(vec)*vec, config.precision_round), 0)
            c_mt = node_mt.complexity()
            c_ev = node_ev.complexity()
            # storing the complexity
            store_results('power', 'mat', ep, vec_size)
            store_results('power', 'mtbdd', ep, c_mt)
            store_results('power', 'evbdd', ep, c_ev)


def test_power_method_sparse(iterations, episodes, max_size, sparsity):
    for ep in range(episodes):
        # initialization
        vec_size = np.random.random_integers(10, high=max_size)
        vec = np.random.rand(vec_size, 1)
        mat = np.multiply(np.random.rand(vec_size, vec_size), np.random.rand(vec_size, vec_size) > sparsity) + \
              np.random.rand(vec_size, vec_size) * np.identity(vec_size)
        for it in range(iterations):
            # multiplying
            vec = np.dot(mat, vec)
            # normalizing
            vec /= np.linalg.norm(vec)
            # creating the BDD representations
            from diagram import MTBDD, EVBDD
            mtbdd = MTBDD()
            evbdd = EVBDD()
            node_mt = mtbdd.create(np.round(config.precision_elements/np.max(vec)*vec, config.precision_round), 0)
            node_ev = evbdd.create(np.round(config.precision_elements/np.max(vec)*vec, config.precision_round), 0)
            c_mt = node_mt.complexity()
            c_ev = node_ev.complexity()
            # storing the complexity
            store_results('power-sparse', 'mat', ep, vec_size)
            store_results('power-sparse', 'mtbdd', ep, c_mt)
            store_results('power-sparse', 'evbdd', ep, c_ev)


def apply_to_zeros(lst, dtype=np.int64):
    inner_max_len = max(map(len, lst))
    result = np.zeros([len(lst), inner_max_len], dtype)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result


def test_performance():
    config.precision_round = 1
    config.precision_elements = 1
    iterations = 25
    episodes = 25
    max_size = 50
    sparsity = 0.8
    # the power method
    test_power_method(iterations, episodes, max_size)
    test_power_method_sparse(iterations, episodes, max_size, sparsity)
    # the directed rooms
    import rl.demos.fourier_mcar as mcar
    import rl.demos.directed_two_rooms as two_rooms
    for ep in range(episodes):
        two_rooms.main(30, 0.1, 0.99, iterations=iterations)
        config.unknown_episode = True
    config.last_episode = 0
    # the mountain car
    for ep in range(episodes):
        mcar.main(num_episodes=1, max_iters=iterations*10)
        config.unknown_episode = True
    # storing the result
    from store_results import store_results
    # import code; code.interact(local=dict(locals().items() + globals().items()))
    for test_key in config.glob_result.keys():
        print test_key
        result_mat = []
        result_mt = []
        result_ev = []
        print config.glob_result[test_key]['mat'].keys()
        for key in config.glob_result[test_key]['mat'].keys():
            result_mat.append(config.glob_result[test_key]['mat'][key])
            result_mt.append(config.glob_result[test_key]['mtbdd'][key])
            result_ev.append(config.glob_result[test_key]['evbdd'][key])
        # if test_key == 'mcar':
        #     import code; code.interact(local=dict(locals().items() + globals().items()))
        # print test_key
        # print result_mat
        # print result_mt
        # print result_ev
        result_mat = apply_to_zeros(result_mat)
        result_mt = apply_to_zeros(result_mt)
        result_ev = apply_to_zeros(result_ev)
        result_mat = np.sum(np.array(result_mat), axis=0)
        result_mt = np.sum(np.array(result_mt), axis=0)
        result_ev = np.sum(np.array(result_ev), axis=0)
        store_results(result_mat, test_key, 'mat')
        store_results(result_mt, test_key, 'mt')
        store_results(result_ev, test_key, 'ev')


def plot_graphs():
    # mat1 = np.array([[1, 4], [1, 4], [1, 4], [1, 4]])
    mat1 = np.random.random((4, 4))
    from diagram_binary import MTBDD, EVBDD
    mtbdd = MTBDD()
    evbdd = EVBDD()
    node1 = mtbdd.create(mat1, 0)
    node4 = evbdd.create(mat1, 0)
    # node1.plot('mt-st-r')
    node4.plot('random')

    # mat1 = mat1.T
    # node1 = mtbdd.create(mat1, 0, False)
    # node4 = evbdd.create(mat1, 0, True)
    # # node1.plot('mt-st-nr')
    # node4.plot('vo-ev-st-o2')


def plot_results():
    # filename1 = '04-29_19-16-38_power-sparse_mat'
    # filename2 = '04-29_19-16-38_power-sparse_mt'
    # filename3 = '04-29_19-16-38_power-sparse_ev'
    # filename1 = '04-29_20-57-12_mcar_mat'
    # filename2 = '04-29_20-57-12_mcar_mt'
    # filename3 = '04-29_20-57-12_mcar_ev'
    # title = 'Q-Learning (mountain car), rounded'
    filename1 = '04-29_20-57-12_2room_mat'
    filename2 = '04-29_20-57-12_2room_mt'
    filename3 = '04-29_20-57-12_2room_ev'
    title = 'LSPI (directed two rooms), rounded'
    data_length1 = 1
    data_length2 = 25
    from load_data import load_data
    import matplotlib.pyplot as plt
    d1 = load_data(filename1)
    d2 = load_data(filename2)
    d3 = load_data(filename3)
    print d1.shape
    print d2.shape
    print d3.shape
    d1 = np.reshape(d1, (data_length1, data_length2))
    d2 = np.reshape(d2, (data_length1, data_length2))
    d3 = np.reshape(d3, (data_length1, data_length2))
    r1 = np.sum(d1, axis=0)
    r2 = np.sum(d2, axis=0)
    r3 = np.sum(d3, axis=0)
    xax = np.arange(1, len(r1)+1)
    fig = plt.figure()
    plt.xlabel('# iterations')
    plt.ylabel('complexity')
    plt.title(title)
    p1 = plt.plot(xax, r1)
    p2 = plt.plot(xax, r2)
    p3 = plt.plot(xax, r3)
    plt.legend((p1[0], p2[0], p3[0]), ('matrix', 'MTBDD', 'EVBDD'))
    fig.savefig('./results/'+title+'png')
    plt.show()


def test_mddd():
    from diagram import MTxDD
    from matrix_and_variable_operations import kronecker_expansion
    mtqdd = MTxDD(3)
    # mat1 = np.array([[1, 7, 0]], dtype=float)
    # mat1 = np.array([[1, 7, 0], [0, -1, 0], [2, 8, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 15, 0]], dtype=float)
    # mat1 = np.random.random((3, 4))
    np.random.seed(0)
    a = (729, 243)
    mat1 = np.flatten(np.random.random(a))
    # mat2 = np.random.random(a)
    # x3gf1 = np.array([[1, 0, 0], [0, 2, 1], [2, 2, 2]])
    # x3gfn = kronecker_expansion(x3gf1, target_mat=mat2)
    # mat1 = np.dot(mat2, x3gfn)
    print 'Reference:'
    print 'Complexity: ' + str(np.prod(mat1.shape))
    # print mat1
    print 'Result:'
    diagram1 = mtqdd.create(mat1, 0, False)
    # print diagram1.to_matrix(7, True)
    print 'Complexity: ' + str(diagram1.complexity())
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reduced:'
    mtqdd.reduce(diagram1)
    # import code; code.interact(local=dict(locals().items() + globals().items()))
    # print diagram1.to_matrix(7, True)
    print 'Complexity: ' + str(diagram1.complexity())
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reduced and lossy:'
    diagram2 = mtqdd.create(mat1, 0, True, dec_digits=2)
    # print diagram2.to_matrix(7, True)
    print 'Complexity: ' + str(diagram2.complexity())


def test_hash():
    from diagram import MTxDD
    mtqdd = MTxDD(4)
    mat1 = np.array([[1, 7, 0], [0, -1, 0], [2, 8, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 15, 0]], dtype=float)
    diagram1 = mtqdd.create(mat1, 0, False)
    print diagram1.__hash__()


def pseudo_reduction():
    basis_function = {
        'id3'       : np.identity(3),
        'gf3'       : np.array([[1, 0, 0], [0, 2, 1], [2, 2, 2]]),
        'rmf3'      : np.array([[1, 0, 0], [1, 2, 0], [1, 1, 1]]),
        # 'id4'       : np.identity(4),
        # 'rmf4'      : np.array([[1, 0, 0, 0], [1, 3, 0, 0], [1, 2, 1, 0], [1, 1, 3, 3]])
    }
    from matrix_and_variable_operations import kronecker_expansion, expand_matrix_exponential, get_req_vars
    from diagram_ternary import MT3DD, AEV3DD, MEV3DD

    for dim in [9]:
        np.random.seed(0)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Dimensionality:   '+str(dim)+' x '+str(dim)
        # mat = np.random.random((dim, dim))
        mat = np.random.normal(loc=0.0, scale=5.0, size=(dim, dim))
        mat_rounded = np.round(mat, 2)
        print mat
        for basis in basis_function:
            mtqdd = MEV3DD()
            print ' - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -'
            print ' -~ ' + basis + ' ~-'
            base = basis_function[basis].shape[0]
            mat_basen = expand_matrix_exponential(mat, get_req_vars(mat, base)[1:], 0, base)
            mat_basen = np.ndarray.flatten(mat_basen)
            mat_basen_rounded = expand_matrix_exponential(mat_rounded, get_req_vars(mat, base)[1:], 0, base)
            mat_basen_rounded = np.ndarray.flatten(mat_basen_rounded)
            basis_expansion = kronecker_expansion(basis_function[basis], target_mat=mat_basen)
            mat_base1 = np.dot(basis_expansion, mat_basen)
            mat_base1_rounded = np.dot(basis_expansion, mat_basen_rounded)
            dd1 = mtqdd.create(mat_base1, 0)
            print '    Full Complexity:  ' + str(dd1.complexity())
            print dd1.to_matrix(9)
            dd2 = mtqdd.create(mat_base1_rounded, 0)
            print '    Lossy Complexity: ' + str(dd2.complexity())


def test_mevdds2():
    from diagram_binary import AABDD
    aadd = AABDD()
    mat1 = np.array([[4, 20, 4], [3, -1, 0], [2, 0, 1], [0, 5, 1], [1, 5, 1], [2, 10, 2], [3, 15, 3]], dtype=float)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reference:'
    print 'Complexity: ' + str(np.prod(mat1.shape))
    print mat1
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AADD'
    diagram1 = aadd.create(mat1, 0, True)
    print 'Complexity: ' + str(diagram1.complexity())
    print diagram1.to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'


basis_function = {
    'id3'       : np.identity(3),
    'gf3'       : np.array([[1, 0, 0], [0, 2, 1], [2, 2, 2]]),
    'rmf3'      : np.array([[1, 0, 0], [1, 2, 0], [1, 1, 1]]),
    # 'id4'       : np.identity(4),
    # 'rmf4'      : np.array([[1, 0, 0, 0], [1, 3, 0, 0], [1, 2, 1, 0], [1, 1, 3, 3]])
}


def test_kronecker():
    print np.array([j for j in range(3**3)])[None]
    from matrix_and_variable_operations import kronecker_expansion
    for i in range(4):
        print kronecker_expansion(basis_function['rmf3'], var=i)


def divide_conquer_kron():
    from matrix_and_variable_operations import kronecker_expansion
    a = np.ones(243*243)
    rand = np.random.random((243*243))
    # kron = kronecker_expansion(basis_function['rmf3'], var=10)
    for i1 in range(3):
        bf = basis_function['rmf3'][i1]
        b = np.kron(a[i1*81:(i1+1)*81], bf)
        print b.shape
        for i2 in range(3):
            cf = basis_function['rmf3'][i2]
            c = np.kron(b[i2*81:(i2+1)*81], cf)
            for i3 in range(3):
                df = basis_function['rmf3'][i3]
                d = np.kron(b[i3*81:(i3+1)*81], df)
                for i4 in range(3):
                    ef = basis_function['rmf3'][i4]
                    e = np.kron(b[i4*81:(i4+1)*81], ef)
                    f = np.remainder(e, 3)
                    # print kron[i1*81+i2*27+i3*9+i4*3]
                    # print f
                    # res = np.dot(f, rand.T)


def test_mevdds():
    from diagram_ternary import MT3DD, AEV3DD, MEV3DD, AA3EVDD
    # from diagram_binary import MT2DD, AEV2DD, MEV2DD, AABDD
    mtdd = MT3DD()
    aevdd = AEV3DD()
    mevdd = MEV3DD()
    aadd = AA3EVDD()
    mat1 = np.array([[8, 20, 8], [3, -1, 0], [2, 0, 1], [0, 5, 1], [1, 5, 1], [2, 10, 2], [11, 15, 11]], dtype=float)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reference:'
    print 'Complexity: ' + str(np.prod(mat1.shape))
    print mat1
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'normal multi-terminal DD'
    diagram1 = mtdd.create(mat1, 0, True)
    print 'Complexity: ' + str(diagram1.complexity())
    print diagram1.to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'additive edge-value DD'
    diagram2 = aevdd.create(mat1, 0, True)
    print 'Complexity: ' + str(diagram2.complexity())
    print diagram2.to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'multiplicative edge-value DD'
    diagram3 = mevdd.create(mat1, 0, True)
    print 'Complexity: ' + str(diagram3.complexity())
    print diagram3.to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AADD'
    diagram4 = aadd.create(mat1, 0, True)
    print 'Complexity: ' + str(diagram4.complexity())
    print diagram4.to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # for node in diagram4.nodes:
    #     try:
    #         if node.child_nodes[0].is_leaf():
    #             print node.offsets
    #     except TypeError:
    #         pass


def test_variable_reordering():
    from diagram_ternary import MT3DD, AEV3DD, MEV3DD, AAEV3DD
    # from diagram_binary import MTDD, AEV2DD, MEV2DD, AAEV2DD
    mtdd = MT3DD()
    aevdd = AEV3DD()
    mevdd = MEV3DD()
    aadd = AAEV3DD()
    # mat1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]], dtype=float)
    mat1 = np.array([[1, 2, 3], [-4, -5, -6], [7, 8, 9], [-10, 11, -12], [13, 14, 15], [16, -17, 18], [19, 20, 21]], dtype=float)
    # np.random.seed(0)
    # mat1 = np.random.random((7,3))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reference:'
    print 'Complexity: ' + str(np.prod(mat1.shape))
    print mat1
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'MTxDD'
    diagram1 = mtdd.create(mat1, 0, True)
    print 'Complexity: ' + str(diagram1.complexity())
    print diagram1.to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'additive edge-value DD'
    diagram2 = aevdd.create(mat1, 0, True)
    print 'Complexity: ' + str(diagram2.complexity())
    print diagram2.to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'multiplicative edge-value DD'
    diagram3 = mevdd.create(mat1, 0, True)
    print 'Complexity: ' + str(diagram3.complexity())
    print diagram3.to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AADD'
    diagram4 = aadd.create(mat1, 0, True)
    print 'Complexity: ' + str(diagram4.complexity())
    print diagram4.to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reference:'
    print 'Complexity: ' + str(np.prod(mat1.shape))
    print mat1.T
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # The transposes
    from diagram_computations import transpose
    transpose(diagram1, 7)
    print 'MTxDD - transposed'
    print 'Complexity: ' + str(diagram1.complexity())
    print diagram1.to_matrix(3, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    transpose(diagram2, 7)
    print 'AEVxDD - transposed'
    print 'Complexity: ' + str(diagram2.complexity())
    print diagram2.to_matrix(3, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    transpose(diagram3, 7)
    print 'MEVxDD - transposed'
    print 'Complexity: ' + str(diagram3.complexity())
    print diagram3.to_matrix(3, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    transpose(diagram4, 7)
    print 'AAEVxDD - transposed'
    print 'Complexity: ' + str(diagram4.complexity())
    print diagram4.to_matrix(3, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'The sums:'
    print 'The reference:   ' + str(np.sum(mat1))
    print 'MTxDD:           ' + str(diagram1.sum())
    print 'AEVxDD:          ' + str(diagram2.sum())
    print 'MEVxDD:          ' + str(diagram3.sum())
    print 'AAEVxDD:         ' + str(diagram4.sum())


def test_elementwise_operations():
    from diagram_computations import multiply_elementwise, addition_elementwise, sum_over_all, dot_product
    # from diagram_ternary import MT3DD, AEV3DD, MEV3DD, AAEV3DD
    from diagram_binary import MT2DD, AEV2DD, MEV2DD, AAEV2DD
    mtdd = MT2DD()
    # mtdd = MT3DD()
    aevdd = AEV2DD()
    # aevdd = AEV3DD()
    mevdd = MEV2DD()
    # mevdd = MEV3DD()
    aadd = AAEV2DD()
    # aadd = AAEV3DD()
    # mat1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]], dtype=float)
    # mat1 = np.array([[1, 2, 3], [-4, -5, -6], [7, 8, 9], [-10, 11, -12], [13, 14, 15], [16, -17, 18], [19, 20, 21]], dtype=float)
    np.random.seed(0)
    mat1 = np.random.random((7, 3))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'The matrix:'
    print mat1
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reference (MULTIPLICATION):'
    print 'Complexity: ' + str(np.prod(mat1.shape))
    print np.multiply(mat1, mat1)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'MTxDD'
    diagram1 = mtdd.create(mat1, 0, True)
    diagram2 = mtdd.create(mat1, 0, True)
    print multiply_elementwise(diagram1, diagram2).to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AEVxDD'
    diagram3 = aevdd.create(mat1, 0, True)
    diagram4 = aevdd.create(mat1, 0, True)
    print multiply_elementwise(diagram3, diagram4).to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'MEVxDD'
    diagram5 = mevdd.create(mat1, 0, True)
    diagram6 = mevdd.create(mat1, 0, True)
    print multiply_elementwise(diagram5, diagram6).to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AAEVxDD'
    diagram7 = aadd.create(mat1, 0, True)
    diagram8 = aadd.create(mat1, 0, True)
    print multiply_elementwise(diagram7, diagram8).to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'The matrix:'
    print mat1
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reference (ADDITION):'
    print 'Complexity: ' + str(np.prod(mat1.shape))
    print np.add(mat1, mat1)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'MTxDD'
    diagram1 = mtdd.create(mat1, 0, True)
    diagram2 = mtdd.create(mat1, 0, True)
    print addition_elementwise(diagram1, diagram2).to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AEVxDD'
    diagram3 = aevdd.create(mat1, 0, True)
    diagram4 = aevdd.create(mat1, 0, True)
    print addition_elementwise(diagram3, diagram4).to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'MEVxDD'
    diagram5 = mevdd.create(mat1, 0, True)
    diagram6 = mevdd.create(mat1, 0, True)
    print addition_elementwise(diagram5, diagram6).to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AAEVxDD'
    diagram7 = aadd.create(mat1, 0, True)
    diagram8 = aadd.create(mat1, 0, True)
    print addition_elementwise(diagram7, diagram8).to_matrix(7, True)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AAEVxDD - sum'
    diagram9 = aadd.create(mat1, 0, True)
    diagram10 = aadd.create(mat1, 0, True)
    print np.sum(mat1)
    print sum_over_all(diagram10)
    print np.sum(np.multiply(mat1, mat1))
    print dot_product(diagram9, diagram10)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'


def test_multiplications():
    from diagram_computations import multiply_elementwise, addition_elementwise, sum_over_all, dot_product, multiply_matrix_by_column_vector, multiply
    # from diagram_ternary import MT3DD, AEV3DD, MEV3DD, AAEV3DD
    from diagram_binary import MT2DD, AEV2DD, MEV2DD, AAEV2DD
    mtdd = MT2DD()
    # mtdd = MT3DD()
    aevdd = AEV2DD()
    # aevdd = AEV3DD()
    mevdd = MEV2DD()
    # mevdd = MEV3DD()
    aadd = AAEV2DD()
    # aadd = AAEV3DD()
    # mat1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]], dtype=float)
    mat2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    vec1 = np.array([2, -3, 5])
    mat1 = np.array([[1, 2, 3], [-4, -5, -6], [7, 8, 9], [-10, 11, -12], [13, 14, 15], [16, -17, 18], [19, 20, 21]], dtype=float)
    # np.random.seed(0)
    # mat1 = np.random.random((7, 3))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'The matrix:'
    print mat1
    print 'The vector:'
    print vec1[None].T
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reference (MULTIPLICATION):'
    print 'Complexity: ' + str(np.prod(mat1.shape))
    ref_mat = np.dot(mat1, vec1[None].T)
    print ref_mat
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'MTxDD'
    diagram1 = mtdd.create(mat1, 0, True)
    diagram2 = mtdd.create(vec1, 0, True)
    mtdd_mat = multiply_matrix_by_column_vector(diagram1, diagram2).to_matrix(7, True)
    print mtdd_mat
    print 'Error:   ' + str(np.sum(mtdd_mat-ref_mat))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AEVxDD'
    diagram3 = aevdd.create(mat1, 0, True)
    diagram4 = aevdd.create(vec1, 0, True)
    aevdd_mat = multiply_matrix_by_column_vector(diagram3, diagram4).to_matrix(7, True)
    print aevdd_mat
    print 'Error:   ' + str(np.sum(aevdd_mat-ref_mat))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'MEVxDD'
    diagram5 = mevdd.create(mat1, 0, True)
    diagram6 = mevdd.create(vec1, 0, True)
    mevdd_mat = multiply_matrix_by_column_vector(diagram5, diagram6).to_matrix(7, True)
    print mevdd_mat
    print 'Error:   ' + str(np.sum(mevdd_mat-ref_mat))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AAEVxDD'
    diagram7 = aadd.create(mat1, 0, True)
    diagram8 = aadd.create(vec1, 0, True)
    aadd_mat = multiply_matrix_by_column_vector(diagram7, diagram8).to_matrix(7, True)
    print aadd_mat
    print 'Error:   ' + str(np.sum(aadd_mat-ref_mat))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'The matrix:'
    print mat1
    print 'The 2nd matrix:'
    print mat2
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Reference (MULTIPLICATION):'
    ref_mat = np.dot(mat1, mat2)
    print ref_mat
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'MTxDD'
    diagram1 = mtdd.create(mat1, 0, True)
    diagram2 = mtdd.create(mat2, 0, True)
    mtdd_mat = multiply(diagram1, diagram2, 3).to_matrix(7, True)
    print mtdd_mat
    print 'Error:   ' + str(np.sum(mtdd_mat-ref_mat))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AEVxDD'
    diagram3 = aevdd.create(mat1, 0, True)
    diagram4 = aevdd.create(mat2, 0, True)
    aevdd_mat = multiply(diagram3, diagram4, 3).to_matrix(7, True)
    print aevdd_mat
    print 'Error:   ' + str(np.sum(aevdd_mat-ref_mat))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'MEVxDD'
    diagram5 = mevdd.create(mat1, 0, True)
    diagram6 = mevdd.create(mat2, 0, True)
    mevdd_mat = multiply(diagram5, diagram6, 3).to_matrix(7, True)
    print mevdd_mat
    print 'Error:   ' + str(np.sum(mevdd_mat-ref_mat))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'AAEVxDD'
    diagram7 = aadd.create(mat1, 0, True)
    diagram8 = aadd.create(mat2, 0, True)
    aadd_mat = multiply(diagram7, diagram8, 3, precision=4).to_matrix(7, True)
    print aadd_mat
    print 'Error:   ' + str(np.sum(aadd_mat-ref_mat))


def test_approx():
    from diagram_computations import approx, exchange_variable_order_with_children
    mat1 = np.array([[ 10,  11,  12,  13,  14,  15,  16,  17],
                     [ 30,  31,  32,  33,  34,  35,  36,  37],
                     [ 50,  51,  52,  53,  54,  55,  56,  57],
                     [ 70,  71,  72,  73,  74,  75,  76,  77],
                     [ 90,  91,  92,  93,  94,  95,  96,  97],
                     [110, 111, 112, 113, 114, 115, 116, 117],
                     [130, 131, 132, 133, 134, 135, 136, 137],
                     [150, 151, 152, 153, 154, 155, 156, 157]], dtype='float64')

    # # mat1 = np.random.random((8, 8))/10
    #
    # # mat1 = np.reshape(mat1, (1, np.prod(mat1.shape)))+ ((np.arange(0, 64, 1))/34) + (np.sin(np.arange(0, 64, 1).astype('float64')/32*np.pi))
    # mat1 = (np.sin(np.arange(0, 64, 1).astype('float64')/16*np.pi))
    from diagram_binary import MT2DD, AEV2DD, MEV2DD, AAEV2DD
    from diagram_ternary import MT3DD, AEV3DD, MEV3DD, AAEV3DD
    aevdd = AEV2DD()
    diagram1 = aevdd.create(mat1, 0)
    # The following will create the variable sequence 013425, which gives every 2nd row and every 2nd column
    #  for 4 (four!) variables.
    # exchange_variable_order_with_children(diagram1, 2)
    # exchange_variable_order_with_children(diagram1, 3)
    mat2 = np.array([np.arange(8, 64, 16), approx(diagram1, 2, 1)[0]])
    print mat2
    print np.reshape(mat2[1], (2,2))
    mat3 = np.array([np.arange(4, 64, 8), approx(diagram1, 3, 1)[0]])
    print mat3
    print np.reshape(mat3[1], (4,2))
    mat4 = np.array([np.arange(2, 64, 4), approx(diagram1, 4, 1)[0]])
    print mat4
    print np.reshape(mat4[1], (4,4))
    mat5 = np.array([np.arange(1, 64, 2), approx(diagram1, 5, 1)[0]])
    print mat5
    print np.reshape(mat5[1], (8,4))
    mat6 = np.array([np.arange(0, 64, 1), approx(diagram1, 6, 1)[0]])
    print mat6
    print np.reshape(mat6[1], (8,8))
    pl.figure()
    pl.plot(mat2[0], mat2[1])
    pl.plot(mat3[0], mat3[1])
    pl.plot(mat4[0], mat4[1])
    pl.plot(mat5[0], mat5[1])
    pl.plot(mat6[0], mat6[1])
    pl.title('2-5 variable approx')
    pl.show()
    # aevdd = MEV2DD()
    # diagram1 = aevdd.create(mat1, 0)
    # mat2 = np.array([np.arange(8, 64, 16), approx(diagram1, 2, 1)[0]])
    # # print mat2
    # mat3 = np.array([np.arange(4, 64, 8), approx(diagram1, 3, 1)[0]])
    # # print mat3
    # mat4 = np.array([np.arange(2, 64, 4), approx(diagram1, 4, 1)[0]])
    # # print mat4
    # mat5 = np.array([np.arange(1, 64, 2), approx(diagram1, 5, 1)[0]])
    # # print mat5
    # mat6 = np.array([np.arange(0, 64, 1), approx(diagram1, 6, 1)[0]])
    # # print mat6
    # pl.figure()
    # pl.plot(mat2[0], mat2[1])
    # pl.plot(mat3[0], mat3[1])
    # pl.plot(mat4[0], mat4[1])
    # pl.plot(mat5[0], mat5[1])
    # pl.plot(mat6[0], mat6[1])
    # pl.title('2-5 variable approx')
    # pl.show()


if __name__ == "__main__":
    test_approx()
    # test_multiplications()
    # test_elementwise_operations()
    # test_mevdds()
    # divide_conquer_kron()
    # divide_conquer_kron()
    # test_mevdds()
    # pseudo_reduction()
    # test_hash()
    # test_mddd()
    # test_reduction()
    # plot_results()
    # plot_graphs()
    # test_performance()
    # test_addition()
    # general_tests()
    # test_multiplication()
    # run_tests2()
    # run_tests()