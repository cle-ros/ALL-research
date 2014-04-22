# -*- coding: utf-8 -*-
"""
@author: Clemens Rosenbaum (cgbr@cs.umass.edu)
"""

import numpy as np
from node import BNode


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
    from diagram_initialization import initialize_diagram, expand_matrix2n
    from diagram_computations import multiply_diagram
    for i in range(number):
        # creating the sizes of the matrices
        size_first = [np.random.random_integers(2, max_size, 1)[0], np.random.random_integers(2, max_size, 1)[0]]
        size_second = [size_first[1], np.random.random_integers(2, max_size, 1)[0]]
        # creating the random matrices
        mat1 = np.random.rand(size_first[0], size_first[1])*2-1
        mat2 = np.random.rand(size_second[0], size_second[1])*2-1
        # adding sparsity
        spars1 = np.round(np.random.rand(size_first[0], size_first[1]) * 0.5/sparsity)
        spars2 = np.round(np.random.rand(size_second[0], size_second[1]) * 0.5/sparsity)
        mat1 = mat1*spars1
        mat2 = mat2*spars2
        # creating the diagrams
        diag1 = BNode('x')
        diag2 = BNode('y')
        initialize_diagram(diag1, mat1, 0)
        initialize_diagram(diag2, mat2, 0)
        # computing the multiplication
        result = multiply_diagram(diag1, diag2, mat1.shape[1])
        result_mat = result.to_matrix(mat1.shape[0], False)
        if not (result_mat == [0]).all():
            reference_result = expand_matrix2n(np.dot(mat1, mat2), np.log2(result_mat.shape), 0)
            diff = np.max(np.abs(reference_result - result_mat))
            if not diff < 1e-10:
                import code; code.interact(local=dict(locals().items() + globals().items()))
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
    print np.max(np.abs(np.dot(mat1, mat2)-diag3.to_matrix(7, True)))
    print('hi')
    print(diag3.d)
    print diag3.to_matrix()
    import code; code.interact(local=dict(locals().items() + globals().items()))
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
    from binary_diagram import MTBDD
    mat1 = np.array([[1, 2, 0], [0, 3, 0], [0, 4, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 5, 0]])
    mtbdd = MTBDD()
    node = mtbdd.create(mat1, 0)
    print node.to_matrix(7)
    print node.get_subdiagrams(5)


def test_reduction():
    from binary_diagram import MTBDD, EVBDD
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
    from binary_diagram import MTBDD, EVBDD
    mat1 = np.array([[1, 7, 0], [0, -1, 0], [2, 8, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 15, 0]], dtype=float)
    print mat1+mat1
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


def test_multiplication():
    from binary_diagram import MTBDD, EVBDD
    mat1 = np.array([[1, 7, 0], [0, -1, 0], [2, 8, 1], [1, 5, 0], [1, 5, 0], [1, 5, 0], [1, 15, 0]], dtype=float)
    print mat1 * 2.5
    print('MTBDD:')
    from diagram_computations import scalar_multiply_diagram
    mtbdd = MTBDD()
    node1 = mtbdd.create(mat1, 0)
    node2 = scalar_multiply_diagram(node1, 2.5)
    print node2.to_matrix(7, True)
    print('EVBDD:')
    evbdd = EVBDD()
    node4 = evbdd.create(mat1, 0)
    node5 = scalar_multiply_diagram(node4, 2.5)
    print node5.to_matrix(7, True)


if __name__ == "__main__":
    test_multiplication()
    # run_tests2()
    # run_tests()