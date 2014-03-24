# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""
import copy
from node import Node, BNode
from diagram_initialization import append_nodes_recursively, get_var_names
import numpy as np


def diagram_shallow_copy(node):
    """
    This function creates a new diagram based on an existing one as "blueprint".
    I.e. the result will have the same shape
    """
    new_node = type(node)(node.name, node.null_value)
    return new_node


def create_target_diagram(dia1, dia2, **options):
    """
    This function first checks whether two diagrams, dia1 and dia2, are compatible, i.e. whether they represent to
    matrices of the same dimension.
    If so, the function creates a diagram of the same dimension, which can be populated with some operation on the two
    operands.
    :param dia1: one diagram (in order)
    :param dia2: another diagram (in order)
    :param options:
        diagram in_place    if a diagram is specified via the in_place parameter, that diagram will be overwritten
                            by the new diagram
        double null_value   specifies the null_value for the new diagram; defaults to the null_value of dia1 and dia2
    :return: :raise Exception:
    """
    null_value = None
    in_place = False
    # option checking
    if 'null_value' in options:
        null_value = options['null_value']
    if 'in_place' in options:
        if isinstance(options['in_place'], Node):
            in_place = options['in_place']
        else:
            raise Exception('Cannot create the resulting diagram in the place of '+options['in_place'])

    # checking for the same nullValue
    if not dia1.null_value == dia2.null_value and not null_value is None:
        raise Exception('The null-value of the two graphs differ. Adjust the null-value of one of the graphs first or '
                        'force an override with the null_value option.')

    if in_place:
        in_place.nodes = {}
        in_place.leaves = {}
        in_place.root = None
        result = in_place
    else:
        result = diagram_shallow_copy(dia1)
    return result


def scalar_multiply_diagram(diagram, scalar):
    """
    This function multiplies all leaves_array in the diagram by a skalar
    :param diagram:
    :param scalar:
    :return:
    """
    result = copy.deepcopy(diagram)
    for leaf in result.leaves_array:
        leaf.value = leaf.value*scalar
    return result


def get_subdiagrams(diagram, n):
    """
    This function takes a diagram and returns a list of all the subdiagrams at the nth level
    :param diagram:
    :param n:
    """
    subdiags = []

    def get_subdiags_rec(node, level):
        # a nested function for the recursive call
        if level == n:
            subdiags.append(diagram)
        else:
            for child in node.child_nodes:
                get_subdiags_rec(node.child_nodes[child], level-1)
    return subdiags


def add_diagrams(dia1, dia2):
    """
    This function adds two graphs. The underlying logic represents matrix addition, and therefore requires the same
     shape of the two diagrams.

    This function does a bottom-up addition
    # the addition sequence:
    1. Go down the diagram, starting with the same variable for each.
    2. for each edge, do the following:
        a. if the edge exists in both diagrams, add the corresponding node to the new diagram and look at both
            subdiagrams for that new node, and goto 2
        b. if the edge does only exist in one diagram, add the corresponding node to the new diagram and ignore the
            subdiagram (i.e. deepcopy the whole subdiagram)
    3. clean up
    """
    #    result = create_target_diagram(dia1, dia2, **options)
    def add_binary_diagrams_rec(node1, node2, found_leaves):
        """
        This function adds two subdiagrams specified by the respective root_node, node1/node2
        """
        # checking for the type of node:
        if node1.is_leaf() and node2.is_leaf():
            value = node1.value+node2.value
            if value == 0:
                return False
            if value in found_leaves:
                return found_leaves[value]
            leaf = node1.leaf_type(value, value)
            found_leaves[value] = leaf
            return leaf
        else:
            # checking for the cases in which a fork exists in both diagrams
            node = type(node1)(node1.name, node1.null_value)
            # checking for the positive fork:
            if node1.p and node2.p:
                p_edge = add_binary_diagrams_rec(node1.p, node2.p, found_leaves)
                if p_edge:
                    node.p = p_edge
                    node.d = p_edge.d + 1
            # checking for the negative fork:
            if node1.n and node2.n:
                n_edge = add_binary_diagrams_rec(node1.n, node2.n, found_leaves)
                if n_edge:
                    node.n = n_edge
                    node.d = n_edge.d + 1
            # checking for forks off node1 and not off node2
            if node1.p and not node2.p:
                node.p = copy.deepcopy(node1.p)
                node.d = node.p.d + 1
            if node1.n and not node2.n:
                node.n = copy.deepcopy(node1.n)
                node.d = node.n.d + 1
            # checking for forks off node2 and not off node1
            if node2.p and not node1.p:
                node.p = copy.deepcopy(node2.p)
                node.d = node.p.d + 1
            if node2.n and not node1.n:
                node.n = copy.deepcopy(node2.n)
                node.d = node.n.d + 1
            # If nothing caught a return statement, return False
            if node.n or node.p:
                return node
            return False

    return add_binary_diagrams_rec(dia1, dia2, {})


def elementwise_multiply_diagrams(dia1, dia2, **options):
    """
    This method multiplies two diagrams element-wise, i.e. the MATLAB .* operation.
    :param dia1:
    :param dia2:
    :return: a diagram representing the element-wise matrix product
    """
#    opt = convert_options(options)
    def elementwise_multiply_diagrams_rec(node1, node2, found_leaves):
        """
        This function multiplies two subdiagrams specified by the respective root_node, node1/node2
        """
        # checking for the type of node:
        # iff both diagrams are nodes
        if node1.is_leaf() and node2.is_leaf():
            value = node1.value*node2.value
            # making sure that leaves are not added multiple times
            if value in found_leaves:
                return found_leaves[value]
            leaf = node1.leaf_type(value, value)
            found_leaves[value] = leaf
            return leaf
        else:
            # checking for the cases in which a fork exists in both diagrams
            node_p = node1.p and node2.p
            node_n = node1.n and node2.n
            if node_p or node_n:
                node = type(node1)(node1.name, node1.null_value)
                succ = False
                # checking for the positive fork:
                if node_p:
                    p_edge = elementwise_multiply_diagrams_rec(node1.p, node2.p, found_leaves)
                    if p_edge:
                        node.p = p_edge
                        node.d = p_edge.d + 1
                        succ = True
                # checking for the negative fork:
                if node_n:
                    n_edge = elementwise_multiply_diagrams_rec(node1.n, node2.n, found_leaves)
                    if n_edge:
                        node.n = n_edge
                        node.d = n_edge.d + 1
                        succ = True
                if succ:
                    return node
            return False

    return elementwise_multiply_diagrams_rec(dia1, dia2, {})


def diagram_sum(node):
    """
    This function sums up all leaf-values of a  diagram disregarding any dimensionality
    """
    nsum = 0
    for leaf in node.leaves:
        nsum += leaf.value
    return nsum


def multiply_by_column_vector(mat_diagram, vec_diagram):
    """
    This function multiplies a matrix, represented by a diagram, with a vector, represented by a diagram.
    General idea:
    1. select the rows
    2. multiply them with the vector (elementwise-multiplication)
    3. add them up
    :param mat_diagram: The diagram representing the matrix
    :param vec_diagram: The diagram representing the vector
    """
    def multiply_bdiagram_by_vector_rec(matd, vecd):
        """
        The recursive function
        """
        try:
            matd.d == False
        except AttributeError:
            raise AttributeError
        if matd.d > vecd.d:
            # still selecting rows:
            node = type(matd)('', matd.d-vecd.d)
            succ = False
            if matd.n:
                n_edge = multiply_bdiagram_by_vector_rec(matd.n, vecd)
                if n_edge:
                    node.n = n_edge
                    succ = True
            if matd.p:
                p_edge = multiply_bdiagram_by_vector_rec(matd.p, vecd)
                if p_edge:
                    node.p = p_edge
                    succ = True
            if succ:
                return node
            return False
        else:
            # if a row is selected, compute the scalar-product (sum of .*)
            mult_diags = elementwise_multiply_diagrams(matd, vecd)
            if mult_diags:
                value = diagram_sum(elementwise_multiply_diagrams(matd, vecd))
            else:
                return False
            leaf = matd.leaf_type(str(value), value)
            return leaf
    return multiply_bdiagram_by_vector_rec(mat_diagram, vec_diagram)


def decompose_paths(node):
    """
    This function decomposes a diagram into the set of its paths from root to leaves
    :param node:
    """
    if node.p is False and node.n is False:
        return []
    def decompose_paths_rec(node_inner, path):
        """
        This function does the recursive path of the decomposition
        :param node_inner:
        :param path:
        """
        if node_inner.is_leaf():
            path = np.append(path, str(node_inner.value))
            return path[None]
        else:
            paths = np.array([])
            if node_inner.p:
                new_path = np.append(path, '1')
                p_paths = decompose_paths_rec(node_inner.p, new_path)
                paths = np.append(paths, p_paths)
            if node_inner.n:
                new_path = np.append(path, '0')
                n_paths = decompose_paths_rec(node_inner.n, new_path)
                paths = np.append(paths, n_paths)
        return paths

    decomposition = decompose_paths_rec(node, np.array([]))
    return decomposition.reshape((decomposition.shape[0]/(node.d+1), node.d+1))


def transpose_diagram(diagram, rows=None):
    """
    This transposes the underlying matrix. The basic operation is to exchange the horizontal with the vertical variables,
    or, in diagram-terms, to exchange the upper half of the diagram representing the vertical variables with the lower
    half, representing the horizontal variables.
    :param diagram:
    """
    if diagram.p is False and diagram.n is False:
        return copy.copy(diagram)
    if rows is None:
        rows1 = diagram.shape[0]
    else:
        rows1 = np.ceil(np.log2(rows))
    paths = decompose_paths(diagram)
    new_paths = np.hstack((paths[:, rows1:-1], paths[:, :rows1], paths[:, -1][None].T))
    # TODO: clean up the diagram creation code, and remove redundancies
    node = type(diagram)('x')
    node.null_value = diagram.null_value
    var_names = np.append(get_var_names(new_paths.shape[1]-1, 'x'), '-1')
    tmp_mat = np.vstack((var_names, new_paths))
    append_nodes_recursively(node, tmp_mat, {})
    if node.p:
        node.d = node.p.d + 1
    else:
        node.d = node.n.d + 1
    return node


def multiply_diagram(diag1, diag2, height, transpose=True):
    """
    This function multiplies two diagrams by the logic of the matrix multiplication
    """
    assert isinstance(diag1, Node)
    assert isinstance(diag2, Node)
    if transpose:
        diag3 = transpose_diagram(diag2, height)
    else:
        diag3 = diag2

    def multiply_bdiagram_rec(node1, node2, w):
        """
        The recursive function for multiplying two MTBDDs
        :param node1: the first diagram
        :param node2: the second diagram
        :param w: the width of the matrix, in 2^n
        :return: their product
        """
        if node1.d == w+1:
            # selected rows of node1
            node = type(node1)('x')
            # multiplying rows by the other diagram
            succ = False
            if node1.n:
                n_edge = multiply_by_column_vector(node2, node1.n)
                if n_edge:
                    node.n = n_edge
                    succ = True
            if node1.p:
                p_edge = multiply_by_column_vector(node2, node1.p)
                if p_edge:
                    node.p = p_edge
                    succ = True
            if succ:
                return node
            else:
                return False
        elif node1.d > w+1:
            # if row-blocks are selected, go further down
            node = type(node1)('x')
            succ = False
            if node1.p:
                p_edge = multiply_bdiagram_rec(node1.p, node2, w)
                if p_edge:
                    node.p = p_edge
                    succ = True
            if node1.n:
                n_edge = multiply_bdiagram_rec(node1.n, node2, w)
                if n_edge:
                    node.n = n_edge
                    succ = True
            if succ:
                return node
            else:
                return False
        else:
            return False
    result = multiply_bdiagram_rec(diag1, diag3, np.ceil(np.log2(height)))
    if not result:
        result = type(diag1)('x')
    return result


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


if __name__ == "__main__":
    misses = test_multiplication(500, 25, 0.6)
    i = 0
    for miss in misses:
        i += 1
    print i
    # run_tests2()
    # run_tests()
