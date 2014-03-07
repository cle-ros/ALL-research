# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""
import copy
# from diagram import *
from node import Node, BNode


def diagram_shallow_copy(node):
    """
    This function creates a new diagram based on an existing one as "blueprint".
    I.e. the result will have the same shape
    """
    new_node = type(node)(node.name, node.null_value)
#    new_node.shape = node.shape
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

    # making sure the two diagrams each have the same shape
#    if not dia1.shape == dia2.shape:
#        raise Exception('Can only add two diagrams if the underlying data is of the same dimensionality.')

    # checking for the same nullValue
    if not dia1.null_value == dia2.null_value and not null_value is None:
        raise Exception('The null-value of the two graphs differ. Adjust the null-value of one of the graphs first or '
                        'force an override with the null_value option.')

    result = object
    if in_place:
        result = in_place
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
                get_subdiags_rec(node.child_nodes[child])
    return subdiags


def transpose_diagram(diagram):
    """
    This transposes the underlying matrix. The basic operation is to exchange the horizontal with the vertical variables,
    or, in diagram-terms, to exchange the upper half of the diagram representing the vertical variables with the lower
    half, representing the horizontal variables.
    :param diagram:
    """
    result = create_target_diagram(diag1, diag1)
    steps_down = diagram.shape[0]
    subdiagrams = get_subdiagrams(diagram.root, steps_down-1)
    #UNFINISHED
    return


def multiply_diagram(diag1, diag2, **options):
    """
    This function multiplies two diagrams by the logic of the matrix multiplication
    """
    result = create_target_diagram(diag1, diag2, **options)
    #UNFINISHED
    return


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
                # checking for the positive fork:
                if node_p:
                    p_edge = elementwise_multiply_diagrams_rec(node1.p, node2.p, found_leaves)
                    if p_edge:
                        node.p = p_edge
                        node.d = p_edge.d + 1
                    else:
                        return False
                # checking for the negative fork:
                if node_n:
                    n_edge = elementwise_multiply_diagrams_rec(node1.n, node2.n, found_leaves)
                    if n_edge:
                        node.n = n_edge
                        node.d = n_edge.d + 1
                    else:
                        return False
                return node
            else:
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
        if matd.d > vecd.d:
            # still selecting rows:
            node = type(matd)('', matd.d-vecd.d)
            if matd.n:
                node.n = multiply_bdiagram_by_vector_rec(matd.n, vecd)
            if matd.p:
                node.p = multiply_bdiagram_by_vector_rec(matd.p, vecd)
            return node
        else:
            # if a row is selected, compute the scalar-product (sum of .*)
            value = diagram_sum(elementwise_multiply_diagrams(matd, vecd))
            leaf = matd.leaf_type(str(value), value)
            return leaf
    return multiply_bdiagram_by_vector_rec(mat_diagram, vec_diagram)


if __name__ == "__main__":
    import numpy as np
    from diagram_initialization import initialize_diagram
    #mat1 = np.random.random_integers(0,5,[3,3])
    #mat2 = np.random.random_integers(-5,0,[3,3])
    mat1 = np.array([[1,2,0],[0,2,0],[0,2,1],[1,2,0]])
    mat2 = np.array([[0,-2,0],[0,-2,0],[0,-2,-1]])
    vec1 = np.array([1.0, 2.0, 3.0])
    diag1 = BNode('x')
    diag2 = BNode('y')
    vecDiag = BNode('z')
    initialize_diagram(vecDiag, vec1, 0)
    initialize_diagram(diag1, mat1, 0)
    initialize_diagram(diag2, mat2, 0)
    print mat1
    print vec1[None].T
#    print 'buh'
    diag3 = multiply_by_column_vector(diag1, vecDiag)
    # diag3 = elementwise_multiply_diagrams(diag1, diag2)
    print 'lala'
    print diag3.to_matrix(4, True)
    print 'hi'
    print diag3.d
    #print diag3.to_matrix()
    # import code; code.interact(local=dict(locals().items() + globals().items()))
    #a=BNode('hallo','x1')
    #b=BNode('hallo1','x2',p=a)
    #c=BNode('hallo1','x2',p=b)
    #d=BNode('hallo1','x2',n=b)