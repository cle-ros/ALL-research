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
    diagram.reinitialize_nodes()
    result = copy.deepcopy(diagram)

    for node in result.nodes:
        node.dtype.scalar_mult(node, scalar)

    return result


def get_subdiagrams(diagram, depth):
    """
    This function takes a diagram and returns a list of all the subdiagrams at the nth level
    :param diagram:
    :param depth:
    """
    subdiags = []

    def get_subdiags_rec(node, level):
        # a nested function for the recursive call
        if level == depth:
            subdiags.append(diagram)
        else:
            for child in node.child_nodes:
                get_subdiags_rec(node.child_nodes[child][0], level-1)
    return subdiags


def add_diagrams(dia1, dia2, to_reduce=True, offset=[0, 0]):
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
    def add_binary_diagrams_rec(node1, node2, ioffset):
        """
        This function adds two subdiagrams specified by the respective root_node, node1/node2
        """
        # checking for the type of node:
        if node1.is_leaf() and node2.is_leaf():
            value = node1.dtype.add(node1, node2, ioffset)
            leaf = node1.leaf_type(value, value, diagram_type=node1.dtype)
            return leaf
        else:
            # creating the node
            node = type(node1)(diagram_type=node1.dtype, nv=node1.null_value)
            # computing the new offset
            n_offset, p_offset = node.dtype.add(node1, node2, ioffset)
            # doing the recursive call
            node.n = add_binary_diagrams_rec(node1.n, node2.n, [0, 0, 0])
            node.p = add_binary_diagrams_rec(node1.p, node2.p, [0, 0, 0])
            # setting the new offsets
            node.no = n_offset
            node.po = p_offset
            node.d = node1.d
            return node

    diagram = add_binary_diagrams_rec(dia1, dia2, offset)
    if to_reduce:
        diagram.dtype.reduce(diagram)
    return diagram


def elementwise_multiply_diagrams_evbdd(dia1, dia2, loffset=0, roffset=0, to_reduce=True, with_offset=False):
    """
    This method multiplies two diagrams element-wise, i.e. the MATLAB .* operation.
    :param dia1:
    :param dia2:
    :return: a diagram representing the element-wise matrix product
    """
    from binary_diagram import EVBDD
#    opt = convert_options(options)

    def elementwise_multiply_diagrams_evbdd_rec(node1, node2, offset1, offset2):
        """
        This function multiplies two subdiagrams specified by the respective root_node, node1/node2
        """
        # idea: go down both diagrams. Collect the first node's offsets. Once reached the final point, take the entire
        # offset for the first node, multiply it with the second node's value, and return it to the higher recursion
        # level
        # checking for the type of node:
        # iff both diagrams are nodes
        if node1.n.is_leaf():
            # making sure that leaves are not added multiple times
            n_value = (node1.n.value + node1.no + offset1) * (node2.n.value + node2.no + offset2)
            p_value = (node1.p.value + node1.po + offset1) * (node2.p.value + node2.po + offset2)
            node = node1.create_node()
            node.d = 1
            node, offset = EVBDD.create_leaves(node, [n_value, p_value])
            return node, offset, 0
        else:
            # computing the previous sum
            # go_on prevents to much computation for some diagram types where a recursive multiplication is not
            # necessary
            node = node1.create_node()
            node.n, n_offset, depth = elementwise_multiply_diagrams_evbdd_rec(node1.n, node2.n, node1.no + offset1, node2.no + offset2)
            node.p, p_offset, depth = elementwise_multiply_diagrams_evbdd_rec(node1.p, node2.p, node1.po + offset1, node2.po + offset2)
            depth += 1
            node, offset = EVBDD.create_tuple(node, n_offset, p_offset)
            node.d = depth
            return node, offset, depth

    diagram, f_offset, odepth = elementwise_multiply_diagrams_evbdd_rec(dia1, dia2, loffset, roffset)
    # making sure that the entire diagram is not "off" by the final offset
    if f_offset != 0 and with_offset is False:
        EVBDD.include_final_offset(diagram, f_offset)
    if to_reduce:
        diagram.dtype.reduce(diagram)
    if with_offset:
        return diagram, f_offset
    return diagram


def elementwise_multiply_diagrams_mtbdd(dia1, dia2, loffset=0, roffset=0, to_reduce=True, with_offset=False):
    """
    This method multiplies two diagrams element-wise, i.e. the MATLAB .* operation.
    :param dia1:
    :param dia2:
    :return: a diagram representing the element-wise matrix product
    """
    # opt = convert_options(options)
    def elementwise_multiply_diagrams_mtbdd_rec(node1, node2, found_leaves):
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
            leaf = node1.create_leaf(value)
            found_leaves[value] = leaf
            return leaf
        else:
            # checking for the cases in which a fork exists in both diagrams
            node_p = node1.p and node2.p
            node_n = node1.n and node2.n
            if node_p or node_n:
                node = node1.create_node()
                succ = False
                # checking for the positive fork:
                if node_p:
                    p_edge = elementwise_multiply_diagrams_mtbdd_rec(node1.p, node2.p, found_leaves)
                    if p_edge:
                        node.p = p_edge
                        node.d = p_edge.d + 1
                        succ = True
                # checking for the negative fork:
                if node_n:
                    n_edge = elementwise_multiply_diagrams_mtbdd_rec(node1.n, node2.n, found_leaves)
                    if n_edge:
                        node.n = n_edge
                        node.d = n_edge.d + 1
                        succ = True
                if succ:
                    return node
            return False

    diagram = elementwise_multiply_diagrams_mtbdd_rec(dia1, dia2, {})
    # making sure that the entire diagram is not "off" by the final offset
    if to_reduce:
        diagram.dtype.reduce(diagram)
    if with_offset:
        return diagram, 0
    return diagram


def diagram_sum(node):
    """
    This function sums up all leaf-values of a  diagram disregarding any dimensionality
    """
    from node import Leaf

    def diagram_sum_rec(node1, offset):
        """
        The recursive pendant
        """
        if isinstance(node1, Leaf):
            # computing the leaf sum
            return node1.dtype.sum(node1, offset)
        else:
            # computing the new offsets
            n_offset, p_offset = node1.dtype.sum(node1, offset)
            # doing the recursive call
            n_value = diagram_sum_rec(node1.n, n_offset)
            p_value = diagram_sum_rec(node1.p, p_offset)
            return n_value + p_value

    return diagram_sum_rec(node, 0)


def multiply_by_column_vector(mat_diagram, vec_diagram, ooffset=0):
    """
    This function multiplies a matrix, represented by a diagram, with a vector, represented by a diagram.
    General idea:
    1. select the rows
    2. multiply them with the vector (elementwise-multiplication)
    3. add them up
    :param mat_diagram: The diagram representing the matrix
    :param vec_diagram: The diagram representing the vector
    """

    from binary_diagram import MTBDD, EVBDD
    if mat_diagram.dtype is MTBDD:
        elementwise_multiply_diagrams = elementwise_multiply_diagrams_mtbdd
    elif mat_diagram.dtype is EVBDD:
        elementwise_multiply_diagrams = elementwise_multiply_diagrams_evbdd

    def multiply_bdiagram_by_vector_rec(matd, vecd, offset):
        """
        The recursive function
        :param offset:
        """
        try:
            matd.d is False
        except AttributeError:
            raise AttributeError
        if matd.d > vecd.d:
            # still selecting rows:
            node = matd.create_node(depth=matd.d-vecd.d)
            succ = False
            if matd.n:
                n_edge = multiply_bdiagram_by_vector_rec(matd.n, vecd, matd.dtype.collapse_node(matd.no, offset))
                if n_edge:
                    node.n = n_edge
                    succ = True
            if matd.p:
                p_edge = multiply_bdiagram_by_vector_rec(matd.p, vecd, matd.dtype.collapse_node(matd.po, offset))
                if p_edge:
                    node.p = p_edge
                    succ = True
            if succ:
                return node
            return False
        else:
            # if a row is selected, compute the scalar-product (sum of .*)
            mult_diags = elementwise_multiply_diagrams(matd, vecd, loffset=offset)
            if mult_diags:
                value = diagram_sum(mult_diags)
            else:
                return False
            leaf = matd.create_leaf(value)
            return leaf
    return multiply_bdiagram_by_vector_rec(mat_diagram, vec_diagram, ooffset)


def transpose_diagram(diagram, rows=None):
    """
    This transposes the underlying matrix. The basic operation is to exchange the horizontal with the vertical variables,
    or, in diagram-terms, to exchange the upper half of the diagram representing the vertical variables with the lower
    half, representing the horizontal variables.
    :param diagram:
    """
    # assert isinstance(diagram, BNode)
    # if diagram.p is False and diagram.n is False:
    #     return copy.copy(diagram)
    # if rows is None:
    #     rows1 = diagram.shape[0]
    # else:
    #     rows1 = np.ceil(np.log2(rows))
    # paths = diagram.decompose_paths()
    # new_paths = np.hstack((paths[:, rows1:-1], paths[:, :rows1], paths[:, -1][None].T))
    # TODO: clean up the diagram creation code, and remove redundancies
    # node = type(diagram)('x')
    # node.null_value = diagram.null_value
    # var_names = np.append(get_var_names(new_paths.shape[1]-1, 'x'), '-1')
    # tmp_mat = np.vstack((var_names, new_paths))
    # append_nodes_recursively(node, tmp_mat, {})
    # if node.p:
    #     node.d = node.p.d + 1
    # else:
    #     node.d = node.n.d + 1
    # return node
    d_instance = diagram.dtype()
    return d_instance.create(diagram.to_matrix(rows).T, 0)


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

    def multiply_bdiagram_rec(node1, node2, w, loffset=0, roffset=0):
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
                n_edge = multiply_by_column_vector(node2, node1.n, loffset)
                if n_edge:
                    node.n = n_edge
                    succ = True
            if node1.p:
                p_edge = multiply_by_column_vector(node2, node1.p, loffset)
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
                p_edge = multiply_bdiagram_rec(node1.p, node2, w, loffset=node1.dtype.collapse_node(node1.po, loffset))
                if p_edge:
                    node.p = p_edge
                    succ = True
            if node1.n:
                n_edge = multiply_bdiagram_rec(node1.n, node2, w, loffset=node1.dtype.collapse_node(node1.no, loffset))
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



