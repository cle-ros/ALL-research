# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""
import copy
from node import Node
from diagram_exceptions import OutOfBounds
import numpy as np


def scalar_multiply_diagram(diagram, scalar):
    """
    This function multiplies all leaves_array in the diagram by a skalar
    :param diagram:
    :param scalar:
    :return:
    """
    diagram.reinitialize_nodes()
    result = copy.deepcopy(diagram)

    diagram.dtype.scalar_mult(diagram, scalar)

    return result


def multiply_diagram(diag1, diag2, height, to_transpose=True):
    """
    This function multiplies two diagrams by the logic of the matrix multiplication
    """
    assert isinstance(diag1, Node)
    assert isinstance(diag2, Node)
    if to_transpose:
        diag3 = transpose(diag2, height)
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
        if node1.d == w:
            # selected rows of node1
            node = node1.create_node()
            # multiplying rows by the other diagram
            succ = False
            if node1.n:
                n_edge = multiply_by_column_vector(node2, node1.n, loffset)
                if n_edge:
                    depth = n_edge.d + 1
                    node.n = n_edge
                    succ = True
            if node1.p:
                p_edge = multiply_by_column_vector(node2, node1.p, loffset)
                if p_edge:
                    depth = p_edge.d + 1
                    node.p = p_edge
                    succ = True
            if succ:
                node.d = depth
                return node
            else:
                return False
        elif node1.d > w:
            # if row-blocks are selected, go further down
            node = type(node1)('x')
            succ = False
            if node1.p:
                p_edge = multiply_bdiagram_rec(node1.p, node2, w, loffset=node1.dtype.collapse_node(node1.po, loffset))
                if p_edge:
                    node.p = p_edge
                    depth = p_edge.d + 1
                    succ = True
            if node1.n:
                n_edge = multiply_bdiagram_rec(node1.n, node2, w, loffset=node1.dtype.collapse_node(node1.no, loffset))
                if n_edge:
                    node.n = n_edge
                    depth = n_edge.d + 1
                    succ = True
            if succ:
                node.d = depth
                return node
            else:
                return False
        else:
            return False
    result = multiply_bdiagram_rec(diag1, diag3, np.ceil(np.log2(height)))
    if not result:
        result = type(diag1)('x')
    return result


def exchange_variable_order_with_children(node, depth):
    """
    This function exchanges the variable encoded by the subdiagrams at a specified level with the following variable,
    i.e. the variable encoded by the child-subdiagrams
    :param node: the main diagram
    :param depth: the level of subdiagrams encoding the variable to be exchanged with its successor
    :return: the same node, with a different variable ordering
    """
    # unfortunately, if the depth equals zero, this is a special case ...
    if depth == 0:
        exchange_matrix = []
        for child_0 in node.child_nodes:
            exchange_row = [None, child_0, node.get_offset(child_0), None, None, None]
            children_1 = node.child_nodes[child_0].child_nodes
            # exchanging the order
            # looping over the different offsets:
            for child_1 in children_1:
                exchange_row[3] = child_1
                exchange_row[4] = node.child_nodes[child_0].get_offset(child_1)
                exchange_row[5] = children_1[child_1]
                exchange_matrix.append(list(exchange_row))
        # swapping the child-relations between the two levels
        # clearing the previous data
        node.child_nodes = {}
        node.offsets = {}
        # computing the combined offsets:
        exchange_matrix = node.dtype.rearrange_offsets(exchange_matrix, node.dtype)
        # iterating over the swaps
        for ex_row in exchange_matrix:
            # creating a new node at first level and setting the depth
            if not ex_row[1] in node.child_nodes:
                node.child_nodes[ex_row[1]] = node.create_node()
                node.set_offset(ex_row[1], ex_row[2])
                node.child_nodes[ex_row[1]].d = ex_row[5].d + 1
            # swapping the children at second level and the depth
            node.child_nodes[ex_row[1]].child_nodes[ex_row[3]] = ex_row[5]
            node.child_nodes[ex_row[1]].set_offset(ex_row[3], ex_row[4])
    # the case for non-zero root level
    else:
        subdiagrams_at_level = list(node.get_subdiagrams(depth-1))
        # looping over the diagrams, to exchange the variable ordering
        for sd in subdiagrams_at_level:
            exchange_matrix = []
            # storing the children, for the child_node relation will be dissolved
            for child_lvl_0 in sd.child_nodes:
                exchange_row = [child_lvl_0, None, None, None, None, None]
                children_level_1 = sd.child_nodes[child_lvl_0].child_nodes
                exchange_matrix_part = []
                # iterating over the children, which will be exchanged with their children in turn
                for child_lvl_1 in children_level_1:
                    exchange_row[1] = child_lvl_1
                    exchange_row[2] = sd.child_nodes[child_lvl_0].get_offset(child_lvl_1)
                    children_level_2 = children_level_1[child_lvl_1].child_nodes
                    # exchanging the order
                    # 1. computing the combined offsets:
                    # looping over the different offsets:
                    for child_lvl_2 in children_level_2:
                        exchange_row[3] = child_lvl_2
                        exchange_row[4] = children_level_1[child_lvl_1].get_offset(child_lvl_2)
                        exchange_row[5] = children_level_2[child_lvl_2]
                        exchange_matrix_part.append(list(exchange_row))
                exchange_matrix_part = node.dtype.rearrange_offsets(exchange_matrix_part, node.dtype)
                exchange_matrix += exchange_matrix_part
            # swapping the child-relations between the two levels
            # clearing the previous data
            sd.child_nodes = {}
            # computing the combined offsets:
            # iterating over the swaps
            for ex_row in exchange_matrix:
                # creating a new node at first level and setting the depth
                if not ex_row[0] in sd.child_nodes:
                    sd.child_nodes[ex_row[0]] = node.create_node()
                    sd.child_nodes[ex_row[0]].d = ex_row[5].d + 2
                # creating a new node at second level and setting the depth
                if not ex_row[1] in sd.child_nodes[ex_row[0]].child_nodes:
                    sd.child_nodes[ex_row[0]].child_nodes[ex_row[1]] = node.create_node()
                    sd.child_nodes[ex_row[0]].set_offset(ex_row[1], ex_row[2])
                    sd.child_nodes[ex_row[0]].child_nodes[ex_row[1]].d = ex_row[5].d + 1
                # swapping the children at second level and the depth
                sd.child_nodes[ex_row[0]].child_nodes[ex_row[1]].child_nodes[ex_row[3]] = ex_row[5]
                sd.child_nodes[ex_row[0]].child_nodes[ex_row[1]].set_offset(ex_row[3], ex_row[4])
    node.reinitialize()
    node.reduce()


def transpose(node, height, to_copy=False):
    """
    Guess what this function does?
    :param node:
    :param height: the height of the original matrix
    :param to_copy: if false, the diagram will be transposed in place. If true, a copy of the diagram will be transposed
    :return:
    """
    if to_copy:
        import copy
        node_t = copy.deepcopy(node)
    else:
        node_t = node
    import numpy as np
    # getting the log of the matrix height, representing the level in the diagram
    height_log = int(np.ceil(np.log10(height)/np.log10(node_t.dtype.base)))
    # if the height_log equals the number of variables in the diagram, the diagram equals its transpose
    if height_log == node_t.d:
        return
    # the permutation chain for the first non-row index
    height_variables = np.arange(height_log-1, -1, -1)
    # and for all the others
    change_vector = np.array([])
    for i in range(node_t.d-height_log):
        change_vector = np.hstack((change_vector, (height_variables+i)))
    # executing the permutations
    for permutation in change_vector:
        exchange_variable_order_with_children(node_t, permutation)
    return node_t


def switch_variable_reference(node, depth):
    import copy as cp
    subdiagrams_at_level = list(node.get_subdiagrams(depth))
    for sd in subdiagrams_at_level:
        children = cp.copy(sd.child_nodes)
        for i in range(node.dtype.base):
            sd.child_nodes[node.dtype.base-1-i] = children[i]


def hash_for_multiple_nodes(node1, node2, offset1, offset2):
    """
    This function creates a hash for multiple nodes. This hash is a long int, and can therefore not be used
    for the builtin python __hash__() functions.
    :param node1:
    :param node2:
    :return:
    """
    return hash(str(node1.__hash__()) + str(node2.__hash__()) + str(offset1) + str(offset2))


def multiply_elementwise(diagram1, diagram2, dec_digits=-1, outer_offset=None):
    """
    This function multiplies two diagram elementwise
    (i.e. like numpy.multiply(d1, d2) or d1.*d2 in Matlab)
    This method raises an out of bounds diagram exception if the two diagrams' sizes do not match.
    :param diagram1: The first diagram
    :param diagram2: The second diagram
    :return: a diagram (a new object) of the elementwise multiplication
    """
    import numpy
    return operation_elementwise(diagram1, diagram2, numpy.multiply, dec_digits, outer_offset)


def addition_elementwise(diagram1, diagram2, dec_digits=-1, outer_offset=None):
    """
    This function provides normal matrix addition.
    :param diagram1:
    :param diagram2:
    :return:
    """
    import numpy
    return operation_elementwise(diagram1, diagram2, numpy.add, dec_digits, outer_offset)


def operation_elementwise(diagram1, diagram2, operation, precision, outer_offset):
    """
    This function multiplies two diagram elementwise
    (i.e. like numpy.multiply(d1, d2) or d1.*d2 in Matlab)
    This method raises an out of bounds diagram exception if the two diagrams' sizes do not match.
    :param diagram1: The first diagram
    :param diagram2: The second diagram
    :param operation: The operation to be performed as a function of the numpy package, i.e. np.multiply
    :return: a diagram (a new object) of the elementwise multiplication
    """
    # The hashmap of the results, to minimize computations
    hashmap_of_results = {}
    dd = diagram1.dtype()
    if precision != -1:
        print precision
        def rounding(array):
            return np.round(array, precision)
    else:
        def rounding(array):
            return array

    # the recursive function
    def operation_elementwise_rec(node1, node2, offset1, offset2):
        # do we know the result already?
        hash_of_current_operation = hash_for_multiple_nodes(node1, node2, offset1, offset2)
        if hash_of_current_operation in hashmap_of_results:
            return hashmap_of_results[hash_of_current_operation]

        node = diagram1.__class__('', diagram_type=diagram1.dtype)

        # Some argument checking
        if node1.is_leaf() != node2.is_leaf():
            raise OutOfBounds
        elif node1.is_leaf() and node2.is_leaf():
            return rounding(operation(node1.value, node2.value))
        elif node1.child_nodes.values()[0].is_leaf():
            node.d = depth = 1
            node, new_offset = dd.create_leaves(node, rounding(operation(node1.to_matrix(outer_offset=offset1), node2.to_matrix(outer_offset=offset2)))[0])
        else:
            offset = {}
            depth = 0
            # looping over the different elements in the base
            # no-offset diagram type?
            if diagram1.offsets == {}:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        operation_elementwise_rec(node1.child_nodes[i], node2.child_nodes[i], None, None)
            else:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        operation_elementwise_rec(node1.child_nodes[i], node2.child_nodes[i],
                                   node1.dtype.to_mat(node1, node1.offsets[i], offset1), node2.dtype.to_mat(node2, node2.offsets[i], offset2))
            depth += 1
            node, new_offset = dd.create_tuple(node, offset)
            node.d = depth
            # because, in all likelihood, the following has to be calculated anyways, calculating it now will
            #  eliminate the need for another recursion through the diagram.
        node.nodes
        node.leaves
        node.__hash__()
        hashmap_of_results[hash_of_current_operation] = [node, new_offset, depth]
        # TODO: the following should work ...
        # node.__hash__()
        # if to_reduce:
        #     if not node.__hash__() in hashtable:
        #         hashtable[node.__hash__()] = node
        #     else:
        #         node = hashtable[node.__hash__()]
        return node, new_offset, depth

    # some argument checking:
    offset_d_1 = diagram1.dtype.null_edge_value if outer_offset is None else outer_offset
    diagram, f_offset, _ = operation_elementwise_rec(diagram1, diagram2, offset_d_1, diagram2.dtype.null_edge_value)
    dd.include_final_offset(diagram, f_offset)
    diagram.reduce()
    return diagram


def sum_over_all(diagram1):
    """
    This function calculates the sum of the diagram
    :param diagram1: The first diagram
    :return: a float
    """
    # The hashmap of the results, to minimize computations
    hashmap_of_results = {}

    # the recursive function
    def sum_rec(node1, offset1):
        # do we know the result already?
        hash_of_current_operation = hash_for_multiple_nodes(node1, node1, offset1, None)
        if hash_of_current_operation in hashmap_of_results:
            return hashmap_of_results[hash_of_current_operation]

        node = diagram1.__class__('', diagram_type=diagram1.dtype)

        # Some argument checking
        if node1.is_leaf():
            return node1.value
        elif node1.child_nodes[node1.child_nodes.keys()[0]].is_leaf():
            return np.sum(node1.to_matrix(outer_offset=offset1))
        else:
            tmp_result = 0
            # looping over the different elements in the base
            # no-offset diagram type?
            if diagram1.offsets == {}:
                for i in range(diagram1.dtype.base):
                    tmp_result += sum_rec(node1.child_nodes[i], None)
            else:
                for i in range(diagram1.dtype.base):
                    tmp_result += sum_rec(node1.child_nodes[i], node1.dtype.to_mat(node1, node1.offsets[i], offset1))
            # because, in all likelihood, the following has to be calculated anyways, calculating it now will
            #  eliminate the need for another recursion through the diagram.
        hashmap_of_results[hash_of_current_operation] = tmp_result
        return tmp_result

    return sum_rec(diagram1, diagram1.dtype.null_edge_value)


def dot_product(diagram_vec_1, diagram_vec_2, dec_digits=-1, outer_offset=None):
    """
    This function computes the dot product (inner product) of two diagrams, each representing a vector.
    :param diagram_vec_1:
    :param diagram_vec_2:
    :return:
    """
    if diagram_vec_1.d != diagram_vec_2.d:
        raise OutOfBounds
    return sum_over_all(multiply_elementwise(diagram_vec_1, diagram_vec_2, dec_digits, outer_offset))


def multiply_matrix_by_column_vector(diagram1, diagram2, precision=-1, outer_offset_1=None, outer_offset_2=None, final_offset=False):
    """
    This function multiplies two diagram elementwise
    (i.e. like numpy.multiply(d1, d2) or d1.*d2 in Matlab)
    This method raises an out of bounds diagram exception if the two diagrams' sizes do not match.
    :param diagram1: The first diagram
    :param diagram2: The second diagram
    :param operation: The operation to be performed as a function of the numpy package, i.e. np.multiply
    :return: a diagram (a new object) of the elementwise multiplication
    """
    # some argument checking
    node_mat_offset_1 = diagram1.dtype.null_edge_value if outer_offset_1 is None else outer_offset_1
    node_mat_offset_2 = diagram1.dtype.null_edge_value if outer_offset_2 is None else outer_offset_1

    # The hashmap of the results, to minimize computations
    hashmap_of_results = {}
    dd = diagram1.dtype()

    # the recursive function
    def multiply_matrix_by_column_vector_rec(node_mat, node_vec, offset_mat, offset_vec):
        # do we know the result already?
        hash_of_current_operation = hash_for_multiple_nodes(node_mat, node_vec, offset_mat, offset_vec)
        if hash_of_current_operation in hashmap_of_results:
            return hashmap_of_results[hash_of_current_operation]

        node = diagram1.__class__('', diagram_type=diagram1.dtype)

        # Are we selecting rows, or performing the dot product?
        if node_mat.d == node_vec.d + 1:
            node.d = depth = 1
            leaf_values = []
            if diagram1.offsets == {}:
                for i in range(dd.base):
                    leaf_values.append(dot_product(node_mat.child_nodes[i], node_vec, precision, dd.to_mat(
                        node_mat.child_nodes[i], None, None)))
            else:
                for i in range(dd.base):
                    leaf_values.append(dot_product(node_mat.child_nodes[i], node_vec, precision, dd.to_mat(
                        node_mat.child_nodes[i], node_mat.offsets[i], offset_mat)))
            node, new_offset = dd.create_leaves(node, np.array(leaf_values))
        # selecting rows
        else:
            offset = {}
            depth = 0
            # looping over the different elements in the base
            # no-offset diagram type?
            if diagram1.offsets == {}:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiply_matrix_by_column_vector_rec(node_mat.child_nodes[i], node_vec, None, None)
            else:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiply_matrix_by_column_vector_rec(node_mat.child_nodes[i], node_vec,
                                   node_mat.dtype.to_mat(node_mat, node_mat.offsets[i], offset_mat), None)
            depth += 1
            node, new_offset = dd.create_tuple(node, offset)
            node.d = depth
            # because, in all likelihood, the following has to be calculated anyways, calculating it now will
            #  eliminate the need for another recursion through the diagram.
        node.nodes
        node.leaves
        node.__hash__()
        hashmap_of_results[hash_of_current_operation] = [node, new_offset, depth]
        # TODO: the following should work ...
        # node.__hash__()
        # if to_reduce:
        #     if not node.__hash__() in hashtable:
        #         hashtable[node.__hash__()] = node
        #     else:
        #         node = hashtable[node.__hash__()]
        return node, new_offset, depth

    # matrix multiplication requires the final offset
    if final_offset:
        return multiply_matrix_by_column_vector_rec(diagram1, diagram2, node_mat_offset_1, node_mat_offset_2)
    else:
        diagram, f_offset, _ = multiply_matrix_by_column_vector_rec(diagram1, diagram2, node_mat_offset_1, node_mat_offset_2)
        dd.include_final_offset(diagram, f_offset)
        diagram.reduce()
        return diagram


def multiply(diagram1, diagram2, height_second_argument, precision=-1, to_transpose=True):
    """
    This function multiplies two diagrams elementwise following the logic of matrix multiplication
    (i.e. like numpy.dot(d1, d2) or d1*d2 in Matlab)
    This method raises an out of bounds diagram exception if the two diagrams' sizes do not match.
    :param diagram1: The first diagram
    :param diagram2: The second diagram
    :param precision: The rounding precision
    :param to_transpose:
    :return: a diagram (a new object) of the elementwise multiplication
    """
    # The hashmap of the results, to minimize computations
    hashmap_of_results = {}
    dd = diagram1.dtype()

    # some argument conversion:
    outer_height = np.ceil(np.log10(height_second_argument)/np.log10(dd.base))
    if to_transpose:
        diagram3 = transpose(diagram2, height_second_argument)
    else:
        diagram3 = diagram2

    # the recursive function
    def multiply_rec(node_mat_1, node_mat_2, offset_mat_1, offset_mat_2):
        # do we know the result already?
        hash_of_current_operation = hash_for_multiple_nodes(node_mat_1, node_mat_2, offset_mat_1, offset_mat_2)
        if hash_of_current_operation in hashmap_of_results:
            return hashmap_of_results[hash_of_current_operation]

        node = diagram1.__class__('', diagram_type=diagram1.dtype)
        offset = {}
        depth = 0
        # Are we selecting rows, or performing the dot product?
        if node_mat_2.d == outer_height + 1:
            if diagram1.offsets == {}:
                for i in range(dd.base):
                    node.child_nodes[i], offset[i], depth = multiply_matrix_by_column_vector(
                        node_mat_1, node_mat_2, precision=precision, final_offset=True)
            else:
                for i in range(dd.base):
                    node.child_nodes[i], offset[i], depth = multiply_matrix_by_column_vector(
                        node_mat_1, node_mat_2, precision=precision, outer_offset_2=offset_mat_2, final_offset=True)
        # selecting rows
        else:
            # looping over the different elements in the base
            # no-offset diagram type?
            if diagram1.offsets == {}:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiply_rec(node_mat_1, node_mat_2.child_nodes[i], None, None)
            else:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiply_rec(node_mat_1, node_mat_2.child_nodes[i],
                                   node_mat_1.dtype.to_mat(node_mat_2, node_mat_2.offsets[i], offset_mat_2), None)
        depth += 1
        node, new_offset = dd.create_tuple(node, offset)
        node.d = depth
        # because, in all likelihood, the following has to be calculated anyways, calculating it now will
        #  eliminate the need for another recursion through the diagram.
        node.nodes
        node.leaves
        node.__hash__()
        hashmap_of_results[hash_of_current_operation] = [node, new_offset, depth]
        # TODO: the following should work ...
        # node.__hash__()
        # if to_reduce:
        #     if not node.__hash__() in hashtable:
        #         hashtable[node.__hash__()] = node
        #     else:
        #         node = hashtable[node.__hash__()]
        return node, new_offset, depth

    diagram, f_offset, _ = multiply_rec(diagram1, diagram3, diagram1.dtype.null_edge_value, diagram2.dtype.null_edge_value)
    dd.include_final_offset(diagram, f_offset)
    diagram.reduce()
    return diagram