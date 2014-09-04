# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""
import numpy as np

from pyDD.diagram.exceptions import UnmatchingDiagramsException


def scalar_multiply_diagram(diagram, scalar):
    """
    This function multiplies all leaves_array in the diagram by a skalar
    :param diagram:
    :param scalar:
    :return:
    :type diagram: node.Node
    :type scalar: float
    :rtype: node.Node
    """
    diagram.reinitialize()
    # result = copy.deepcopy(diagram)

    diagram.dtype.scalar_mult(diagram, scalar)

    return diagram


def exchange_variable_order_with_children(node, depth):
    """
    This function exchanges the variable encoded by the subdiagrams at a specified level with the following variable,
    i.e. the variable encoded by the child-subdiagrams
    :param node: the main diagram
    :param depth: the level of subdiagrams encoding the variable to be exchanged with its successor
    :return: the same node, with a different variable ordering
    :type node: node.Node
    :type depth: int
    :rtype: node.Node
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
    :return: the node, transposed
    :type node: node.Node
    :type height: int
    :type to_copy: bool
    :rtype: node.Node
    """
    if to_copy:
        import copy
        node_t = copy.deepcopy(node)
    else:
        node_t = node
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


def multiply_elementwise(diagram1, diagram2, offset_1=None, offset_2=None, approximation_precision=0,
                         decimal_precision=-1, in_place=False, matrix_multiplication=False):
    """
    This function multiplies two diagram elementwise
    (i.e. like numpy.multiply(d1, d2) or d1.*d2 in Matlab)
    This method raises an out of bounds diagram exception if the two diagrams' sizes do not match.
    :param diagram1: The first diagram
    :param diagram2: The second diagram
    :return: a diagram (a new object) of the elementwise multiplication
    :type diagram1: node.Node
    :type diagram2: node.Node
    :type offset_1: numpy.ndarray
    :type offset_2: numpy.ndarray
    :type approximation_precision: int
    :type decimal_precision: int
    :type in_place: int
    :type matrix_multiplication: bool
    :rtype: node.Node
    """
    import numpy
    return operation_elementwise(diagram1, diagram2, numpy.multiply, offset_1, offset_2, approximation_precision,
                                 decimal_precision, in_place, matrix_multiplication)


def addition_elementwise(diagram1, diagram2, offset_1=None, offset_2=None, approximation_precision=0,
                         decimal_precision=-1):
    """
    This function provides normal matrix addition.
    :param diagram1:
    :param diagram2:
    :return:
    :type diagram1: node.Node
    :type diagram2: node.Node
    :type offset_1: numpy.ndarray
    :type offset_2: numpy.ndarray
    :type approximation_precision: int
    :type decimal_precision: int
    :rtype: node.Node
    """
    import numpy
    return operation_elementwise(diagram1, diagram2, numpy.add, offset_1, offset_2, approximation_precision,
                                 decimal_precision)


def fill_up_with_child_levels(floating_node):
    """
    This function creates child nodes until the depth indicated in the node is correct.
    :param floating_node:
    :return:
    :type floating_node: node.Node
    :rtype: node.Node
    """
    node = floating_node
    if node.d != 1 and node.child_nodes.values()[0].is_leaf():
        while node.d != 1:
            tmp_children = node.child_nodes
            tmp_offsets = node.offsets
            new_child_node = node.create_node(depth=node.d-1)
            new_child_node.child_nodes = tmp_children
            new_child_node.offsets = tmp_offsets
            node = new_child_node
        return floating_node
    while not node.is_leaf():
        if node.d == 1:
            new_child_node = node.create_leaf(node.dtype.null_leaf_value)
        else:
            new_child_node = node.create_node(depth=node.d-1)
        for i in range(node.dtype.base):
            node.child_nodes[i] = new_child_node
            if node.offsets != {}:
                node.offsets[i] = node.dtype.null_edge_value
        node = new_child_node
    return floating_node


def operation_elementwise(diagram1, diagram2, operation, offset_1, offset_2, approximation_precision=0,
                          decimal_precision=-1, in_place=False, matrix_multiplication=False):
    """
    This function multiplies two diagram elementwise
    (i.e. like numpy.multiply(d1, d2) or d1.*d2 in Matlab)
    This method raises an out of bounds diagram exception if the two diagrams' sizes do not match.
    :param diagram1: The first diagram
    :param diagram2: The second diagram
    :param operation: The operation to be performed as a function of the numpy package, i.e. np.multiply
    :param offset_1:
    :param offset_2:
    :param approximation_precision:
    :param decimal_precision:
    :param in_place: For performance and memory reasons, the computation can be done in place of one of the args (1/2)
    :param matrix_multiplication: is the computation part of matrix multiplication? (Affects the approx-precision)
    :return: a diagram (a new object) of the elementwise multiplication
    :type diagram1: node.Node
    :type diagram2: node.Node
    :type operation: function
    :type offset_1: numpy.ndarray
    :type offset_2: numpy.ndarray
    :type approximation_precision: int
    :type decimal_precision: int
    :type in_place: int
    :type matrix_multiplication: bool
    :rtype: node.Node
    """
    arg_1_precision = 2 * approximation_precision if matrix_multiplication else approximation_precision
    # The hashmap of the results, to minimize computations
    hashmap_of_results = {}
    dd = diagram1.dtype
    if decimal_precision != -1:
        def rounding(array):
            return np.round(array, decimal_precision)
    else:
        def rounding(array):
            return array

    # the recursive function
    def operation_elementwise_rec(node1, node2, offset1, offset2):
        """
        The recursive sub-function
        :param node1:
        :param node2:
        :param offset1:
        :param offset2:
        :return:
        :type node1: node.Node
        :type node2: node.Node
        :type offset1: numpy.ndarray
        :type offset2: numpy.ndarray
        """
        # do we know the result already?
        hash_of_current_operation = hash_for_multiple_nodes(node1, node2, offset1, offset2)
        if hash_of_current_operation in hashmap_of_results:
            return hashmap_of_results[hash_of_current_operation]

        node = diagram1.__class__('', diagram_type=diagram1.dtype)

        # Some argument checking
        if node1.is_leaf() != node2.is_leaf():
            raise UnmatchingDiagramsException
        elif node1.is_leaf() and node2.is_leaf():
            return rounding(operation(node1.value, node2.value))
        elif node1.d == 1 or node2.d == 2*approximation_precision:
            if in_place == 1:
                node = node1
                depth = node1.d
            elif in_place == 2:
                node = node2
                depth = node2.d
            else:
                node.d = depth = 1
            if approximation_precision == 0 or matrix_multiplication:
                node, new_offset = dd.create_leaves(node, rounding(operation(
                    node1.to_matrix(outer_offset=offset1, approximation_precision=arg_1_precision),
                    node2.to_matrix(outer_offset=offset2, approximation_precision=approximation_precision)))[0])
            else:
                node, new_offset = dd.create_tuple(node, rounding(
                    operation(node1.to_matrix(outer_offset=offset1,
                                              approximation_precision=arg_1_precision),
                              node2.to_matrix(outer_offset=offset2,
                                              approximation_precision=approximation_precision)))[0])
                fill_up_with_child_levels(node)
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
                                                  node1.dtype.to_mat(node1, node1.offsets[i], offset1),
                                                  node2.dtype.to_mat(node2, node2.offsets[i], offset2))
            depth += 1
            node, new_offset = dd.create_tuple(node, offset)
            node.d = depth
        # because, in all likelihood, the following has to be calculated anyways, calculating it now will
        #  eliminate the need for another recursion through the diagram.
        node.reinitialize()
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
    offset_d_1 = diagram1.dtype.null_edge_value if offset_1 is None else offset_1
    offset_d_2 = diagram2.dtype.null_edge_value if offset_2 is None else offset_2
    diagram, f_offset, _ = operation_elementwise_rec(diagram1, diagram2, offset_d_1, offset_d_2)
    dd.include_final_offset(diagram, f_offset)
    return diagram


def sum_over_all(diagram1, approximation_precision=0):
    """
    This function calculates the sum of the diagram
    :param diagram1: The first diagram
    :param approximation_precision: the precision of the approximation level
    :return: a float
    :type diagram1: node.Node
    :type approximation_precision: int
    :rtype: float
    """
    # The hashmap of the results, to minimize computations
    hashmap_of_results = {}

    # the recursive function
    def sum_rec(node1, offset1):
        """
        The recursive sub-function
        :param node1:
        :param offset1:
        :return:
        :type node1: node.Node
        :type offset1: numpy.ndarray
        """
        # do we know the result already?
        hash_of_current_operation = hash_for_multiple_nodes(node1, node1, offset1, None)
        if hash_of_current_operation in hashmap_of_results:
            return hashmap_of_results[hash_of_current_operation]

        # Some argument checking
        if node1.is_leaf():
            return node1.value
        elif node1.d == approximation_precision:
            return node1.dtype.to_mat(next(iter(node1.leaves)), loffset=node1.dtype.null_edge_value,
                                      goffset=offset1)
        elif node1.child_nodes[node1.child_nodes.keys()[0]].is_leaf():
            tmp_result = np.sum(node1.to_matrix(outer_offset=offset1))
            hashmap_of_results[hash_of_current_operation] = tmp_result
            return tmp_result
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


def dot_product(diagram_vec_1, diagram_vec_2, offset_1=None, offset_2=None, approximation_precision=0,
                decimal_precision=-1, in_place=False, matrix_multiplication=False):
    """
    This function computes the dot product (inner product) of two diagrams, each representing a vector.
    :param diagram_vec_1: the first vector
    :param diagram_vec_2: the second vector
    :param offset_1: the first vector's offset
    :param offset_2: the second vector's offset
    :param approximation_precision: the level of approximation (in excluded vars)
    :param decimal_precision: the level of decimal precision
    :param in_place: shall some calculation be done in place? if so, specify '1' or '2' for 1st or 2nd diagram
    :param matrix_multiplication: is this function call part of a matrix multiplication?
    :return:
    :type diagram_vec_1: node.Node
    :type diagram_vec_2: node.Node
    :type offset_1: numpy.ndarray
    :type offset_2: numpy.ndarray
    :type approximation_precision: int
    :type decimal_precision: int
    :type in_place: int
    :type matrix_multiplication: bool
    """
    if diagram_vec_1.d - approximation_precision != diagram_vec_2.d:
        raise UnmatchingDiagramsException
    return sum_over_all(multiply_elementwise(diagram_vec_1, diagram_vec_2, offset_1, offset_2, approximation_precision,
                                             decimal_precision, in_place, matrix_multiplication),
                        approximation_precision)


def multiply_matrix_by_column_vector(diagram_mat, diagram_vec, outer_offset_mat=None, outer_offset_vec=None,
                                     approximation_precision=0, decimal_precision=-1, final_offset=False,
                                     in_place=False):
    """
    This function multiplies a diagram representing a matrix, and a diagram representing a column vector
    (i.e. like numpy.dot(d1, d2) or d1*d2 in Matlab)
    This method raises an out of bounds diagram exception if the two diagrams' sizes do not match.
    :param diagram_mat: The matrix
    :param diagram_vec: The vector
    :param approximation_precision: The number of variables (i.e. levels in the diagram) to be ignored
    :param decimal_precision: The rounding precision (in decimal digits)
    :param outer_offset_mat: The beginning offset of the matrix diagram
    :param outer_offset_vec: The beginning offset of the vector diagram
    :param final_offset: a boolean - should the final offset be included, or returned?
    :return: a diagram (a new object) of the elementwise multiplication; the final offset (if specified)
    :type diagram_mat: node.Node
    :type diagram_vec: node.Node
    :type outer_offset_mat: numpy.ndarray
    :type outer_offset_vec: numpy.ndarray
    :type approximation_precision: int
    :type decimal_precision: int
    :type final_offset: bool
    :type in_place: int
    :rtype: node.Node
    """
    # some argument checking
    node_mat_offset = diagram_mat.dtype.null_edge_value if outer_offset_mat is None else outer_offset_mat
    node_vec_offset = diagram_vec.dtype.null_edge_value if outer_offset_vec is None else outer_offset_vec

    # The hashmap of the results, to minimize computations
    hashmap_of_results = {}
    dd = diagram_mat.dtype

    # the recursive function
    def multiply_matrix_by_column_vector_rec(node_mat, node_vec, offset_mat, offset_vec):
        """
        The recursive sub-function
        :param node_mat:
        :param node_vec:
        :param offset_mat:
        :param offset_vec:
        :return:
        :type node_mat: node.Node
        :type node_vec: node.Node
        :type offset_mat: numpy.ndarray
        :type offset_vec: numpy.ndarray
        :rtype: [node.Node, numpy.ndarray, int]
        """
        # do we know the result already?
        hash_of_current_operation = hash_for_multiple_nodes(node_mat, node_vec, offset_mat, offset_vec)
        if hash_of_current_operation in hashmap_of_results:
            return hashmap_of_results[hash_of_current_operation]

        node = diagram_mat.__class__('', diagram_type=diagram_mat.dtype)

        # Are we selecting rows, or performing the dot product?
        if node_mat.d < node_vec.d:
            from pyDD.diagram.exceptions import UnmatchingDiagramsException
            raise UnmatchingDiagramsException
        elif node_mat.d - 1 == node_vec.d + approximation_precision:
            node.d = depth = 1 + approximation_precision
            leaf_values = []
            if diagram_mat.offsets == {}:
                for i in range(dd.base):
                    leaf_values.append(dot_product(node_mat.child_nodes[i], node_vec, dd.to_mat(
                        node_mat.child_nodes[i]), approximation_precision=approximation_precision,
                        decimal_precision=decimal_precision, in_place=in_place, matrix_multiplication=True))
            else:
                for i in range(dd.base):
                    leaf_values.append(dot_product(node_mat.child_nodes[i], node_vec, dd.to_mat(
                        node_mat.child_nodes[i], node_mat.offsets[i], offset_mat), offset_vec,
                        approximation_precision, decimal_precision, in_place, matrix_multiplication=True))
            node, new_offset = dd.create_leaves(node, np.array(leaf_values))
            if approximation_precision != 0:
                fill_up_with_child_levels(node)
        # selecting rows
        else:
            offset = {}
            depth = 0
            # looping over the different elements in the base
            # no-offset diagram type?
            if diagram_mat.offsets == {}:
                for i in range(diagram_mat.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiply_matrix_by_column_vector_rec(node_mat.child_nodes[i], node_vec, None, None)
            else:
                for i in range(diagram_mat.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiply_matrix_by_column_vector_rec(node_mat.child_nodes[i], node_vec,
                                                             node_mat.dtype.to_mat(node_mat, node_mat.offsets[i],
                                                                                   offset_mat), offset_vec)
            depth += 1
            node, new_offset = dd.create_tuple(node, offset)
            node.d = depth
            # because, in all likelihood, the following has to be calculated anyways, calculating it now will
            #  eliminate the need for another recursion through the diagram.
        node.reinitialize()
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
        return multiply_matrix_by_column_vector_rec(diagram_mat, diagram_vec, node_mat_offset, node_vec_offset)
    else:
        diagram, f_offset, _ = multiply_matrix_by_column_vector_rec(diagram_mat, diagram_vec, node_mat_offset,
                                                                    node_vec_offset)
        dd.include_final_offset(diagram, f_offset)
        diagram.reduce()
        return diagram


def push_approximation_columns(diagram, diagram_height, approximation_precision):
    """
    This function reorders the diagram's variables to fit the approximation precision
    :param diagram: The node to be transformed
    :param diagram_height: the height of the matrix represented by the diagram
    :param approximation_precision: the approximation level
    :return:
    :type diagram: node.Node
    :type diagram_height: int
    :type approximation_precision: int
    :rtype : node.Node
    """
    height = np.ceil(np.log10(diagram_height)/np.log10(diagram.dtype.base))
    # creating the array of exchanges
    exchange_order = []
    for i in range(approximation_precision):
        exchange_order += range(height-i, diagram.d-approximation_precision-i, 1)
    for i in exchange_order:
        exchange_variable_order_with_children(diagram, i)
    diagram.in_order = False
    diagram.exchange_order = exchange_order
    return diagram


def reset_variable_order(diagram):
    """
    This function reorders the diagram to its original variable order
    :param diagram:
    :return:
    :type diagram: node.Node
    :rtype: node.Node
    """
    if not diagram.in_order:
        for i in reversed(diagram.exchange_order):
            exchange_variable_order_with_children(diagram, i)
        diagram.in_order = True
        diagram.exchange_order = []
    return diagram


def multiply(diagram1, diagram2, height_second_argument, approximation_precision=0, decimal_precision=-1,
             to_transpose=True, in_place=0, exchange_order='smart'):
    """
    This function multiplies two diagrams elementwise following the logic of matrix multiplication
    (i.e. like numpy.dot(d1, d2) or d1*d2 in Matlab)
    This method raises an out of bounds diagram exception if the two diagrams' sizes do not match.
    :param diagram1: The first diagram
    :param diagram2: The second diagram
    :param height_second_argument: The height of the matrix (required for the correct row/column selection logic)
    :param approximation_precision: The number of variables (depth in the diagram) to be ignored
    :param decimal_precision: The rounding precision
    :param to_transpose:
    :param in_place:
    :param exchange_order:
    :return: a diagram (a new object) of the elementwise multiplication
    :type diagram1: node.Node
    :type diagram2: node.Node
    :type height_second_argument: int
    :type approximation_precision: int
    :type decimal_precision: int
    :type to_transpose: bool
    :type in_place: int
    :type exchange_order: str
    :rtype: node.Node
    """
    # The hashmap of the results, to minimize computations
    hashmap_of_results = {}
    dd = diagram1.dtype

    # some argument conversion:
    outer_height = np.ceil(np.log10(height_second_argument)/np.log10(dd.base))
    if to_transpose:
        diagram3 = transpose(diagram2, height_second_argument)
    else:
        diagram3 = diagram2
    # pushing the horizontal approximation variables in the front (variable reordering logic)
    if approximation_precision > 0:
        if exchange_order == 'smart':
            if not diagram1.in_order:
                push_approximation_columns(diagram1, height_second_argument, approximation_precision)
            if not diagram3.in_order:
                push_approximation_columns(diagram3, height_second_argument, approximation_precision)
        elif exchange_order == 'force':
            push_approximation_columns(diagram1, height_second_argument, approximation_precision)
            push_approximation_columns(diagram3, height_second_argument, approximation_precision)

    # the recursive function
    def multiply_rec(node_mat_1, node_mat_2, offset_mat_1, offset_mat_2):
        """
        The recursive sub-function
        :param node_mat_1:
        :param node_mat_2:
        :param offset_mat_1:
        :param offset_mat_2:
        :return:
        :type node_mat_1: node.Node
        :type node_mat_2: node.Node
        :type offset_mat_1: numpy.ndarray
        :type offset_mat_2: numpy.ndarray
        """
        # do we know the result already?
        hash_of_current_operation = hash_for_multiple_nodes(node_mat_1, node_mat_2, offset_mat_1, offset_mat_2)
        if hash_of_current_operation in hashmap_of_results:
            return hashmap_of_results[hash_of_current_operation]

        # Are we selecting rows, or performing the dot product?
        if node_mat_1.d - approximation_precision < outer_height:
            raise UnmatchingDiagramsException
        elif node_mat_1.d - approximation_precision == outer_height:
            if diagram1.offsets == {}:
                node, new_offset, depth = \
                    multiply_matrix_by_column_vector(node_mat_2, node_mat_1,
                                                     approximation_precision=approximation_precision,
                                                     decimal_precision=decimal_precision, final_offset=True,
                                                     in_place=in_place)
            else:
                node, new_offset, depth = \
                    multiply_matrix_by_column_vector(node_mat_2, node_mat_1, outer_offset_vec=offset_mat_1,
                                                     approximation_precision=approximation_precision,
                                                     decimal_precision=decimal_precision, final_offset=True,
                                                     in_place=in_place)
        # selecting rows
        else:
            node = diagram1.__class__('', diagram_type=diagram1.dtype)
            offset = {}
            depth = 0
            # looping over the different elements in the base
            # no-offset diagram type?
            if diagram1.offsets == {}:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiply_rec(node_mat_1.child_nodes[i], node_mat_2, None, None)
            else:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiply_rec(node_mat_1.child_nodes[i], node_mat_2,
                                     node_mat_1.dtype.to_mat(node_mat_1, node_mat_1.offsets[i], offset_mat_1), None)
            node, new_offset = dd.create_tuple(node, offset)
            depth += 1
        node.d = depth
        # because, in all likelihood, the following has to be calculated anyways, calculating it now will
        #  eliminate the need for another recursion through the diagram.
        node.reinitialize()
        hashmap_of_results[hash_of_current_operation] = [node, new_offset, depth]
        # TODO: the following should work ...
        # node.__hash__()
        # if to_reduce:
        #     if not node.__hash__() in hashtable:
        #         hashtable[node.__hash__()] = node
        #     else:
        #         node = hashtable[node.__hash__()]
        return node, new_offset, depth

    diagram, f_offset, _ = multiply_rec(diagram1, diagram3, diagram1.dtype.null_edge_value,
                                        diagram3.dtype.null_edge_value)
    dd.include_final_offset(diagram, f_offset)
    diagram.reduce()
    return diagram

