# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""
import copy
from node import Node
from diagram_exceptions import OutOfBounds
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

    diagram.dtype.scalar_mult(diagram, scalar)

    return result


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
    from diagram import AEVxDD
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
            node, offset = AEVxDD.create_leaves(node, [n_value, p_value])
            return node, offset, 1
        else:
            # computing the previous sum
            # go_on prevents to much computation for some diagram types where a recursive multiplication is not
            # necessary
            node = node1.create_node()
            node.n, n_offset, depth = elementwise_multiply_diagrams_evbdd_rec(node1.n, node2.n, node1.no + offset1, node2.no + offset2)
            node.p, p_offset, depth = elementwise_multiply_diagrams_evbdd_rec(node1.p, node2.p, node1.po + offset1, node2.po + offset2)
            depth += 1
            node, offset = AEVxDD.create_tuple(node, n_offset, p_offset)
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

    from diagram import MTBDD, EVBDD
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
            matd.d is None
        except AttributeError:
            raise AttributeError
        if matd.d > vecd.d:
            # still selecting rows:
            node = matd.create_node()
            succ = False
            if matd.n:
                n_edge = multiply_bdiagram_by_vector_rec(matd.n, vecd, matd.dtype.collapse_node(matd.no, offset))
                if n_edge:
                    node.n = n_edge
                    depth = n_edge.d + 1
                    succ = True
            if matd.p:
                p_edge = multiply_bdiagram_by_vector_rec(matd.p, vecd, matd.dtype.collapse_node(matd.po, offset))
                if p_edge:
                    node.p = p_edge
                    depth = p_edge.d + 1
                    succ = True
            if succ:
                node.d = depth
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


def hash_for_multiple_nodes(node1, node2):
    """
    This function creates a hash for multiple nodes. This hash is a long int, and can therefore not be used
    for the builtin python __hash__() functions.
    :param node1:
    :param node2:
    :return:
    """
    return hash(str(node1.__hash__())+str(node2.__hash__()))


def multiplication_elementwise(diagram1, diagram2):
    """
    This function multiplies two diagram elementwise
    (i.e. like numpy.multiply(d1, d2) or d1.*d2 in Matlab)
    This method raises an out of bounds diagram exception if the two diagrams' sizes do not match.
    :param diagram1: The first diagram
    :param diagram2: The second diagram
    :return: a diagram (a new object) of the elementwise multiplication
    """
    # The hashmap of the results, to minimize computations
    hashmap_of_results = {}

    # the recursive function
    def multiplication_elementwise_rec(node1, node2, offset1, offset2):
        # do we know the result already?
        hash_of_current_operation = hash_for_multiple_nodes(node1, node2)
        if hash_of_current_operation in hashmap_of_results:
            return hashmap_of_results[hash_of_current_operation]

        node = diagram1.__class__('', diagram_type=diagram1.__class__)

        # Some argument checking
        if node1.is_leaf() != node2.is_leaf():
            raise OutOfBounds
        elif node1.child_nodes[node1.child_nodes.keys()[0]].is_leaf():
            node, new_offset = diagram1.dtype.create_leaves(node, np.multiply(node1.to_matrix(outer_offset=offset1), node2.to_matrix(outer_offset=offset2)))
            node.d = depth = 1
        else:
            offset = {}
            depth = 0
            # looping over the different elements in the base
            # no-offset diagram type?
            if diagram1.offsets == {}:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiplication_elementwise_rec(node1.child_nodes[i], node2.child_nodes[i], None, None)
            else:
                for i in range(diagram1.dtype.base):
                    node.child_nodes[i], offset[i], depth = \
                        multiplication_elementwise_rec(node1.child_nodes[i], node2.child_nodes[i],
                                   node1.dtype.to_mat(node1, offset1)[i], node1.dtype.to_mat(node1, offset1)[i])
            depth += 1
            node, new_offset = node.dtype.create_tuple(node, offset)
            node.d = depth
            # because, in all likelihood, the following has to be calculated anyways, calculating it now will
            #  eliminate the need for another recursion through the diagram.
        node.nodes
        node.leaves
        node.__hash__()
        hashmap_of_results[hash_of_current_operation] = node
        # TODO: the following should work ...
        # node.__hash__()
        # if to_reduce:
        #     if not node.__hash__() in hashtable:
        #         hashtable[node.__hash__()] = node
        #     else:
        #         node = hashtable[node.__hash__()]
        return node, new_offset, depth

    diagram, f_offset, _ = multiplication_elementwise_rec(diagram1, diagram2, diagram1.dtype.null_edge_value, diagram2.dtype.null_edge_value)
    diagram.dtype.include_final_offset(diagram, f_offset)
    diagram.reduce()
    return diagram

    # def sum(self):
    #     """
    #     This method returns the matrix represented by the diagram
    #     :param rows:
    #     :param cropping:
    #     """
    #     import numpy as np
    #
    #     # covering zero-matrices
    #     if self.child_nodes == {}:
    #         return self.null_value
    #
    #     def sum_rec(node, offset):
    #         # making sure the node exists
    #         if not node:
    #             return 0
    #         # checking whether the node is a leaf
    #         elif node.is_leaf():
    #             return np.sum(node.dtype.to_mat(node, offset))
    #         else:
    #             tmp_result = 0
    #             # the recursive call
    #             # checking for the kind of diagram. MTxxx?
    #             if self.offsets == {}:
    #                 for edge_name in node.child_nodes:
    #                     tmp_result += sum_rec(node.child_nodes[edge_name], node.dtype.to_mat(node, 0, 0))
    #             # or edge-value dd?
    #             else:
    #                 for edge_name in node.child_nodes:
    #
