# -*- coding: utf-8 -*-
"""
@author: Clemens Rosenbaum (cgbr@cs.umass.edu)
"""

import numpy as np
import re
from diagram_matrix_and_variable_operations import get_req_vars, get_var_names, expand_matrix2n


def get_conjunctions(leaves):
    # convert the list of leaves_array to a dictionary for the different values
    """
    This method computes the conjuctions defining the formula
    :param leaves:
    :return:
    """
    formula = {}
    for comp in leaves:
        if not comp[1] in formula:
            formula[comp[1]] = []
        formula[comp[1]].append(comp[0])
    return formula


def get_non_null_components(matrix, null_value, no_vars):
    """
    This function gets all indices for all non-null values inside of a given
    matrix
    :param matrix:
    :param null_value:
    :param no_vars:
    """
    mat = matrix - np.ones(matrix.shape)*null_value
    # getting all non-null indices
    # row, col = np.nonzero(mat)
    # print row
    # print col
    row = np.arange(0, mat.shape[0], 1)
    col = np.arange(0, mat.shape[1], 1)
    row, col = [np.tile(row, len(col)), np.repeat(col, len(row))]
    print row
    print col
    leaves = []
    indices = []
    is_vector = no_vars[2] == 0
    # cycling through those indices
    for i in range(row.shape[0]):
        # converting the integer indices to boolean ones
        r = re.sub('0b', '', bin(row[i]))
        if is_vector:
            ind = list('0'*(no_vars[1]-len(r)) + r)
        else:
            c = re.sub('0b', '', bin(col[i]))
            ind = list('0'*(no_vars[1]-len(r)) + r + '0'*(no_vars[2]-len(c))+c)
        indices.append(ind)
        # appending the according matrix entry
        leaves.append(matrix[row[i], col[i]])
    return np.array(indices, dtype=int), np.array(leaves)


def sort_variables(indices, variables):
    """
    This method sorts the variables of a given boolean function
    DEPRECATED, should not be used
    :param indices:
    :param variables:
    """
    # the sorting is rather simple - it sorts by the order of variables having
    # one value

    # Sorting the variables by occurrences of 0 or 1, i.e. by "randomness"
    order = np.argsort(np.maximum(np.sum(indices, axis=0), np.sum(np.logical_not(indices), axis=0)))
    variables = np.array(variables)
#    return indices.T[order].T,variables[order],order
    return indices, variables, order


def compute_diagram(node, bool_mat, leaves, var_names, mat_size):
    """
    This function computes a diagram from a given tree-object, a matrix
    containing all paths to non-null values in a boolean form, the according 
    leaves_array and the name of the variables.
    :param node:
    :param bool_mat:
    :param leaves:
    :param var_names:
    :param mat_size:
    """
    # concatenating the boolean matrix, the leaves_array and the variable names
    # to one large, and easier to handle, matrix
    # TODO: FINISH UP HERE!
    # creating the matrix for the recursive call
    try:
        tmp_mat = np.append(var_names[None], bool_mat, axis=0)
        tmp_mat = np.append(tmp_mat, np.append([-1], np.array(leaves, dtype='str'), axis=1)[None].T, axis=1)
        append_nodes_recursively(node, tmp_mat, {})
    except ValueError:
        # if the matrix is only of zeros, the diagram representation is the root only.
        # therefore, return the unmodified root.
        return node
    # calling the function which builds the tree recursively
    if node.p:
        node.d = node.p.d + 1
    else:
        node.d = node.n.d + 1
    return node


def append_nodes_recursively(node, child_matrix, found_leaves):
    """
    This function appends nodes to a given parent recursively, based on the 
    specifications given in the 'children' matrix
    :param node:
    :param child_matrix:
    """
    # TODO: begin from the leaves, for a more efficient diagram structure
    # appending leaves_array
    if child_matrix.shape[1] == 2:
        for i in range(child_matrix.shape[0]-1):
            # does the leave exist?
            value = child_matrix[i+1, 1]
            if value in found_leaves:
                leaf = found_leaves[value]
            else:
                leaf = node.leaf_type(child_matrix[i+1, 1], np.double(child_matrix[i+1, 1]))
                found_leaves[value] = leaf
            # deciding on the edge to the leave (false/pos)
            if child_matrix[i+1, 0] == '0':
                node.n = leaf
            if child_matrix[i+1, 0] == '1':
                node.p = leaf
        return leaf
    # appending nodes
    else:
        # dividing the matrix into the positive and the negative child-submatrices
        pos_children = np.append(child_matrix[0, 1:][None], child_matrix[child_matrix[:, 0] == '1'][:, 1:], axis=0)
        neg_children = np.append(child_matrix[0, 1:][None], child_matrix[child_matrix[:, 0] == '0'][:, 1:], axis=0)
        # creating the 'positive' subdiagram
        if pos_children.shape[0] > 1:
            # creating the name of the node, denoted by the create_path leading to it
            node_name = child_matrix[0, 1]+'.'+node.name
            # adding the node
            new_node = type(node)(node_name, child_matrix.shape[1]-2)
            # adding the edge
            node.p = new_node
            # doing the recursive call
            append_nodes_recursively(new_node, pos_children, found_leaves)
        # creating the 'negative' subdiagram
        if neg_children.shape[0] > 1:
            # creating the name of the node, denoted by the create_path leading to it
            node_name = child_matrix[0, 1]+'.-'+node.name
            # adding the node
            new_node = type(node)(node_name, child_matrix.shape[1]-2)
            # adding the edge
            node.n = new_node
            # doing the recursive call
            append_nodes_recursively(new_node, neg_children, found_leaves)


def reduce_diagram(tree):
    return


def initialize_diagram(node, matrix, null_value, var_string='x'):
    # getting the number of required vars
    """
    Initializing the diagram, wrapping around compute_diagram
    :param node:
    :param matrix:
    :param null_value:
    :return:
    """
    #TODO: clean up!
    no_vars = get_req_vars(matrix)
    matrix = expand_matrix2n(matrix, no_vars[1:], null_value)
    # constructing the tree
    # adding the nodes, named x0,x1,...xn, where n is the number of required vars
    indices, leaves = get_non_null_components(matrix, null_value, no_vars)

    var_names = get_var_names(no_vars[0], var_string)
    sorted_indices, sorted_variable_names, sorted_order = sort_variables(indices, var_names)
    compute_diagram(node, sorted_indices, leaves, sorted_variable_names, no_vars)
    return no_vars[1:], node
