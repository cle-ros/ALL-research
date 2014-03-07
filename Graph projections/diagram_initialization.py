# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:25:35 2014

@author: clemens
"""


#import matplotlib.pyplot as plt

import numpy as np
import re

# from diagram import Diagram
from diagram_matrix_and_variable_operations import *


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
    row, col = np.nonzero(mat)
    leaves = []
    indices = []
    # cycling through those indices
    for i in range(row.shape[0]):
        # converting the integer indices to boolean ones
        r = re.sub('0b', '', bin(row[i]))
        c = re.sub('0b', '', bin(col[i]))
        ind = list('0'*(no_vars[1]-len(r)) + r + '0'*(no_vars[2]-len(c))+c)
        indices.append(ind)
        # appending the according matrix entry
        leaves.append(matrix[row[i], col[i]])
    return np.array(indices, dtype=int), np.array(leaves)


def compute_diagram(node, bool_mat, leaves, var_names, matSize):
    """
    This function computes a diagram from a given tree-object, a matrix
    containing all paths to non-null values in a boolean form, the according 
    leaves_array and the name of the variables.
    :param node:
    :param bool_mat:
    :param leaves:
    :param var_names:
    :param matSize:
    """
    # concatenating the boolean matrix, the leaves_array and the variable names
    # to one large, and easier to handle, matrix
    print var_names
    print bool_mat
    print var_names.shape
    print bool_mat.shape
    tmp_mat = np.append(var_names[None], bool_mat, axis=0)
    tmp_mat = np.append(tmp_mat, np.append([-1], leaves, axis=1)[None].T, axis=1)
    # calling the function which builds the tree recursively
    append_nodes_recursively(node, tmp_mat, {})
    node.d = node.p.d + 1
    return node


def append_nodes_recursively(node, child_matrix, found_leaves):
    """
    This function appends nodes to a given parent recursively, based on the 
    specifications given in the 'children' matrix
    :param node:
    :param child_matrix:
    """
    # appending leaves_array
    if child_matrix.shape[1] == 2:
        for i in range(child_matrix.shape[0]-1):
            # does the leave exist?
            # leaf = object
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
            # creating the name of the node, denoted by the path leading to it
            node_name = child_matrix[0, 1]+'.'+node.name
            # adding the node
            new_node = type(node)(node_name, child_matrix.shape[1]-2)
            # adding the edge
            node.p = new_node
            # doing the recursive call
            append_nodes_recursively(new_node, pos_children, found_leaves)
        # creating the 'negative' subdiagram
        if neg_children.shape[0] > 1:
            # creating the name of the node, denoted by the path leading to it
            node_name = child_matrix[0, 1]+'.-'+node.name
            # adding the node
            new_node = type(node)(node_name, child_matrix.shape[1]-2)
            # adding the edge
            node.n = new_node
            # doing the recursive call
            append_nodes_recursively(new_node, neg_children, found_leaves)


def reduceDiagram(tree):
    return

def initialize_diagram(node, matrix, null_value):
    # getting the number of required vars
    """
    Initializing the diagram, wrapping around compute_diagram
    :param node:
    :param matrix:
    :param null_value:
    :return:
    """
    no_vars = get_req_vars(matrix)
    matrix = expand_matrix2n(matrix, no_vars[1:], null_value)
    # constructing the tree
    # adding the nodes, named x0,x1,...xn, where n is the number of required vars
    indices, leaves = get_non_null_components(matrix, null_value, no_vars)

    var_names = get_var_names(no_vars[0])
    sorted_indices, sorted_variable_names, sorted_order = sort_variables(indices, var_names)
    compute_diagram(node, sorted_indices, leaves, sorted_variable_names, no_vars)
    return no_vars[1:], node
