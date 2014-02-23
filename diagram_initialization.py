# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:25:35 2014

@author: clemens
"""


#import matplotlib.pyplot as plt

import numpy as np
import re

# from diagram import Diagram
from node import *
from diagram_matrix_and_variable_operations import *


def get_conjunctions(leaves):
    # convert the list of leaves to a dictionary for the different values
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
    row,col = np.nonzero(mat)
    leaves = []
    indices = []
    # cycling through those indices
    for i in range(row.shape[0]):
        # converting the integer indices to boolean ones
        r = re.sub('0b','',bin(row[i]))
        c = re.sub('0b','',bin(col[i]))
        ind = list('0'*(no_vars[1]-len(r)) +r+ '0'*(no_vars[2]-len(c))+c)
        indices.append(ind)
        # appending the according matrix entry
        leaves.append(matrix[row[i],col[i]])
    return np.array(indices,dtype=int),np.array(leaves)


def sort_variables(indices, variables):
    """
    This method sorts the variables of a given boolean function
    DEPRECATED, should not be used
    :param indices:
    :param variables:
    """
    # the sorting is rather simple - it sorts by the order of variables having
    # one value

    # Sorting the variables by occurrances of 0 or 1, i.e. by "randomness"
    order = np.argsort(np.maximum(np.sum(indices,axis=0),np.sum(np.logical_not(indices),axis=0)))
    variables=np.array(variables)
#    return indices.T[order].T,variables[order],order
    return indices,variables,order


def compute_diagram(diagram, boolMat, leaves, varNames, matSize):
    """
    This function computes a diagram from a given tree-object, a matrix
    containing all paths to non-null values in a boolean form, the according 
    leaves and the name of the variables.
    :param diagram:
    :param boolMat:
    :param leaves:
    :param varNames:
    :param matSize:
    """
    # concatenating the boolean matrix, the leaves and the variable names
    # to one large, and easier to handle, matrix
    tmp_mat = np.append(varNames[None],boolMat,axis=0)
    tmp_mat = np.append(tmp_mat,np.append([-1],leaves,axis=1)[None].T,axis=1)
    root = diagram.add_node(varNames[0], varNames[0])
    # calling the function which builds the tree recursively
    append_nodes_recursively(diagram, tmp_mat, root)
    return root


def append_nodes_recursively(diagram, children, parent):
    """
    This function appends nodes to a given parent recursively, based on the 
    specifications given in the 'children' matrix
    :param diagram:
    :param children:
    :param parent:
    """
    # appending leaves
    if children.shape[1] == 2:
        for i in range(children.shape[0]-1):
            # does the leave exist?
            leaf = object
            if not diagram.has_leaf(children[i+1,1]):
                leaf = diagram.add_leaf(children[i+1,1], np.double(children[i+1,1]))
            else:
                leaf = diagram.leaves[children[i+1,1]]
            # deciding on the edge to the leave (false/pos)
            if children[i+1,0] == '0':
                diagram.add_n_edge(parent, leaf)
            if children[i+1,0] == '1':
                diagram.add_p_edge(parent, leaf)
        return diagram
    # appending nodes
    else:
        # dividing the matrix into the positive and the negative child-submatrices
        pos_children = np.append(children[0,1:][None],children[children[:,0]=='1'][:,1:],axis=0)
        neg_children = np.append(children[0,1:][None],children[children[:,0]=='0'][:,1:],axis=0)
        # creating the 'positive' subdiagram
        if pos_children.shape[0] >1:
            # creating the name of the node, denoted by the path leading to it            
            node_name = children[0,1]+'.'+parent.name
            # adding the node
            new_node = diagram.add_node(node_name, children[0,1])
            # adding the edge
            diagram.add_p_edge(parent, new_node)
            # doing the recursive call
            append_nodes_recursively(diagram, pos_children, new_node)
        # creating the 'negative' subdiagram
        if neg_children.shape[0] >1:
            # creating the name of the node, denoted by the path leading to it
            node_name = children[0,1]+'.-'+parent.name
            # adding the node
            new_node = diagram.add_node(node_name, children[0,1])
            # adding the edge
            diagram.add_n_edge(parent, new_node)
            # doing the recursive call
            append_nodes_recursively(diagram, neg_children, new_node)
        return

def reduceDiagram(tree):
    return

def initialize_diagram(diagram, matrix, null_value):
    # getting the number of required vars
    """
    Initializing the diagram, wrapping around compute_diagram
    :param diagram:
    :param matrix:
    :param null_value:
    :return:
    """
    no_vars = get_req_vars(matrix)
    matrix = expand_matrix2n(matrix,no_vars[1:],null_value)
    # constructing the tree
    # adding the nodes, named x0,x1,...xn, where n is the number of required vars
    indices,leaves = get_non_null_components(matrix,null_value,no_vars)

    var_names = get_var_names(no_vars[0])
    sorted_indices, sorted_variable_names, sorted_order = sort_variables(indices, var_names)
    root = compute_diagram(diagram, sorted_indices, leaves, sorted_variable_names, no_vars)
    
    return root, var_names[:no_vars[1]], var_names[no_vars[1]:]
