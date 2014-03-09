# -*- coding: utf-8 -*-
"""
@author: Clemens Rosenbaum (cgbr@cs.umass.edu)
"""
import numpy as np

#import networks as nx

# in a later version, this function should be used to reduce the matrix
# dimensionality to 2
def flatten_matrix(matrix):
    """
    This function reduces an n-dimensional matrix to a 2-dimensional one
    :param matrix:
    :return:
    """
    if len(matrix.shape) > 2:
        print('Please reduce matrix to be 2-dimensional. Aborting')
        return
    return matrix


def get_req_vars(mat):
    """
    this function creates a list of required variables, in binary form
    :param mat:
    :return:
    """
    # checking the matrix dimension
    # 1. getting a binary representation of the shape
    v_size = np.ceil(np.log2(mat.shape[0]))
    try:
        h_size = np.ceil(np.log2(mat.shape[1]))
    except IndexError:
        h_size = 0
    # 2. computing the number of variables required for a matrix of this size
    return [v_size + h_size, v_size, h_size]


def expand_matrix2n(matrix, demanded_size, null_value):
    """
    this function expands a given matrix to be of size 2^n x 2^m, filling the
    unknown values with the specified nullValue
    :param matrix:
    :param demanded_size:
    :param null_value: The null value
    :return: The full 2^nx2^m matrix
    """
    # TODO: maybe the resulting matrix should be squared, to "fill up" the remaining space with an identity matrix
    mat = np.ones([2**demanded_size[0], 2**demanded_size[1]]) * null_value
    try:
        mat[0:matrix.shape[0], 0:matrix.shape[1]] = matrix
    except IndexError:
        mat[0:matrix.shape[0], 0] = matrix[None]
    return mat


# this function computes the names of the boolean variables    
def get_var_names(no_vars, name):
    """
    This function converts a number of required variables to a list of variable names, i.e. x1, x2, ...
    :param no_vars:
    :return:
    """
    var_names = []
    for i in range(int(no_vars)):
        var_names.append(name + str(i+1))
    return var_names