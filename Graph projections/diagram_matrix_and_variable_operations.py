# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:26:54 2014

@author: clemens
"""
import re
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
        print 'Please reduce matrix to be 2-dimensional. Aborting'
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
    vSize = np.ceil(np.log2(mat.shape[0]))
    try:
        hSize = np.ceil(np.log2(mat.shape[1]))
    except IndexError:
        hSize = 0
    dim = max([int(vSize), int(hSize)])
    # 2. computing the number of variables required for a matrix of this size
    return [2 * dim, vSize, hSize]


def expand_matrix2n(matrix, demSize, nullValue):
    """
    this function expands a given matrix to be of size 2^n x 2^m, filling the
    unknown values with the specified nullValue
    :param matrix:
    :param demSize:
    :param nullValue:
    :return:
    """
    mat = np.ones([2**demSize[0], 2**demSize[1]]) * nullValue
    try:
        mat[0:matrix.shape[0], 0:matrix.shape[1]] = matrix
    except IndexError:
        mat[0:matrix.shape[0], 0] = matrix[None]
    return mat


# this function computes the names of the boolean variables    
def get_var_names(noVars):
    """
    This funtion converts a number of required variables to a list of variable names, i.e. x1, x2, ...
    :param noVars:
    :return:
    """
    varNames = []
    for i in range(int(noVars)):
        varNames.append('x'+str(i+1))
    return varNames