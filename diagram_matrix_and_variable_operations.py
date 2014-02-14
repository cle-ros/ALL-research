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
def flattenMatrix(matrix):
    if len(matrix.shape) > 2:
        print 'Please reduce matrix to be 2-dimensional. Aborting'
        return
    return matrix

def getReqVars(mat):
    # checking the matrix dimension 
    # 1. getting a binary representation of the shape
    vSize = re.sub("0b0*",'',bin(mat.shape[0]))
    hSize = re.sub("0b0*",'',bin(mat.shape[1]))
    # 2. computing the number of variables required for a matrix of this size
    return [len(vSize)+len(hSize),len(vSize),len(hSize)]

# this function expands a given matrix to be of size 2^n x 2^m, filling the 
# unknown values with the specified nullValue
def expandMatrix2n(matrix,demSize,nullValue):
    mat = np.ones([2**demSize[0],2**demSize[1]]) * nullValue
    mat[0:matrix.shape[0],0:matrix.shape[1]] = matrix
    return mat

# this function computes the names of the boolean variables    
def getVarNames(noVars):
    varNames = []
    for i in range(noVars):
        varNames.append('x'+str(i+1))
    return varNames