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


def get_req_vars(mat, base):
    """
    this function creates a list of required variables, in base^n form
    :param mat: the matrix
    :param base: the basis of the access
    :return:
    """
    # checking the matrix dimension
    import math
    # 1. getting a binary representation of the shape
    v_size = int(np.ceil(math.log(mat.shape[0], base)))
    try:
        h_size = int(np.ceil(math.log(mat.shape[1], base)))
    except IndexError:
        h_size = 0
    # 2. computing the number of variables required for a matrix of this size
    #   should it be an identity or a zero-matrix?
    return [v_size + h_size, v_size, h_size]
    # return [2*max(v_size, h_size), max(v_size, h_size), max(v_size, h_size)]


def expand_matrix_exponential(matrix, demanded_size, null_value, base):
    """
    this function expands a given matrix to be of size base^n x base^m, filling the
    unknown values with the specified nullValue
    :param matrix:
    :param demanded_size:
    :param null_value: The null value
    :param base: the base
    :return: The full 2^nx2^m matrix
    """
    # TODO: maybe the resulting matrix should be squared, to "fill up" the remaining space with an identity matrix
    mat = np.ones([base**demanded_size[0], base**demanded_size[1]]) * null_value
    # mat = np.identity(2**demanded_size[0])
    try:
        mat[0:matrix.shape[0], 0:matrix.shape[1]] = matrix
    except IndexError:
        mat[0:matrix.shape[0], 0] = matrix[None]
    return mat


def kronecker_expansion(basis_matrix, target_mat=None, var=None):
    """
    This function computes the kronecker expansion of the given matrix, either to fit the size of the target matrix
     or to basis^var size.
    :param basis_matrix: the basis of the expansion
    :param target_mat: the target matrix of size (m x n) to be fitted to (excludes var)
    :param var: the required expansions (basis^var) (excludes dim)
    :return: the expanded kronecker basis matrix
    """
    basis = basis_matrix.shape[0]
    # argument checking
    if var is None and not target_mat is None:
        var = get_req_vars(target_mat, basis)[0]
    elif not var is None and target_mat is None:
        pass
    else:
        raise ValueError('Either specify target_mat OR var, not both!')

    # computing the product
    kron_basis = basis_matrix
    for _ in range(var-1):
        kron_basis = np.remainder(np.kron(basis_matrix, kron_basis), basis)

    return kron_basis


def field_multiplication(a, b, basis):
    """
    multiplies two integers by rules of field theory, where the field has a size of basis
    :param a: the first argument
    :param b: the second argument
    :param basis: the type of basis
    :return: the multiplied value
    """
    if basis == 'gf4':
        return mults['gf4'][1][a, b]
    else:
        return np.remainder(a*b, mults['gf4'][0])


mults = {
    'gf4': [4, np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])]
}

adds = {
    'gf4': [4, np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])]
}