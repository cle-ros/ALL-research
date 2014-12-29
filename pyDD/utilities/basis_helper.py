import numpy as np
from pyDD.utilities.general_helper_functions import get_req_vars, mults

__author__ = 'clemens'


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
        # kron_basis = np.kron(basis_matrix, kron_basis)

    return kron_basis


def multiply_field(a, b, basis_power):
    """
    multiplies two integers by rules of field theory, where the field has a size of basis
    :param a: the first argument
    :param b: the second argument
    :param basis_power: the type of basis
    :return: the multiplied value
    """
    if basis_power == 4:
        return mults['quarternary'][1][a, b]
    else:
        return np.remainder(np.multiply(a, b), basis_power)


mults = {
    'quarternary': [4, np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])]
}

adds = {
    'quarternary': [4, np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])]
}