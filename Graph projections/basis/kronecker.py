__author__ = 'clemens'

import numpy


def expansion(base_matrix, power, group_base=3):
    """
    This function computes the kronecker expansion of a given basis matrix
    :param base_matrix: the base matrix
    :param power: the power to be computed
    :param group_base: the base of the underlying group, defaults to 3
    :return: the expanded base matrix
    :type base_matrix: numpy.ndarray
    :type power: int
    :type group_base: int
    :rtype : numpy.ndarray
    """
    expanded_matrix = numpy.ndarray(base_matrix)
    for i in range(power-1):
        expanded_matrix = numpy.remainder(numpy.kron(expanded_matrix, base_matrix), group_base)
    return expanded_matrix


def fit_input_array(input_array, base_matrix):
    """
    Computes the power required for the expansion
    :param input_array: the input array
    :param base_matrix: the base matrix
    :return: the power required
    """
    flattened_size = numpy.prod(input_array.shape)
    return numpy.ceil(numpy.log10(flattened_size)/numpy.log10(base_matrix.shape[0]))


def transform(input_array, base_matrix, group_base=3):
    """
    Combining the expansion and fit_input_array functions to project the given input array onto the new bases
    :param input_array: the input array
    :param base_matrix: the base matrix
    :param group_base: the base of the underlying group, defaults to 3
    :return: the transformed array
    :type input_array: numpy.ndarray
    :type base_matrix: numpy.ndarray
    :type group_base: int
    :rtype : numpy.ndarray
    """
    conversion_matrix = expansion(base_matrix, fit_input_array(input_array, base_matrix), group_base)
    new_size = numpy.ceil(numpy.log10(input_array.shape)/numpy.log10(group_base))
    return numpy.dot(conversion_matrix, input_array.flatten()).reshape(new_size)