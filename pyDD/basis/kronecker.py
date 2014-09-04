__author__ = 'clemens'

import numpy


def expansion(base_matrix, power, group_base=3, normalize=False):
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
    if power == 0:
        return numpy.array([1])
    expanded_matrix = numpy.array(base_matrix)
    for i in range(power-1):
        expanded_matrix = numpy.remainder(numpy.kron(expanded_matrix, base_matrix), group_base)
    if normalize:
        expanded_matrix /= numpy.max(expanded_matrix)
    return expanded_matrix


def fit_input_array(input_array, base_matrix):
    """
    Computes the power required for the expansion
    :param input_array: the input array
    :param base_matrix: the base matrix
    :return: the power required
    """
    flattened_size = numpy.prod(input_array.shape)
    return numpy.int_(numpy.ceil(numpy.log10(flattened_size)/numpy.log10(base_matrix.shape[0])))


def transform(input_array, base_matrix, group_base=3, power=-1):
    """
    Combining the expansion and fit_input_array functions to project the given input array onto the new bases
    :param input_array: the input array
    :param base_matrix: the base matrix
    :param group_base: the base of the underlying group, defaults to 3
    :param power: the power of the basis. Defaults to a the power needed to project the input array
    :return: the transformed array
    :type input_array: numpy.ndarray
    :type base_matrix: numpy.ndarray
    :type group_base: int
    :type power: int
    :rtype : numpy.ndarray
    """
    if power == -1:
        power = fit_input_array(input_array, base_matrix)
    conversion_matrix = expansion(base_matrix, power, group_base, normalize=True)
    input_array_extended = numpy.hstack((input_array.flatten(), numpy.zeros(conversion_matrix.shape[0]-
                                                                            len(input_array.flatten()))))
    return numpy.dot(conversion_matrix, input_array_extended)