"""
This file contains functions that are not specific to any solver or domain, but can be used for different
applications.
"""
__author__ = 'Clemens Rosenbaum'

import numpy as np


def machine_limit_exp(var, const, l=-700.0, h=700.0):
    var_mn = np.abs(var)
    var_mx = np.abs(var)
    const_mn = np.min(np.sign(var) * const)
    const_mx = np.max(np.sign(var) * const)
    if np.abs(var) * const_mn < l:
        var_mn = np.abs(l / const_mn)
    if np.abs(var) * const_mx > h:
        var_mx = np.abs(h / const_mx)
    return np.min([var_mn, var_mx, np.abs(var)]) * np.sign(var)


def compute_increase_vector(width, decay, magnitude):
    """
    This function computes a bell-shaped vector; this is useful for allowing for error in the value function.
    :param width: the length of the vector
    :param decay: the decay from the center (in d^i)
    :param magnitude: the amplitude at the center
    :return:
    :type width: int
    :type decay: float
    :type magnitude: float
    :rtype: list
    """
    dr = width if (width % 2) == 1 else width + 1
    vec = np.zeros((dr, ))
    middle = int((dr-1)/2)
    for i in range(0, middle+1, 1):
        vec[middle + i] = magnitude * (decay ** i)
        vec[middle - i] = magnitude * (decay ** i)
    # print 'vec: ', vec
    return vec


def compute_value_index(array, value):
    """
    Given an array and a value, this function computes the array-index being closest to the given number in value.
    :param array: an array (monotone!)
    :param value: the value
    :return:
    :type array: numpy.array
    :type value: float
    :rtype: int
    """
    try:
        return (np.abs(array-np.array(value)[:, 0])).argmin()
    except (TypeError, IndexError):
        return(np.abs(array-value)).argmin()


def add_value_vector(prev_values, add_vector, index):
    """
    This function adds the add_vector to the prev_values vector, s.t. the center of add_vector is at index of prev_val
    :param prev_values:
    :param add_vector:
    :param index:
    :return:
    """
    # index += 1
    def compute_indices(ind, mid, van, val):
        miai = -1 * min(0, ind - mid - 1)
        maai = min(val, van - ind + mid + 1)
        mivi = max(0, ind - mid - 1)
        mavi = min(van, ind + mid)
        return miai, maai, mivi, mavi

    middle = int((len(add_vector) - 1)/2)

    val_ret = np.array(prev_values)
    min_add_index, max_add_index, min_val_index, max_val_index = compute_indices(index,
                                                                                 middle,
                                                                                 len(prev_values),
                                                                                 len(add_vector))

    val_ret[min_val_index:max_val_index] += add_vector[min_add_index: max_add_index]
    return val_ret # / val_ret.__abs__().max()


def combinations(source_set, no_comb):
    """
    This function computes all the combinations of no_comb (different and mutable) elements from the given source_set.
    no_comb can be a list, in which case all combinations of all specified lengths are computed.
    :param source_set:
    :param no_comb:
    :return:
    :type source_set: list
    :type no_comb: list
    :rtype: list
    """
    if isinstance(no_comb, int):
        no_elements = [no_comb]
    else:
        no_elements = no_comb
    import itertools
    elements = []
    for comb in no_elements:
        tmp_elements = [i for i in itertools.combinations(source_set, comb)]
        elements += tmp_elements
    return elements

