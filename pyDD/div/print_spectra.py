__author__ = 'clemens'

import numpy as np
from pyDD.basis.matrices import *
from pyDD.basis.kronecker import expansion


def plot_spectra(basis='RMF3', power=3):
    basis_matrix = rmf3
    if basis == 'GF3':
        basis_matrix = gf3

    total_length = basis_matrix.shape[0]**power
    spectra = []
    spectra_normalized = []
    for i in range(power):
        spectra.append(expansion(basis_matrix, power))



