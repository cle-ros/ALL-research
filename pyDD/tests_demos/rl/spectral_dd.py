__author__ = 'Will Dabney, Clemens Rosenbaum'

# Original implementation by Will Dabney; adapted for working with decision diagrams by Clemens Rosenbaum

import itertools

import numpy

from pyrl.basis import trivial


class SpectralBasis(trivial.TrivialBasis):
    """
    Spectral Analysis based value approximation.
    """

    def __init__(self, nvars, ranges, order=3):
        nterms = pow(order + 1.0, nvars)
        self.numTerms = nterms
        self.order = order
        self.ranges = numpy.array(ranges)
        self.basis = numpy.array([[1, 0, 0], [1, 2, 0], [1, 1, 1]])
        self.transformation_matrix = None

    def computeFeatures(self, features):
        if len(features) == 0:
            return numpy.ones((1,))
        if self.transformation_matrix is None:
            from basis.kronecker import transform
            self.transformation_matrix = transform(features, self.basis)
        if self.transformation_matrix.shape[0] != len(features):
            features = numpy.hstack((features, numpy.zeros(self.transformation_matrix.shape[0]-len(features))))
        return numpy.dot(self.transformation_matrix, features)

