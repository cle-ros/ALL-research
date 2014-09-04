__author__ = 'Will Dabney, Clemens Rosenbaum'

# Original implementation by Will Dabney; adapted for working with decision diagrams by Clemens Rosenbaum


import numpy
import kronecker as kron
import itertools
from pyrl.basis import trivial


class SpectralBasis(trivial.TrivialBasis):
    """
    Spectral Analysis based value approximation.
    """

    def __init__(self, nvars, ranges, order=2):
        nterms = pow(order + 1.0, nvars)
        self.numTerms = nterms
        self.order = order
        self.ranges = numpy.array(ranges)
        # self.basis = numpy.array([[2, 0, 0], [2, 1, 0], [2, 2, 2]])
        self.basis = numpy.array([[1, 0, 0], [1, 1, 1], [1, 2, 1]])
        self.transformation_matrix = None
        iter = itertools.product(range(order + 1), repeat=nvars)
        self.multipliers = numpy.array([list(map(int, x)) for x in iter])
        print 'using the spectral basis'

    def computeFeatures(self, features):
        if len(features) == 0:
            return numpy.ones((1,))
        basis_features = numpy.array([self.scale(features[i], i) for i in range(len(features))])
        features_projected = numpy.dot(self.multipliers, basis_features)
        features_transformed = kron.transform(features_projected, self.basis, 4)
        return features_transformed

