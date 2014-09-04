__author__ = 'clemens'

from pyDD.diagram.diagram import Diagram, MTxDD, AEVxDD, MEVxDD, AAxEVDD


class MT2DD(MTxDD):
    def __init__(self):
        MTxDD.__init__(self, 2)


class AEV2DD(AEVxDD):
    def __init__(self):
        AEVxDD.__init__(self, 2)


class MEV2DD(MEVxDD):
    def __init__(self):
        MEVxDD.__init__(self, 2)


class AAEV2DD(AAxEVDD):
    """
    This is an implementation of Scott Sanner's affine algebraic DDs
    """
    def __init__(self):
        AAxEVDD.__init__(self, 2)
