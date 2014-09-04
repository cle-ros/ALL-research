__author__ = 'clemens'


from pyDD.diagram.diagram import Diagram, MTxDD, AEVxDD, MEVxDD, AAxEVDD


class RMF3DD(MTxDD):
    def __init__(self):
        MTxDD.__init__(self, 3)

    def transform_basis(self, blocks):
        """
        The transform function to change the basis. For the MTxDDs, this is the identity projection.
        :param blocks: the different sections of data to be transformed
        :return: An unchanged array
        """
        return [blocks[0], blocks[0]+2*blocks[1], blocks[0]+blocks[1]+blocks[2]]


class MT3DD(MTxDD):
    def __init__(self):
        MTxDD.__init__(self, 3)


class AEV3DD(AEVxDD):
    def __init__(self):
        AEVxDD.__init__(self, 3)


class MEV3DD(MEVxDD):
    def __init__(self):
        MEVxDD.__init__(self, 3)


class AAEV3DD(AAxEVDD):
    def __init__(self):
        AAxEVDD.__init__(self, 3)