__author__ = 'clemens'

from diagram import Diagram, MTxDD, AEVxDD, MEVxDD, AAxEVDD


class BinaryDiagram(Diagram):
    base = 2


class MT2DD(MTxDD, BinaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class AEV2DD(AEVxDD, BinaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class MEV2DD(MEVxDD, BinaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class AAEV2DD(AAxEVDD, BinaryDiagram):
    """
    This is an implementation of Scott Sanner's affine algebraic DDs
    """
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)
