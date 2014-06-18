__author__ = 'clemens'


from diagram import Diagram, MTxDD, AEVxDD, MEVxDD


class TernaryDiagram(Diagram):
    base = 3


class RMF3DD(MTxDD, TernaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)

    def transform_basis(self, blocks):
        """
        The transform function to change the basis. For the MTxDDs, this is the identity projection.
        :param blocks: the different sections of data to be transformed
        :return: An unchanged array
        """
        return [blocks[0], blocks[0]+2*blocks[1], blocks[0]+blocks[1]+blocks[2]]


class MT3DD(MTxDD, TernaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class AEV3DD(AEVxDD, TernaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class MEV3DD(MEVxDD, TernaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)
