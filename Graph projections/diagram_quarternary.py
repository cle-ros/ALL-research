__author__ = 'clemens'

from diagram import Diagram


class QuarternaryDiagram(Diagram):
    base = 4


class MT4DD(MTxDD, QuarternaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class AEV4DD(AEVxDD, QuarternaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class MEV4DD(MEVxDD, QuarternaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class AAEV4DD(AAxEVDD, QuarternaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)