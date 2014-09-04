__author__ = 'clemens'

from pyDD.diagram.diagram import Diagram, MTxDD, AEVxDD, MEVxDD, AAxEVDD


class QuarternaryDiagram(Diagram):
    base = 4


class MT4DD(MTxDD, QuarternaryDiagram):
    def __init__(self):
        from pyDD.diagram.node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class AEV4DD(AEVxDD, QuarternaryDiagram):
    def __init__(self):
        from pyDD.diagram.node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class MEV4DD(MEVxDD, QuarternaryDiagram):
    def __init__(self):
        from pyDD.diagram.node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)


class AAEV4DD(AAxEVDD, QuarternaryDiagram):
    def __init__(self):
        from pyDD.diagram.node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)