__author__ = 'clemens'

from diagram import Diagram


class BinaryDiagram(Diagram):
    base = 2


class MTBDD(BinaryDiagram):
    def __init__(self):
        from node import BNode, BLeaf
        BinaryDiagram.__init__(self, BNode, BLeaf)

    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates the leaves from the values given, and the node one step up
        """
        parent_node.n = self.leaf_type(leaf_values[0], leaf_values[0], diagram_type=MTBDD)
        parent_node.p = self.leaf_type(leaf_values[1], leaf_values[1], diagram_type=MTBDD)
        return parent_node, 0

    def create_tuple(self, node, offset):
        return node, 0

    @staticmethod
    def to_mat(leaf, loffset=None, goffset=None):
        """
        The diagram-type specific function to convert nodes to matrices
        """
        import numpy as np
        if leaf.is_leaf():
            return np.array(leaf.value)[None]
        else:
            return None

    @staticmethod
    def include_final_offset(node, offset):
        """
        This function includes an offset remaining after creating the diagram into the diagram.
        """
        for leaf in node.leaves:
            leaf.value = leaf.value + offset

    @staticmethod
    def add(node1, node2, offset=[0, 0]):
        """
        This function adds two nodes
        """
        from node import Leaf
        if isinstance(node1, Leaf):
            return node1.value + node2.value
        else:
            return 0, 0

    @staticmethod
    def sum(node, offset):
        """
        This function takes a node and an offset to compute the new offset for the children nodes.
        """
        from node import Leaf
        if isinstance(node, Leaf):
            return node.value
        else:
            return 0, 0

    @staticmethod
    def scalar_mult(node, scalar):
        """
        The helper function for scalar multiplication.
        """
        from node import Leaf
        if isinstance(node, Leaf):
            node.value = node.value * scalar
            return node
        else:
            return node

    @staticmethod
    def collapse_node(edge_offset, offset):
        return None

    @staticmethod
    def mult(self, node):
        """
        The helper method for scalar multiplication
        """
        raise NotImplementedError


class EVBDD(BinaryDiagram):
    def __init__(self):
        from node import BNode, BLeaf
        BinaryDiagram.__init__(self, BNode, BLeaf)

    @staticmethod
    def create_leaves(parent_node, leaf_values):
        """
        This function creates the leaves from the values given, and the node one step up
        """
        from node import BLeaf
        parent_node.n = parent_node.p = BLeaf(0, 0, diagram_type=EVBDD)
        parent_node.no = 0
        parent_node.po = leaf_values[1] - leaf_values[0]
        return parent_node, leaf_values[0]

    @staticmethod
    def create_tuple(node, offset):
        """
        Computes the offset for a node, given the offset of its children
        """
        node.no = 0
        node.po = offset[1] - offset[0]
        return node, offset[0]

    @staticmethod
    def collapse_node(edge_offset, offset):
        return edge_offset + offset

    @staticmethod
    def to_mat(node, goffset=0, loffset=0):
        """
        The diagram-type specific function to convert nodes to matrices
        """
        import numpy as np
        from node import Node, Leaf
        if isinstance(node, Leaf):
            return np.array((node.value + goffset))[None]
        elif isinstance(node, Node):
            return loffset + goffset
        else:
            raise TypeError

    @staticmethod
    def include_final_offset(node, offset):
        """
        This function includes an offset remaining after creating the diagram into the diagram.
        """
        for leaf in node.leaves:
            leaf.value = leaf.value + offset

    @staticmethod
    def add(node1, node2, offset=[0, 0]):
        """
        This function adds two nodes, returning the respective offset for n and p
        """
        from node import Leaf
        if isinstance(node1, Leaf):
            return node1.value + node2.value + offset[0] + offset[1]
        else:
            return node1.no + node2.no + offset[0], node1.po + node2.po + offset[1]

    @staticmethod
    def sum(node, offset):
        """
        This function takes an offset and a node to create the two new offsets for the different children
        """
        from node import Leaf
        if isinstance(node, Leaf):
            return node.value + offset
        else:
            return node.no + offset, node.po + offset

    @staticmethod
    def scalar_mult(node, scalar):
        """
        The helper function for scalar multiplication.
        """
        from node import Leaf
        if isinstance(node, Leaf):
            node.value = node.value * scalar
            return node
        else:
            node.no = node.no * scalar
            node.po = node.po * scalar
            return node

    @staticmethod
    def mult(node1, node2, offset=None, n_offset=None, p_offset=None, mtype=None):
        """
        The helper method for scalar multiplication
        """
        if offset is None:
            offset = 0
        if mtype == 'Node':
            return node2.value * offset, offset, True
        elif mtype == 'First':
            return node1.no + offset, node1.po + offset, True
        elif mtype == 'Second':
            return node2.no*offset, node2.po*offset, offset
        raise TypeError
