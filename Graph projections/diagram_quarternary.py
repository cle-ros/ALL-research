__author__ = 'clemens'

from diagram import Diagram


class QuarternaryDiagram(Diagram):
    base = 4


class MTQDD(QuarternaryDiagram):
    def __init__(self):
        from node import Node, Leaf
        Diagram.__init__(self, Node, Leaf)

    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates the leaves from the values given, and the node one step up
        """
        for i in range(self.base):
            # if zero_suppressed
            parent_node.child_nodes[i] = self.leaf_type(leaf_values[i], leaf_values[i], diagram_type=self.__class__)
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
        if offset != 0:
            for leaf in node.leaves:
                leaf.value = leaf.value + offset