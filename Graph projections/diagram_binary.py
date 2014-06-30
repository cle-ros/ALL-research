__author__ = 'clemens'

from diagram import Diagram, MTxDD, AEVxDD, MEVxDD


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


class AABDD(BinaryDiagram):
    """
    This is an implementation of Scott Sanner's affine algebraic DDs
    """
    default_offset = [0, 1]

    def __init__(self):
        """
        The init method
        :rtype : a copy of itself
        """
        from node import BNode, BLeaf
        Diagram.__init__(self, BNode, BLeaf)

    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates terminal nodes given the parent node and the leaf values
        """
        """
        How to decide on the offsets:
        1. the leaf-value @ branch 0 pushes the according add. offset to 0
        2. the multiplicative offset
        2. for max-branch: add. offset + mult. offset = max(leaves)
        3. @ leaf-level: mult. offset = 0
        """
        # TODO: find generalization!!
        import numpy as np
        # creating the leaf object
        parent_node.child_nodes[0] = parent_node.child_nodes[1] = self.leaf_type(0, 0, diagram_type=self.__class__)

        # creating the multiplicative coefficient
        mult_coeff = 1 if (leaf_values[1]-leaf_values[0]) == 0 else 1/(leaf_values[1]-leaf_values[0])

        # creating the offsets
        parent_node.offsets[0] = np.array([0, mult_coeff])
        parent_node.offsets[1] = np.array([(leaf_values[1]-leaf_values[0])*mult_coeff, mult_coeff])
        return parent_node, [leaf_values[0], (1/mult_coeff)]

    def create_tuple(self, node, offset):
        """
        This method defines how AABDDs branch.
        """
        import numpy as np
        # creating the multiplicative coefficient
        # mult_coeff = 1 if (offset[1][0]-offset[0][0]) == 0 else 1/(offset[1][0]-offset[0][0])
        # # for edge in offset:
        # #     try:
        # #         mult_coeff = 1/offset[edge][np.nonzero(offset[edge])[0][0]]
        # #         break
        # #     except IndexError:
        # #         mult_coeff = 1
        #
        # # creating the new offsets
        # node.offsets[0] = np.array([0, mult_coeff*offset[0][1]])
        # node.offsets[1] = np.array([(offset[1][0]-offset[0][0])*mult_coeff, mult_coeff*offset[1][1]])
        # return node, [offset[0][0], (1/mult_coeff)]
        if offset[0][0] == 0:
            # # creating the new offsets
            node.offsets[0] = np.array([0, offset[0][1]])
            node.offsets[1] = np.array([(offset[1][0]-offset[0][0]), offset[1][1]])
            return node, [offset[0][0], 1]
        else:
            node.offsets[0] = np.array([0, offset[0][1]])
            node.offsets[1] = np.array([0, (offset[1][0]/offset[0][0])*offset[1][1]])
            return node, [0, offset[0][0]]

    @staticmethod
    def to_mat(node, loffset, goffset=default_offset):
        """
        The diagram-type specific function to convert nodes to matrices
        """
        goffset = AABDD.default_offset if goffset is None else goffset
        loffset = AABDD.default_offset if loffset is None else loffset
        import numpy as np
        from node import Node, Leaf
        if isinstance(node, Leaf):
            return np.array((goffset[0] + goffset[1]*(loffset[0] + loffset[1]*node.value)))[None]
        elif isinstance(node, Node):
            return goffset[0] + goffset[1]*loffset[0], goffset[1]*loffset[1]
        else:
            raise TypeError

    @staticmethod
    def include_final_offset(node, offset):
        for oindex in node.offsets:
            node.offsets[oindex][0] = offset[0] + offset[1]*node.offsets[oindex][0]
            node.offsets[oindex][1] *= offset[1]

    def add(self, node1, offset):
        """
        This function adds two nodes
        :rtype : array of offsets (e.g. n-offset, p-offset)
        :param node1: the first node
        :param offset: the parent offset
        """
        raise NotImplementedError

    def sum(self, offset):
        """
        the helper function for summing a diagram
        """
        raise NotImplementedError

    def scalar_mult(self, scalar):
        """
        The helper function for scalar multiplication
        """
        raise NotImplementedError

    def mult(self, node):
        """
        The helper method for elementwise multiplication
        """
        raise NotImplementedError

    def collaple_node(self, offset):
        """
        This function "opposes" the create-functions, i.e. it does the reverse, collapsing operation
        """
        raise NotImplementedError

    def get_path(self):
        raise NotImplementedError

    def flatten(self):
        raise NotImplementedError
