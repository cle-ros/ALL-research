# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""


class Diagram:
    """
    This is an interface/abstract class for all different diagram types
    """

    def __init__(self, nt, lt):
        """
        The init method
        :rtype : a copy of itself
        """
        self.node_type = nt
        self.leaf_type = lt
        # self.paths = self.get_path()

    def create_tuple(self, node, offset):
        """
        This method defines the general framework to branch. It relies on specific implementations for the different
         binary diagram types
        """
        raise NotImplementedError

    def to_mat(self, loffset, goffset):
        raise NotImplementedError

    @staticmethod
    def include_final_offset(node, offset):
        raise NotImplementedError

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

    def create_leaves(self, parent_node, leaf_values):
        raise NotImplementedError

    def flatten(self):
        raise NotImplementedError

    def reduce(self, node_o):
        """
        This function reduces a tree, given in node, to the fitting diagram
        :rtype : None - the change will be applied to the argument
        :param node_o : the tree (or diagram) to be reduced
        """
        # initializing a hashtable for all the nodes in the tree
        hashtable = {}
        for node_it in node_o.nodes:
            # storing each node only once in the table
            if not node_it.__hash__() in hashtable:
                hashtable[node_it.__hash__()] = node_it

        def reduce_rec(node):
            """
            The recursive method for the reduction.
            """
            if node.is_leaf():
                return
            for edge in node.child_nodes:
                # replacing the subdiagram with a singular isomorphic one
                node.child_nodes[edge] = hashtable[node.child_nodes[edge].__hash__()]
                # and going down recursively along that subdiagram
                reduce_rec(node.child_nodes[edge])

        # calling the reduction method
        reduce_rec(node_o)
        # reinitializing the diagram
        node_o.reinitialize_leaves()
        node_o.reinitialize_nodes()
        return node_o

    def transform_basis(self, values):
        """
        The transform function to change the basis. For the MTxDDs, this is the identity projection.
        :param blocks: the different sections of data to be transformed
        :return: An unchanged array
        """
        block_len = len(values)/self.base
        blocks = [values[i*block_len:(i+1)*block_len] for i in range(self.base)]
        return blocks

    def create(self, matrix, null_value, to_reduce=True, dec_digits=-1):
        """
        this function creates a diagram of the specified type of the given matrix
        :param matrix:      The data to be represented
        :param null_value:  The null value (for *-suppressed DDs)
        :param to_reduce:   Whether the tree shall be represented as a diagram
        :param dec_digits:  The number of decimal digits to round to
        """
        from matrix_and_variable_operations import expand_matrix_exponential, get_req_vars
        # get the required number of vars
        no_vars = get_req_vars(matrix, self.base)
        # expand the matrix to be of size 2^nx2^m
        matrix = expand_matrix_exponential(matrix, no_vars[1:], null_value, self.base)
        # get the not-suppressed values
        leaves = matrix.flatten()
        # should the values be rounded to increase compression?
        if dec_digits != -1:
            import numpy
            leaves = numpy.round(leaves, dec_digits)

        def create_diagram_rec(values):
            """
            The recursive function
            """
            node = self.node_type('', diagram_type=self.__class__)
            entry_length = len(values)/self.base
            if entry_length == 1:
                node, new_offset = self.create_leaves(node, values)
                node.d = depth = 1
            else:
                # somewhere around here the create_tuple has to be used.
                offset = {}
                depth = 0
                value_blocks = self.transform_basis(values)
                # looping over the different elements in the base
                for i in range(self.base):
                    node.child_nodes[i], offset[i], depth = create_diagram_rec(value_blocks[i])
                depth += 1
                node, new_offset = self.create_tuple(node, offset)
                node.d = depth
            # because, in all likelihood, the following has to be calculated anyways, calculating it now will
            #  eliminate the need for another recursion through the diagram.
            node.nodes
            node.leaves
            node.__hash__()
            return node, new_offset, depth

        diagram, f_offset, _ = create_diagram_rec(leaves)

        # making sure that the entire diagram is not "off" by the final offset
        self.include_final_offset(diagram, f_offset)
        if to_reduce:
            self.reduce(diagram)

        return diagram


class MTxDD(Diagram):
    """
    This is the class for all multi-terminal DDs of arbitrary basis. The basis is set at initialization
    """
    def __init__(self, basis):
        from node import Node, Leaf
        self.base = basis
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


class AEVxDD(Diagram):
    """
    This is the class for all additive edge-valued DDs of arbitrary basis. The basis is set at initialization
    """
    def __init__(self, basis):
        from node import Node, Leaf
        self.base = basis
        Diagram.__init__(self, Node, Leaf)

    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates the leaves from the values given, and the node one step up
        """
        from node import Leaf
        parent_node.child_nodes[0] = Leaf(0, 0, diagram_type=AEVxDD)
        parent_node.offsets[0] = 0
        for i in range(1, self.base, 1):
            parent_node.child_nodes[i] = parent_node.child_nodes[0]
            parent_node.offsets[i] = leaf_values[i] - leaf_values[0]
        return parent_node, leaf_values[0]

    def create_tuple(self, node, offset):
        """
        Computes the offset for a node, given the offset of its children
        """
        node.offsets[0] = 0
        for i in range(1, self.base, 1):
            node.offsets[i] = offset[i] - offset[0]
        return node, offset[0]

    @staticmethod
    def include_final_offset(node, offset):
        """
        This function includes an offset remaining after creating the diagram into the diagram.
        """
        for leaf in node.leaves:
            leaf.value = leaf.value + offset

    @staticmethod
    def to_mat(node, goffset=0, loffset=0):
        """
        The diagram-type specific function to convert nodes to matrices
        """
        loffset = 0 if loffset is None else loffset
        import numpy as np
        from node import Node, Leaf
        if isinstance(node, Leaf):
            return np.array((node.value + goffset))[None]
        elif isinstance(node, Node):
            return loffset + goffset
        else:
            raise TypeError


class MEVxDD(Diagram):
    """
    This is the class for all additive edge-valued DDs of arbitrary basis. The basis is set at initialization
    """
    def __init__(self, basis):
        from node import Node, Leaf
        self.base = basis
        Diagram.__init__(self, Node, Leaf)

    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates the leaves from the values given, and the node one step up
        """
        from node import Leaf
        import numpy
        parent_node.child_nodes[0] = Leaf(1, 1, diagram_type=MEVxDD)
        try:
            base_factor = leaf_values[numpy.nonzero(leaf_values)[0][0]]
        except IndexError:
            base_factor = 1
        for i in range(self.base):
            parent_node.child_nodes[i] = parent_node.child_nodes[0]
            parent_node.offsets[i] = leaf_values[i] / base_factor
        return parent_node, base_factor

    def create_tuple(self, node, offset):
        """
        Computes the offset for a node, given the offset of its children
        """
        base_factor = offset[0] if offset[0] != 0 else 1
        for i in range(self.base):
            node.offsets[i] = offset[i] / base_factor
        return node, offset[0]

    def transform_basis(self, values):
        """
        The transform function to change the basis. For all non-spectral DDs, this is the identity projection.
        :param values: the data to be transformed
        :return: An unchanged array
        """
        block_len = len(values)/self.base
        blocks = [values[i*block_len:(i+1)*block_len] for i in range(self.base)]
        return blocks

    @staticmethod
    def include_final_offset(node, offset):
        """
        This function includes an offset remaining after creating the diagram into the diagram.
        """
        for leaf in node.leaves:
            leaf.value = leaf.value * offset

    @staticmethod
    def to_mat(node, goffset=1, loffset=1):
        """
        The diagram-type specific function to convert nodes to matrices
        """
        loffset = 1 if loffset is None else loffset
        import numpy as np
        from node import Node, Leaf
        if isinstance(node, Leaf):
            return np.array((node.value * goffset))[None]
        elif isinstance(node, Node):
            return loffset * goffset
        else:
            raise TypeError