# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""

from singleton import Singleton


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

    def to_mat(self, loffset, goffset, reorder=False):
        raise NotImplementedError

    @staticmethod
    def include_final_offset(node, offset):
        raise NotImplementedError

    @staticmethod
    def add(diagram1, diagram2, offset):
        """
        This function adds two nodes
        :rtype : array of offsets (e.g. n-offset, p-offset)
        :param node1: the first node
        :param offset: the parent offset
        """
        raise NotImplementedError

    @staticmethod
    def sum(diagram, offset):
        """
        the helper function for summing a diagram
        """
        raise NotImplementedError

    @staticmethod
    def scalar_mult(diagram, scalar):
        """
        The helper function for scalar multiplication
        """
        raise NotImplementedError

    @staticmethod
    def mult(diagram1, diagram2):
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

    @staticmethod
    def rearrange_offsets(exchange_matrix, dtype):
        import numpy as np
        # computing the combined offsets
        comb_offsets = np.array([])
        offset_matrix = np.array([])
        for ex_row in exchange_matrix:
            try:
                comb_offsets = np.vstack((comb_offsets, dtype.to_mat(None, loffset=ex_row[4], goffset=ex_row[2], reorder=True)))
                offset_matrix = np.vstack((offset_matrix, np.array([ex_row[2], ex_row[4]])))
            except ValueError:
                comb_offsets = dtype.to_mat(None, loffset=ex_row[4], goffset=ex_row[2], reorder=True)
                offset_matrix = np.array([ex_row[2], ex_row[4]])
        # recomputing the new offsets
        # finding the new 1st level branches:
        branches = np.array([[row[1], row[3]] for row in exchange_matrix])
        comb_mat = []
        # cycling over the different possible offsets
        for i in range(dtype.base):
            new_offset = dtype.recompute_offsets(comb_offsets[np.where(branches[:, 1] == i)], dtype.base)
            for j in range(len(new_offset[1])):
                comb_mat.append([exchange_matrix[np.where(branches[:, 1] == i)[0][j]][0], i, new_offset[0],
                                branches[np.where(branches[:, 1] == i)][j, 0],
                                new_offset[1][j], exchange_matrix[np.where(branches[:, 1] == i)[0][j]][5]])
        return comb_mat

    @staticmethod
    def recompute_offsets(offsets, base):
        raise NotImplementedError

    def transform_basis(self, values):
        """
        The transform function to change the basis. For the MTxDDs, this is the identity projection.
        :param values: the different sections of data to be transformed
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
        # initializing the reduction
        hashtable = {}
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
            if to_reduce:
                if not node.__hash__() in hashtable:
                    hashtable[node.__hash__()] = node
                else:
                    node = hashtable[node.__hash__()]
            return node, new_offset, depth

        diagram, f_offset, _ = create_diagram_rec(leaves)

        # making sure that the entire diagram is not "off" by the final offset
        self.include_final_offset(diagram, f_offset)
        # if to_reduce:
        #     diagram.reduce()

        return diagram


class MTxDD(Diagram):
    """
    This is the class for all multi-terminal DDs of arbitrary basis. The basis is set at initialization
    """
    null_edge_value = None

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
    def to_mat(leaf, loffset=None, goffset=None, reorder=False):
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

    @staticmethod
    def rearrange_offsets(exchange_matrix, dtype):
        """
        This function rearranges the offsets for two variables switching order
        """
        ret_mat = []
        for exchange_row in exchange_matrix:
            ret_mat.append([exchange_row[0], exchange_row[3], exchange_row[4], exchange_row[1], exchange_row[2], exchange_row[5]])
        return ret_mat

    @staticmethod
    def scalar_mult(diagram, scalar):
        """
        The helper function for scalar multiplication
        """
        for leaf in diagram.leaves:
            leaf.value *= scalar

    @staticmethod
    def sum(diagram, offset):
        """
        the helper function for summing a diagram
        """
        raise NotImplementedError


class AEVxDD(Diagram):
    """
    This is the class for all additive edge-valued DDs of arbitrary basis. The basis is set at initialization
    """
    null_edge_value = [0]

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
    def to_mat(node, loffset=0, goffset=0, reorder=False):
        """
        The diagram-type specific function to convert nodes to matrices
        """
        goffset = 0 if goffset is None else goffset
        import numpy as np
        from node import Node, Leaf
        if isinstance(node, Leaf):
            return np.array((node.value + loffset))[None]
        elif isinstance(node, Node) or reorder:
            return loffset + goffset
        else:
            raise TypeError

    @staticmethod
    def recompute_offsets(offsets, base):
        """
        This method recomputes the offsets of different nodes in the DD after their encoded variable has changed.
        """
        return offsets[0][0], [(offsets[i]-offsets[0])[0] for i in range(base)]

    @staticmethod
    def scalar_mult(diagram, scalar):
        """
        The helper function for scalar multiplication
        """
        for node in diagram.nodes:
            if node.is_leaf():
                node.value *= scalar
            else:
                for oindex in node.offsets:
                    node.offsets[oindex] *= scalar

    @staticmethod
    def sum(diagram, offset):
        """
        the helper function for summing a diagram
        """
        raise NotImplementedError


class MEVxDD(Diagram):
    """
    This is the class for all additive edge-valued DDs of arbitrary basis. The basis is set at initialization
    """
    null_edge_value = [1]

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
    def to_mat(node, loffset=1, goffset=1, reorder=False):
        """
        The diagram-type specific function to convert nodes to matrices
        """
        goffset = 1 if goffset is None else goffset
        import numpy as np
        from node import Node, Leaf
        if isinstance(node, Leaf):
            return np.array((node.value * loffset))[None]
        elif isinstance(node, Node) or reorder:
            return loffset * goffset
        else:
            raise TypeError

    @staticmethod
    def recompute_offsets(offsets, base):
        """
        This method recomputes the offsets of different nodes in the DD after their encoded variable has changed.
        """
        base_factor = offsets[0][0] if offsets[0] != 0 else 1
        return offsets[0][0], [offsets[i][0]/base_factor for i in range(base)]

    @staticmethod
    def scalar_mult(diagram, scalar):
        """
        The helper function for scalar multiplication
        """
        for oindex in diagram.offsets:
            diagram.offsets[oindex] *= scalar


class AAxEVDD(Diagram):
    """
    This class gives a generalization of  Scott Sanner's AADDs
    """
    null_edge_value = [0, 1]

    def __init__(self, basis):
        from node import Node, Leaf
        self.base = basis
        Diagram.__init__(self, Node, Leaf)

    def create_leaves(self, node, leaf_values):
        """
        This function creates terminal nodes given the parent node and the leaf values

        How to decide on the edge values:
        1. the leaf-value @ branch 0 pushes the according add. offset to 0
        2. the multiplicative offset
        2. for max-branch: add. offset + mult. offset = max(leaves)
        3. @ leaf-level: mult. offset = 0
        """
        # TODO: find generalization!!
        import numpy as np
        # creating the leaf object
        node.child_nodes[0] = self.leaf_type(0, 0, diagram_type=self.__class__)

        # creating the offsets
        # deciding on mult or add rule
        # if leaf_values[0] == 0:
        if leaf_values[0] == 0 or (leaf_values[1]-leaf_values[0] < leaf_values[1]/leaf_values[0]):
            node.offsets[0] = np.array([0, 1], dtype='float64')
            for i in range(1, self.base, 1):
                node.child_nodes[i] = node.child_nodes[0]
                node.offsets[i] = np.array([(leaf_values[i]-leaf_values[0]), 1], dtype='float64')
            return node, [leaf_values[0], 1]
        else:
            node.offsets[0] = np.array([1, 1], dtype='float64')
            for i in range(1, self.base, 1):
                node.child_nodes[i] = node.child_nodes[0]
                node.offsets[i] = np.array([leaf_values[i]/leaf_values[0], (leaf_values[i]/leaf_values[0])], dtype='float64')
            return node, [0, leaf_values[0]]

    def create_tuple(self, node, offset):
        """
        This method defines how AABDDs branch.
        """
        import numpy as np
        # creating the new offsets
        # if offset[0][0] == 0:
        if offset[0][1] == 0 or offset[1][0]-offset[0][0] < offset[1][1]/offset[0][1]:
            node.offsets[0] = np.array([0, offset[0][1]], dtype='float64')
            for i in range(1, self.base, 1):
                node.offsets[i] = np.array([(offset[i][0]-offset[0][0]), offset[i][1]], dtype='float64')
            return node, [offset[0][0], 1]
        else:
            node.offsets[0] = np.array([offset[0][0]/offset[0][1], 1], dtype='float64')
            for i in range(1, self.base, 1):
                node.offsets[i] = np.array([offset[i][0]/offset[0][1], (offset[i][1]/offset[0][1])], dtype='float64')
            return node, [0, offset[0][1]]

    @staticmethod
    def to_mat(node, loffset, goffset=null_edge_value, reorder=False):
        """
        The diagram-type specific function to convert nodes to matrices
        """
        goffset = AAxEVDD.null_edge_value if goffset is None else goffset
        loffset = AAxEVDD.null_edge_value if loffset is None else loffset
        import numpy as np
        from node import Node, Leaf
        if isinstance(node, Leaf):
            return np.array((goffset[0] + goffset[1]*(loffset[0] + loffset[1]*node.value)))[None]
        elif isinstance(node, Node) or reorder:
            return goffset[0] + goffset[1]*loffset[0], goffset[1]*loffset[1]
        else:
            raise TypeError

    @staticmethod
    def include_final_offset(node, offset):
        """
        In certain cases an offset remains after the creation of the DD. This function includes its information in the
        DD.
        """
        for oindex in node.offsets:
            node.offsets[oindex][0] = offset[0] + offset[1]*node.offsets[oindex][0]
            node.offsets[oindex][1] *= offset[1]

    @staticmethod
    def recompute_offsets(offsets, base):
        """
        This method recomputes the offsets of different nodes in the DD after their encoded variable has changed.
        """
        # TODO: When the overall AADD branching criterion is designed, this has to be considered as well.
        return [0, 1], [[offsets[i][0], offsets[i][1]] for i in range(base)]

    @staticmethod
    def scalar_mult(diagram, scalar):
        """
        The helper function for scalar multiplication
        """
        for oindex in diagram.offsets:
            diagram.offsets[oindex] *= scalar