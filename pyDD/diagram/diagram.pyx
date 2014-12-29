# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""

from pyDD.utilities.singleton import Singleton


class Diagram:
    """
    This is an interface/abstract class for all different diagram types
    """
    base = None
    __metaclass__ = Singleton

    def __init__(self, nt, lt):
        """
        The init method
        :rtype : Diagram
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

    def to_mat(self, loffset, goffset=None, reorder=False):
        """
        This method converts two offsets to a corresponding matrix form. If the node is a leaf, it includes its value.
        :param loffset:
        :param goffset:
        :param reorder:
        :raise NotImplementedError:
        :type loffset: numpy.ndarray
        :type goffset: numpy.ndarray
        :type reorder: bool
        :rtype: numpy.ndarray
        """
        raise NotImplementedError

    @staticmethod
    def include_final_offset(node, offset):
        raise NotImplementedError

    @staticmethod
    def add(diagram1, diagram2, offset):
        """
        This function adds two nodes
        :rtype : array of offsets (e.g. n-offset, p-offset)
        :param diagram1: the first node
        :param diagram2: the second node
        :param offset: the parent offset
        :type diagram1: node.Node
        :type diagram2: node.Node
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
        """
        This method computes the new offsets resulting from a variable-exchange. I.e., it takes the existing offsets,
        combines them, and computes new ones based on the changed order of the variables.
        :param exchange_matrix: An array defining the
        :param dtype:
        :return:
        """
        import numpy as np
        # computing the combined offsets
        comb_offsets = np.array([])
        offset_matrix = np.array([])
        for ex_row in exchange_matrix:
            try:
                comb_offsets = np.vstack((comb_offsets, dtype.to_mat(None, loffset=ex_row[4], goffset=ex_row[2],
                                                                     reorder=True)))
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

    def split_elements_to_basis(self, values):
        """
        The transform function to change the basis. For the MTxDDs, this is the identity projection.
        :param values: the different sections of data to be transformed
        :return: An unchanged array
        """
        block_len = len(values)/self.base
        blocks = [values[i*block_len:(i+1)*block_len] for i in range(self.base)]
        return blocks

    def create(self, matrix, null_value, to_reduce=True, dec_digits=-1, kron_exp=None):
        """
        this function creates a diagram of the specified type of the given matrix
        :param matrix:      The data to be represented
        :param null_value:  The null value (for *-suppressed DDs)
        :param to_reduce:   Whether the tree shall be represented as a diagram
        :param dec_digits:  The number of decimal digits to round to
        :param kron_exp:    The basis expansion matrix
        :return: the diagram
        :type matrix: numpy.ndarray
        :type null_value: float
        :type to_reduce: bool
        :type dec_digits: int
        :type kron_exp: numpy.ndarray
        :rtype: pyDD.diagram.node.pyx.Node
        """
        from pyDD.utilities.general_helper_functions import expand_matrix_exponential, get_req_vars
        if kron_exp.shape[0] != self.base:
            raise BaseException('Mismatching bases for Kronecker expansion')
        # initializing the reduction
        hashtable = {}
        # get the required number of vars
        no_vars = get_req_vars(matrix, self.base)
        # expand the matrix to be of size 2^nx2^m
        matrix = expand_matrix_exponential(matrix, no_vars[1:], null_value, self.base)
        # get the not-suppressed values
        leaves = matrix.flatten()
        # should the values be rounded to increase compression?
        import numpy
        if dec_digits != -1:
            leaves = numpy.round(leaves, dec_digits)

        def create_diagram_rec(values):
            """
            The recursive function
            :type values: numpy.ndarray
            :rtype node: node.Node
            :rtype new_offset: list
            :rtype depth: int
            """
            node = self.node_type('', diagram_type=self.__class__())
            entry_length = len(values)/self.base
            if entry_length == 1:
                node, new_offset = self.create_leaves(node, values)
                node.d = depth = 1
            else:
                # somewhere around here the create_tuple has to be used.
                offset = {}
                depth = 0
                value_blocks = self.split_elements_to_basis(values)
                # looping over the different elements in the base
                for i in range(self.base):
                    node.child_nodes[i], offset[i], depth = create_diagram_rec(value_blocks[i] )
                depth += 1
                node, new_offset = self.create_tuple(node, offset)
                node.d = depth
            # because, in all likelihood, the following has to be calculated anyways, calculating it now will
            #  eliminate the need for another recursion through the diagram.
            node.reinitialize()
            if to_reduce:
                if not node.__hash__() in hashtable:
                    hashtable[node.__hash__()] = node
                else:
                    node = hashtable[node.__hash__()]
            return node, new_offset, depth

        diagram, f_offset, _ = create_diagram_rec(leaves, numpy.ones(self.base**2))

        # making sure that the entire diagram is not "off" by the final offset
        self.include_final_offset(diagram, f_offset)
        if to_reduce:
            diagram.reduce()
        diagram.shape = matrix.shape

        return diagram


class MTxDD(Diagram):
    """
    This is the class for all multi-terminal DDs of arbitrary basis. The basis is set at initialization
    """
    null_edge_value = None

    def __init__(self, basis):
        from pyDD.diagram.node import Node, Leaf
        self.base = basis
        Diagram.__init__(self, Node, Leaf)

    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates the leaves from the values given, and the node one step up
        :param parent_node: the parent node
        :param leaf_values: the leaf values
        :return: the parent node, with properly modified child properties
        :type parent_node: node.Node
        :type leaf_values: numpy.ndarray
        :rtype: node.Node
        """
        for i in range(self.base):
            # if zero_suppressed
            parent_node.child_nodes[i] = self.leaf_type(leaf_values[i], leaf_values[i], diagram_type=self.__class__)
        return parent_node, 0.0

    def create_tuple(self, node, offset):
        return node, 0.0

    @staticmethod
    def to_mat(leaf, loffset=None, goffset=None, reorder=False):
        """
        This method converts two offsets to a corresponding matrix form. If the node is a leaf, it includes its value.
        :param loffset:
        :param goffset:
        :param reorder:
        :type loffset: numpy.ndarray
        :type goffset: numpy.ndarray
        :type reorder: bool
        :rtype: numpy.ndarray
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
        if offset != 0.0:
            for leaf in node.leaves:
                leaf.value = leaf.value + offset

    @staticmethod
    def rearrange_offsets(exchange_matrix, dtype):
        """
        This function rearranges the offsets for two variables switching order
        """
        ret_mat = []
        for exchange_row in exchange_matrix:
            ret_mat.append([exchange_row[0], exchange_row[3], exchange_row[4], exchange_row[1], exchange_row[2],
                            exchange_row[5]])
        return ret_mat

    @staticmethod
    def scalar_mult(diagram, scalar):
        """
        The helper function for scalar multiplication
        """
        for leaf in diagram.leaves:
            leaf.value *= scalar


class AEVxDD(Diagram):
    """
    This is the class for all additive edge-valued DDs of arbitrary basis. The basis is set at initialization
    """
    null_edge_value = 0.0
    null_leaf_value = 0.0

    def __init__(self, basis):
        from pyDD.diagram.node import Node, Leaf
        self.base = basis
        Diagram.__init__(self, Node, Leaf)

    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates the leaves from the values given, and the node one step up
        :param parent_node: the parent node
        :param leaf_values: the leaf values
        :return: the parent node, with properly modified child properties
        :type parent_node: node.Node
        :type leaf_values: numpy.ndarray
        :rtype: node.Node
        """
        from pyDD.diagram.node import Leaf
        parent_node.child_nodes[0] = Leaf(0.0, 0, diagram_type=AEVxDD)
        average = leaf_values.mean()
        for i in range(self.base):
            parent_node.child_nodes[i] = parent_node.child_nodes[0]
            parent_node.offsets[i] = leaf_values[i] - average
        return parent_node, average

    def create_tuple(self, node, offset):
        """
        Computes the offset for a node, given the offset of its children
        """
        average = 0.0
        for os in offset:
            try:
                average += offset[os]
            except IndexError:
                average += os
        average /= len(offset)
        node.offsets[0] = 0.0
        for i in range(self.base):
            node.offsets[i] = offset[i] - average
        return node, average

    @staticmethod
    def include_final_offset(node, offset):
        """
        This function includes an offset remaining after creating the diagram into the diagram.
        """
        for leaf in node.leaves:
            leaf.value = leaf.value + offset

    @staticmethod
    def to_mat(node, loffset=0.0, goffset=None, reorder=False):
        """
        This method converts two offsets to a corresponding matrix form. If the node is a leaf, it includes its value.
        :param loffset:
        :param goffset:
        :param reorder:
        :raise TypeError:
        :type loffset: numpy.ndarray
        :type goffset: numpy.ndarray
        :type reorder: bool
        :rtype: numpy.ndarray
        """
        goffset = 0 if goffset is None else goffset
        import numpy as np
        from pyDD.diagram.node import Node, Leaf
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


class MEVxDD(Diagram):
    """
    This is the class for all additive edge-valued DDs of arbitrary basis. The basis is set at initialization
    """
    null_edge_value = 1.0
    null_leaf_value = 1.0

    def __init__(self, basis):
        from pyDD.diagram.node import Node, Leaf
        self.base = basis
        Diagram.__init__(self, Node, Leaf)

    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates the leaves from the values given, and the node one step up
        :param parent_node: the parent node
        :param leaf_values: the leaf values
        :return: the parent node, with properly modified child properties
        :type parent_node: node.Node
        :type leaf_values: numpy.ndarray
        :rtype: node.Node
        """
        from pyDD.diagram.node import Leaf
        import numpy
        parent_node.child_nodes[0] = Leaf(1.0, 1, diagram_type=MEVxDD)
        try:
            base_factor = leaf_values[numpy.nonzero(leaf_values)[0][0]]
        except IndexError:
            base_factor = 1.0
        for i in range(self.base):
            parent_node.child_nodes[i] = parent_node.child_nodes[0]
            parent_node.offsets[i] = leaf_values[i] / base_factor
        return parent_node, base_factor

    def create_tuple(self, node, offset):
        """
        Computes the offset for a node, given the offset of its children
        """
        base_factor = offset[0] if offset[0] != 0.0 else 1.0
        for i in range(self.base):
            node.offsets[i] = offset[i] / base_factor
        return node, offset[0]

    def split_elements_to_basis(self, values):
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
    def to_mat(node, loffset=1.0, goffset=1.0, reorder=False):
        """
        This method converts two offsets to a corresponding matrix form. If the node is a leaf, it includes its value.
        :param loffset:
        :param goffset:
        :param reorder:
        :type loffset: numpy.ndarray
        :type goffset: numpy.ndarray
        :type reorder: bool
        :rtype: numpy.ndarray
        """
        goffset = 1 if goffset is None else goffset
        import numpy as np
        from pyDD.diagram.node import Node, Leaf
        if isinstance(node, Leaf):
            return np.array((node.value * loffset))[None]
        elif isinstance(node, Node) or reorder:
            return np.multiply(loffset, goffset)
        else:
            raise TypeError

    @staticmethod
    def recompute_offsets(offsets, base):
        """
        This method recomputes the offsets of different nodes in the DD after their encoded variable has changed.
        """
        base_factor = offsets[0][0] if offsets[0] != 0.0 else 1.0
        return offsets[0][0], [offsets[i][0]/base_factor for i in range(base)]

    @staticmethod
    def scalar_mult(diagram, scalar):
        """
        The helper function for scalar multiplication
        """
        for oindex in diagram.offsets:
            diagram.offsets[oindex] *= scalar


class FEVxDD(Diagram):
    """
    This class gives a generalization of  Scott Sanner's AADDs
    """
    import numpy
    null_edge_value = numpy.array([0.0, 1.0])
    null_leaf_value = 0.0

    def __init__(self, basis):
        from pyDD.diagram.node import Node, Leaf
        self.base = basis
        Diagram.__init__(self, Node, Leaf)

    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates terminal nodes given the parent node and the leaf values

        How to decide on the edge values:
        1. the leaf-value @ branch 0 pushes the according add. offset to 0
        2. the multiplicative offset
        2. for max-branch: add. offset + mult. offset = max(leaves)
        3. @ leaf-level: mult. offset = 0

        :param parent_node: the parent node
        :param leaf_values: the leaf values
        :return: the parent node, with properly modified child properties
        :type parent_node: node.Node
        :type leaf_values: numpy.ndarray
        :rtype: node.Node
        """
        # TODO: find generalization!!
        import numpy as np
        # creating the leaf object
        parent_node.child_nodes[0] = self.leaf_type(0.0, 0, diagram_type=self.__class__)

        # creating the offsets
        # deciding on mult or add rule
        # additive_coefficient = np.mean(leaf_values)
        # new_offsets = np.array([leaf_values[i]-additive_coefficient for i in range(self.base)])
        # max_difference = np.max(np.abs(new_offsets))
        # mult_coefficient = max_difference if max_difference != 0.0 else 1.0
        # for i in range(self.base):
        #     node.child_nodes[i] = node.child_nodes[0]
        #     node.offsets[i] = np.array([((new_offsets[i])/mult_coefficient), mult_coefficient], dtype='float64')
        # return node, [additive_coefficient, mult_coefficient]
        if leaf_values[0] == 0 or (leaf_values[1]-leaf_values[0] < leaf_values[1]/leaf_values[0]):
            parent_node.offsets[0] = np.array([0, 1], dtype='float64')
            for i in range(1, self.base, 1):
                parent_node.child_nodes[i] = parent_node.child_nodes[0]
                parent_node.offsets[i] = np.array([(leaf_values[i]-leaf_values[0]), 1], dtype='float64')
            return parent_node, [leaf_values[0], 1]
        else:
            parent_node.offsets[0] = np.array([1, 1], dtype='float64')
            for i in range(1, self.base, 1):
                parent_node.child_nodes[i] = parent_node.child_nodes[0]
                parent_node.offsets[i] = np.array([leaf_values[i]/leaf_values[0], (leaf_values[i]/leaf_values[0])],
                                                  dtype='float64')
            return parent_node, [0, leaf_values[0]]

    def create_tuple(self, node, offset):
        """
        This method defines how AABDDs branch.
        """
        import numpy as np
        # creating the new offsets
        # if offset[0][0] == 0:
        # additive_coefficient = np.mean(np.array(offset.values())[0, :])
        # mult_coefficient = np.median(np.array(offset.values())[1, :])
        # mult_coefficient = mult_coefficient if mult_coefficient != 0.0 else 1.0
        # node.offsets[0] = np.array([0, offset[0][1]], dtype='float64')
        # for i in range(self.base):
        #     node.offsets[i] = np.array([((offset[i][0]-additive_coefficient)/mult_coefficient), mult_coefficient],
        # dtype='float64')
        # return node, [additive_coefficient, mult_coefficient]
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
    def to_mat(node, loffset, goffset=None, reorder=False):
        """
        This method converts two offsets to a corresponding matrix form. If the node is a leaf, it includes its value.
        :param loffset:
        :param goffset:
        :param reorder:
        :type loffset: numpy.ndarray
        :type goffset: numpy.ndarray
        :type reorder: bool
        :rtype: numpy.ndarray
        """
        goffset = FEVxDD.null_edge_value if goffset is None else goffset
        loffset = FEVxDD.null_edge_value if loffset is None else loffset
        import numpy as np
        from pyDD.diagram.node import Node, Leaf
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
        import numpy
        # TODO: When the overall AADD branching criterion is designed, this has to be considered as well.
        return numpy.array([0, 1]), [numpy.array([offsets[i][0], offsets[i][1]]) for i in range(base)]

    @staticmethod
    def scalar_mult(diagram, scalar):
        """
        The helper function for scalar multiplication
        """
        for oindex in diagram.offsets:
            diagram.offsets[oindex] *= scalar


class AAxDD(FEVxDD):
    def create_leaves(self, parent_node, leaf_values):
        """
        This function creates terminal nodes given the parent node and the leaf values

        How to decide on the edge values:
        1. the leaf-value @ branch 0 pushes the according add. offset to 0
        2. the multiplicative offset
        2. for max-branch: add. offset + mult. offset = max(leaves)
        3. @ leaf-level: mult. offset = 0

        :param parent_node: the parent node
        :param leaf_values: the leaf values
        :return: the parent node, with properly modified child properties
        :type parent_node: node.Node
        :type leaf_values: numpy.ndarray
        :rtype: node.Node
        """
        import numpy as np
        # TODO: find generalization!!
        # creating the leaf object
        # finding the smallest value:
        min_v = np.min(leaf_values)
        max_v = np.max(leaf_values)
        child_node = self.leaf_type(0.0, 1, diagram_type=self.__class__)
        mult_re = abs((max_v-min_v)/min_v) if min_v != 0.0 else 1.0
        for i in range(self.base):
            parent_node.child_nodes[i] = child_node
            parent_node.offsets[i] = np.array([(leaf_values[i]-min_v)/mult_re, 0.0], dtype='float64')
        return parent_node, [min_v, mult_re]

    def create_tuple(self, node, offset):
        """
        This method defines how AABDDs branch.
        """
        import numpy as np
        # determining the range of additive coefficients
        min_add = offset[0][0]
        max_add = offset[0][0]
        min_mul = offset[0][1]
        max_mul = offset[0][1]
        for i in offset:
            min_add = min(min_add, offset[i][0])
            max_add = max(max_add, offset[i][0])
            min_mul = min(min_mul, offset[i][1])
            max_mul = max(max_mul, offset[i][1])
        mult_re = np.nanmax([abs((max_add-min_add)/max_add), 1.0, max_mul])
        # print mult_re
        for i in range(self.base):
            node.offsets[i] = np.array([(offset[i][0]-min_add)/mult_re, offset[i][1]/mult_re], dtype='float64')
        return node, [min_add, mult_re]
