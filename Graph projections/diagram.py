# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""


class Diagram:
    """
    This is an interface/abstract class for all different diagram types
    """
    base = 4

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

    def include_final_offset(self, offset):
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

    def create(self, matrix, null_value, to_reduce=True, dec_digits=-1):
        """
        this function creates a diagram of the specified type of the given matrix
        :param matrix:      The data to be represented
        :param null_value:  The null value (for *-suppressed DDs)
        :param to_reduce:   Whether the tree shall be represented as a diagram
        :param dec_digits:  The number of decimal digits to round to
        """
        from diagram_matrix_and_variable_operations import expand_matrix_exponential, get_req_vars
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
            entry_length = len(values)/self.__class__.base
            if entry_length == 1:
                node, new_offset = self.create_leaves(node, values)
                node.d = depth = 1
            else:
                # somewhere around here the create_tuple has to be used.
                offset = {}
                depth = 0
                block_len = len(values)/self.base
                # looping over the different elements in the base
                for i in range(self.base):
                    node.child_nodes[i], offset[i], depth = create_diagram_rec(values[i*block_len:(i+1)*block_len])
                depth += 1
                node, new_offset = self.create_tuple(node, offset)
                node.d = depth
            # because, in all likelihood, the following has to be calculated anyways, calculating it now will
            #  eliminate the need for another recursion through the diagram.
            n = node.nodes
            node.__hash__()
            return node, new_offset, depth

        diagram, f_offset, _ = create_diagram_rec(leaves)

        # making sure that the entire diagram is not "off" by the final offset
        if f_offset != 0:
            self.include_final_offset(diagram, f_offset)
        if to_reduce:
            self.reduce(diagram)

        return diagram

