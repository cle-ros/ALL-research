# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""


class BinaryDiagram:
    def __init__(self, nt, lt):
        """
        The init method
        :rtype : a copy of itself
        """
        self.node_type = nt
        self.leaf_type = lt
        # self.paths = self.get_path()

    def get_path(self):
        raise NotImplementedError

    def create_leaves(self, parent_node, leaf_values):
        raise NotImplementedError

    @staticmethod
    def reduce(node_o):
        """
        This function reduces the diagram to a (close to) minimal representation.
        """
        from node import Leaf

        def collect_traces_rec(node):
            """
            this function creates a dictionary of all the traces in the diagram
            """
            if isinstance(node, Leaf):
                return {str(node.value): [[], []]}, str(node.value)
            else:
                traces_p = traces_n = {}
                recent_trace_n = recent_trace_p = ''
                # recursively collecting the traces
                #   adding the current node as reference for the trace, because the reference in the parent node
                #   has to be updated
                if node.n:
                    traces_n, recent_trace_n = collect_traces_rec(node.n)
                    if recent_trace_n in traces_n:
                        traces_n[recent_trace_n][0].append(node)
                    else:
                        traces_n[recent_trace_n] = [[node], []]
                if node.p:
                    traces_p, recent_trace_p = collect_traces_rec(node.p)
                    if recent_trace_p in traces_p:
                        traces_p[recent_trace_p][1].append(node)
                    else:
                        traces_p[recent_trace_p] = [[], [node]]
                traces = {}
                # merging all traces to a new dictionary
                for ctrace in traces_n:
                    if ctrace in traces:
                        traces[ctrace][0] = traces[ctrace][0] + traces_n[ctrace][0]
                        traces[ctrace][1] = traces[ctrace][1] + traces_n[ctrace][1]
                    else:
                        traces[ctrace] = traces_n[ctrace]
                for ctrace in traces_p:
                    if ctrace in traces:
                        traces[ctrace][0] = traces[ctrace][0] + traces_p[ctrace][0]
                        traces[ctrace][1] = traces[ctrace][1] + traces_p[ctrace][1]
                    else:
                        traces[ctrace] = traces_p[ctrace]
                # returning it
                if node.no:
                    recent_trace_n = str(node.no) + recent_trace_n
                if node.po:
                    recent_trace_p = str(node.po) + recent_trace_p
                return traces, '0:'+recent_trace_n+'1:'+recent_trace_p

        known_traces, last_trace = collect_traces_rec(node_o)
        # looping over all the traces to reduce
        for trace in known_traces:
            try:
                trace_reference = known_traces[trace][0][0].n
                for j in range(0, len(known_traces[trace][0]), 1):
                    known_traces[trace][0][j].n = trace_reference
            except IndexError:
                # if the trace defining the current node is unknown, i.e. has not appeared in the negative branch
                trace_reference = known_traces[trace][1][0].p
            for j in range(0, len(known_traces[trace][1]), 1):
                known_traces[trace][1][j].p = trace_reference
        node_o.reinitialize_leaves()
        node_o.reinitialize_nodes()
        return node_o

    def flatten(self):
        raise NotImplementedError

    def create(self, matrix, null_value, to_reduce=True):
        """
        this function creates a diagram of the specified type of the given matrix
        """
        from diagram_initialization import get_req_vars
        from diagram_matrix_and_variable_operations import expand_matrix2n
        # get the required number of vars
        no_vars = get_req_vars(matrix)
        # expand the matrix to be of size 2^nx2^m
        matrix = expand_matrix2n(matrix, no_vars[1:], null_value)
        # get the not-suppressed values
        leaves = matrix.flatten()

        def create_diagram_rec(values):
            node = self.node_type('', diagram_type=self.__class__)
            entry_length = len(values)/2
            if entry_length == 1:
                node, offset = self.create_leaves(node, values)
                node.d = depth = 0
            else:
                # somewhere around here the create_tuple has to be used.
                node.p, p_offset, depth = create_diagram_rec(values[entry_length:])
                node.n, n_offset, depth = create_diagram_rec(values[0:entry_length])
                depth += 1
                node, offset = self.create_tuple(node, n_offset, p_offset)
                node.d = depth
            return node, offset, depth

        diagram, f_offset, d = create_diagram_rec(leaves)

        # making sure that the entire diagram is not "off" by the final offset
        if f_offset != 0:
            self.include_final_offset(diagram, f_offset)
        if to_reduce:
            self.reduce(diagram)

        return diagram

    def create_tuple(self, node, n_offset, p_offset):
        """
        This method defines the general framework to branch. It relies on specific implementations for the different
         binary diagram types
        """
        raise NotImplementedError

    def to_mat(self, loffset, goffset):
        raise NotImplementedError

    def include_final_offset(self, offset):
        raise NotImplementedError

    def add(self, node1, offset=[0, 0]):
        """
        This function adds two nodes
        :rtype : array of offsets (e.g. n-offset, p-offset)
        :param node1: the first node
        :param node2: the second node
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

    def create_tuple(self, node, n_offset, p_offset):
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
    def create_tuple(node, n_offset, p_offset):
        """
        Computes the offset for a node, given the offset of its children
        """
        node.no = 0
        node.po = p_offset - n_offset
        return node, n_offset

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