# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:23:43 2014

@author: clemens
"""

from diagram.exceptions import TerminalNodeException


class Node(object):
    """
    The generic node class for the nodes of diagrams. Generally encodes:
    - the number of children (outgoing branches)
    - the child-objects, indexed by an integer
    - the resp. edge-values, indexed by an integer
    - a value, if it's a Leaf
    """
    properties = {}
    from diagram.diagram import MTxDD

    def __init__(self, denominator='', diagram_type=MTxDD, depth=None, nullvalue=0):
        """        
        all required information are the name of the node
        :param denominator:
        :param diagram_type:
        :param depth:
        :param nullvalue:
        :return: This is a constructor. Guess.
        :type denominator: str
        :type diagram_type: diagram.Diagram
        :type depth: int
        :type nullvalue: float
        :rtype : node.Node
        """
        self.name = denominator
        self.in_order = True
        self.exchange_order = []
        self.d = depth
        self.dtype = diagram_type
        self.child_nodes = {}
        self.offsets = {}
        self.leaves_set = set()
        self.nodes_set = set()
        self.leaf_type = Leaf
        self.null_value = nullvalue
        self.shape = (-1, -1)
        self.hash_value = None

    def get_offset(self, index):
        """
        This function handles possible index exceptions when accessing the offsets
        :param index: the index of the offset to be accessed
        :return: the value, if it exists. None, otherwise.
        """
        if self.is_leaf():
            raise TerminalNodeException
        try:
            return self.offsets[index]
        except KeyError:
            return None

    def set_offset(self, index, value):
        """
        This function handles possibility of there not being any offset to be set
        :param index: the index of the offset to be accessed
        :param value: the value to be set. Nothing will be set if None
        :return:
        """
        if self.is_leaf():
            raise TerminalNodeException
        if value is None:
            return
        else:
            self.offsets[index] = value

    @property
    def leaves(self):
        """
        This property returns the leaves_array (i.e. all the nodes of type 'Leaf' part of this diagram)
        :return: a list containing the leaves
        :rtype: list
        """
        # is the current node a leaf?
        if self.is_leaf():
            return {self}
        # or does it already have leaf-entries?
        elif not self.leaves_set == set():
            return self.leaves_set
        # if not, recursively return all children
        else:
            childrens_leaves = set()
            for child in self.child_nodes:
                childrens_leaves = childrens_leaves.union(self.child_nodes[child].leaves)
            # storing it for later use
            self.leaves_set = childrens_leaves
            return childrens_leaves

    @leaves.setter
    def leaves(self, leaves_array):
        """
         The leaf - setter function.
        """
        self.leaves_set = leaves_array

    def reinitialize(self):
        """
        A method combining the functionality of reinitialize_nodes, reinitialize_leaves and __hash__(reinit)
        :return:
        """
        if self.is_leaf():
            self.__hash__(reinit=True)
            return {self}, {self}
        else:
            children_leaves = set()
            children_nodes = {self}
            # iterating over the children
            for child in self.child_nodes:
                cur_child_leaves, cur_child_nodes = self.child_nodes[child].reinitialize()
                children_leaves = children_leaves.union(cur_child_leaves)
                children_nodes = children_nodes.union(cur_child_nodes)
            # storing the sets for later use
            self.__hash__(reinit=True)
            self.leaves = children_leaves
            self.nodes = children_nodes
            return children_leaves, children_nodes

    @property
    def nodes(self):
        """
        This property returns the nodes_set
        :rtype : list
        :return: a list of leaf-nodes
        """
        # is the current node a leaf?
        if self.is_leaf():
            return {self}
        # or does it already have leaf-entries?
        elif not self.nodes_set == set():
            return self.nodes_set
        # if not, recursively return all children
        else:
            children_nodes = {self}
            for child in self.child_nodes:
                children_nodes = children_nodes.union(self.child_nodes[child].nodes)
            # storing it for later use
            self.nodes_set = children_nodes
            return children_nodes

    @nodes.setter
    def nodes(self, nodes_array):
        """
         The leaf - setter function.
        """
        self.nodes_set = nodes_array

    def is_leaf(self):
        """
        This function checks whether the current node is a leaf node
        """
        return isinstance(self, Leaf)

    def get_subdiagrams(self, depth):
        """
        This method returns all subdiagrams of a specified level as a list
        :param depth:
        :return:
        """
        subdiagrams = set()

        def get_sds_rec(node, sds, level, cur_level):
            if level == cur_level:
                return sds.union({node})
            else:
                # the children have to be sorted for some usages
                children = set()
                for child in node.child_nodes:
                    children = children.union(get_sds_rec(node.child_nodes[child], sds, level, cur_level+1))
                return children
        subdiagrams = get_sds_rec(self, subdiagrams, depth, 0)
        return subdiagrams

    def get_subdiagrams_grouped_by_level(self):
        """
        This function creates a list of sets, where the 0th list contains the root node, and the last entry
        is a set of the leaves. The lists in between correspond to the "distance" from the root, sorted
        by that distance.
        """
        subds = []

        def get_subds_gbl_rec(node, level):
            """
            The recursive call
            """
            try:
                subds[level] = subds[level].union({node})
            except IndexError:
                subds.append({node})
            if not isinstance(node, Leaf):
                for child in node.child_nodes:
                    get_subds_gbl_rec(node.child_nodes[child][0], level+1)

        get_subds_gbl_rec(self, 0)
        return subds

    def add(self, node, **offset):
        """
        This method adds the current node and the argument, returning a new diagram
        """
        return self.dtype.add(self, node, **offset)

    def sum(self):
        """
        This method returns the sum of all elements represented by the diagram
        :returns : the sum
        :rtype : numpy.float64
        """
        import numpy as np

        # covering zero-matrices
        if self.child_nodes == {}:
            return self.null_value

        def sum_rec(node, offset):
            # making sure the node exists
            if not node:
                return 0
            # checking whether the node is a leaf
            elif node.is_leaf():
                return np.sum(node.dtype.to_mat(node, offset))
            else:
                tmp_result = 0
                # the recursive call
                # checking for the kind of diagram. MTxxx?
                if self.offsets == {}:
                    for edge_name in node.child_nodes:
                        tmp_result += sum_rec(node.child_nodes[edge_name], node.dtype.to_mat(node, 0, 0))
                # or edge-value dd?
                else:
                    for edge_name in node.child_nodes:
                        tmp_result += sum_rec(node.child_nodes[edge_name], node.dtype.to_mat(node,
                                                                                             node.offsets[edge_name],
                                                                                             offset))

                return tmp_result

        return sum_rec(self, None)

    def create_leaf(self, value):
        """
        A convenience method
        """
        return self.leaf_type(value, value, diagram_type=self.dtype)

    def create_node(self, depth=None):
        """
        A convenience method
        """
        if not depth is None:
            return type(self)(diagram_type=self.dtype, nullvalue=self.null_value, depth=depth)
        else:
            return type(self)(diagram_type=self.dtype, nullvalue=self.null_value)

    def complexity(self, mode='#nodes'):
        """
        Calculates and returns the complexity of the diagram
        """
        if mode == '#nodes':
            return len(self.nodes)

    def plot(self, name):
        raise NotImplementedError

    def decompose_paths(self):
        """
        This function decomposes a diagram into the set of its paths from root to leaves
        :param self:
        """
        if self.child_nodes == {}:
            return []

        import numpy as np

        def decompose_paths_rec(node_inner, path):
            """
            This function does the recursive create_path of the decomposition
            :param node_inner:
            :param path:
            """
            if node_inner.is_leaf():
                path = np.append(path, str(node_inner.value))
                return path[None]
            else:
                paths = np.array([])
                for edge_name in node_inner.child_nodes:
                    new_path = np.append(path, str(edge_name))
                    paths = np.append(paths, decompose_paths_rec(node_inner.child_nodes[edge_name], new_path))
            return paths

        decomposition = decompose_paths_rec(self, np.array([]))
        return decomposition.reshape((decomposition.shape[0]/(self.d+1), self.d+1))

    def to_matrix(self, rows=1, cropping=True, outer_offset=None, approximation_precision=0):
        """
        This function computes the approximation of the diagram at the nth level and returns it as a matrix.
        """
        import numpy as np

        def to_matrix_rec(node, offset):
            # making sure the node exists
            if not node:
                return None, 0
            # checking whether the node is a leaf
            elif node.is_leaf():
                return node.dtype.to_mat(node, offset)
            elif node.d == approximation_precision:
                return node.dtype.to_mat(node.leaves.__iter__().next(), offset)
            else:
                # the recursive call
                # mat_shape = node.dtype.base**node.d
                # base_mat = np.ones(mat_shape)*diagram.null_value
                base_mat = np.array([])
                # checking for the kind of diagram. MTxxx?
                if self.offsets == {}:
                    for edge_name in range(node.dtype.base):
                        tmp_result = to_matrix_rec(node.child_nodes[edge_name], node.dtype.to_mat(node, 0, 0))
                        base_mat = np.hstack((base_mat, tmp_result))
                # or edge-value dd?
                else:
                    for edge_name in range(node.dtype.base):
                        tmp_result = to_matrix_rec(node.child_nodes[edge_name],
                                                   node.dtype.to_mat(node, node.offsets[edge_name], offset))
                        try:
                            base_mat = np.hstack((base_mat, tmp_result))
                        except ValueError:
                            base_mat = base_mat[None]
                            base_mat = np.hstack((base_mat, tmp_result))
                return base_mat

        result = to_matrix_rec(self, outer_offset)
        row_vars = np.log10(rows)/np.log10(self.dtype.base)
        # ratio_of_approx = row_vars/(self.d-approximation_precision)
        # rows = self.dtype.base**np.ceil(ratio_of_approx)
        rows_pot = self.dtype.base**np.ceil(row_vars)
        cols_pot = np.max(result.shape)/rows_pot
        result = np.reshape(result, (rows_pot, cols_pot))
        # if desired, crop the result of all zero columns/rows in the lower right
        if cropping and not rows == 1:
            uncropped = True
            while uncropped:
                uncropped = False
                if (result[:, -1] == 0).all():
                    result = result[:, :-1]
                    uncropped = True
                if (result[-1, :] == 0).all():
                    result = result[:-1, :]
                    uncropped = True
        return result

    @property
    def m(self):
        return self.to_matrix()

    def __hash__(self, reinit=False):
        """
        A test function to make nodes hashable. The hash is the address of the python object.
        """
        if not self.hash_value is None and not reinit:
            return self.hash_value
        elif isinstance(self, Leaf):
            self.hash_value = Hash.leaf_hash(self)
            return self.hash_value
        else:
            self.hash_value = Hash.node_hash(self)
            return self.hash_value

    def reduce(self):
        """
        This function reduces a tree, given in node, to the fitting diagram
        :rtype : None - the change will be applied to the argument
        :param self : the tree (or diagram) to be reduced
        """
        # initializing a hashtable for all the nodes in the tree
        hashtable = {}
        for node_it in self.nodes:
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
        reduce_rec(self)
        # reinitializing the diagram
        self.reinitialize()
        return self


class Hash:
    def __init__(self):
        pass

    @staticmethod
    def leaf_hash(leaf):
        return hash(leaf.value)

    @staticmethod
    def node_hash(node):
        return hash(str([edge for edge in node.child_nodes]) + str(node.offsets) + ''.join([repr(abs(node.child_nodes[i]
                                                                                                     .__hash__())) for i
                                                                                            in node.child_nodes]))


class Leaf(Node):
    """
    This special node-type is reserved for modeling the leaves_array of the diagram
    """

    def __init__(self, denominator, val, diagram_type):
        """
        Simply calles the super method and sets the special attribute "value"
        :param denominator:
        :param val:
        """
        Node.__init__(self, denominator, diagram_type=diagram_type)
        self.child_nodes = None
        self.value = val
        self.shape = [1, 1]
        self.d = 0

    @staticmethod
    def add_child(self, child, number, offset=None):
        """
        This method overrides the add_child method of Node, to prevent a leaf with a child
        :param offset:
        :param number:
        :raise Exception:
        """
        raise TerminalNodeException('Trying to add a child to a leaf node.')

    def to_matrix(self, rows=1, cropping=True, outer_offset=None, approximation_precision=0):
        """
        Returns the value of the leaf in numpy-matrix form
        """
        import numpy as np
        return np.array(self.value)
