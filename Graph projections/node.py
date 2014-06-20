# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:23:43 2014

@author: clemens
"""


class Node(object):
    """
    The generic node class, which should not be used directly in most cases
    (think of it as being like an interface. The Objects to be used should be optimized
    for the specific case, i.e. binary nodes (see class BNode))
    """
    properties = {}
    from diagram_binary import MTBDD

    def __init__(self, denominator='', diagram_type=MTBDD, depth=None, nv=0, mat=None, var='x'):
        """        
        all required information are the name of the node
        :param mat:
        """
        self.name = denominator
        self.d = depth
        self.dtype = diagram_type
        self.child_nodes = {}
        self.offsets = {}
        self.leaves_set = set()
        self.nodes_set = set()
        self.leaf_type = Leaf
        self.null_value = nv
        self.shape = (0, 0)
        self.paths = set()
        self.hash_value = None
        if not var is None:
            self.variable = var
        if not mat is None:
            from diagram_initialization import initialize_diagram
            self.shape = initialize_diagram(self, mat, nv, var)

    def add_child(self, child, number, offset=None):
        """        
        setting a node as a child node, where number denotes an internal reference
        :param offset:
        """
        if isinstance(child, Leaf):
            self.child_nodes[number] = child
            self.leaves_set.add(child)
            if not offset is None:
                self.offsets[number] = offset
        elif isinstance(child, Node):
            self.child_nodes[number] = child
            if not offset is None:
                self.offsets[number] = offset
        else:
            raise Exception('Trying to add a non-node object as child node')

    @property
    def leaves(self):
        """
        This property returns the leaves_array and the end of this diagram
        :rtype : array of leaf-nodes
        :return:
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

    def get_leaf(self, leaf):
        """
        This method returns a boolean value symbolizing whether the leaf is in the leaves_array
        """
        # do we have a node-object passed?
        if isinstance(leaf, Leaf):
            if leaf in self.leaves:
                return leaf
        # if not, look for the name/value
        else:
            for known_leaf in self.leaves:
                if known_leaf.value == leaf:
                    return known_leaf
        #raise NoSuchNode('The leaf '+str(leaf)+' is not a leaf of node ' + self.name)
        raise NoSuchNode('The object '+str(leaf)+' is not a leaf of node ' + self.name)

    def reinitialize_leaves(self):
        """
        This method reinitializes the leaf-array in case some operation on the diagram changed it
        :return: set of all leave nodes
        """
        # TODO: merge reinit and creation of leaves and nodes, for one includes the other
        # is the current node a leaf?
        if self.is_leaf():
            return {self}
        # if not, recursively return all children
        else:
            childrens_leaves = set()
            for child in self.child_nodes:
                childrens_leaves = childrens_leaves.union(self.child_nodes[child].reinitialize_leaves())
            # storing it for later use
            self.leaves_set = childrens_leaves
            return childrens_leaves

    def reinitialize_nodes(self):
        """
        This method returns the nodes_set, but reinitializes it first (if it has changed, e.g. via reduction)
        :rtype : array of leaf-nodes
        :return: set of all nodes in the diagram
        """
        # TODO: merge reinit and creation of leaves and nodes, for one includes the other
        # is the current node a leaf?
        if self.is_leaf():
            return {self}
        # if not, recursively return all children
        else:
            children_nodes = {self}
            for child in self.child_nodes:
                children_nodes = children_nodes.union(self.child_nodes[child].reinitialize_nodes())
            # storing it for later use
            self.nodes_set = children_nodes
            return children_nodes

    def reinitialize(self):
        """
        A method combining the functionality of both reinitialize_nodes and reinitialize_leaves
        :return:
        """
        if self.is_leaf():
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
            self.leaves = children_leaves
            self.nodes = children_nodes
            return children_leaves, children_nodes

    @property
    def nodes(self):
        """
        This property returns the nodes_set
        :rtype : array of leaf-nodes
        :return:
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

    def get_node(self, node):
        """
        This method returns a boolean value symbolizing whether the leaf is in the leaves_array
        """
        # do we have a node-object passed?
        if isinstance(node, Node):
            if node in self.nodes:
                return node
        # if not, look for the name/value
        else:
            for known_node in self.nodes:
                if known_node.denominator == str(node):
                    return known_node
        #raise NoSuchNode('The leaf '+str(leaf)+' is not a leaf of node ' + self.name)
        raise NoSuchNode('The object '+str(node)+' is not a leaf of node ' + self.name)

    def is_child(self, node):
        """
        This function checks whether a node already is a child of the given node
        """
        if isinstance(node, Node):
            return node in self.child_nodes.values()
        else:
            raise NoSuchNode('Unrecognized node-reference')

    def remove_child(self, node):
        """
        Removes the specified node as a child
        """
        for child in self.child_nodes:
            if node == self.child_nodes[child]:
                self.child_nodes.pop(child)
                return
        raise Warning('Trying to remove child '+node.name+', which is not child of node '+self.name)

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
        subdiagrams = []

        def get_sds_rec(node, sds, level, cur_level):
            if level == cur_level:
                sds.append(node)
            else:
                # the children have to be sorted for some usages
                children = []
                for child in node.child_nodes.keys():
                    children.append(child)
                children.sort()
                for child in children:
                    get_sds_rec(node.child_nodes[child], sds, level, cur_level+1)
        get_sds_rec(self, subdiagrams, depth, 0)
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

    def get_paths(self, refresh=False):
        """
        A helper method to construct all paths derivating from this node. Returning a set.
        """

        def paths_rec(node, path, encountered_paths, refr):
            """
            The recursive counterpart
            """
            if node.paths == set() or refr:
                # checking for leaves
                if isinstance(node, Leaf):
                    node.paths = {node.value}
                    return path+str(node.value)
                else:
                    # iterating over child nodes
                    for child in node.child_nodes:
                        # collecting offset strings
                        if not node.child_nodes[child][1] is None:
                            offset_string = str(node.child_nodes[child][1])
                        else:
                            offset_string = ''
                        # creating the path string
                        new_path = path + str(child) + offset_string + ','
                        path1 = paths_rec(node.child_nodes[child][0], new_path, encountered_paths, refr)
                        if isinstance(path1, str):
                            encountered_paths.append(path1)
                    return encountered_paths
            else:
                return self.paths

        return paths_rec(self, '', [], refresh)

    def add(self, node, **offset):
        """
        This method adds the current node and the argument, returning a new diagram
        """
        return self.dtype.add(self, node, **offset)

    def sum(self, **offset):
        """
        This method sums the current node
        """
        return self.dtype.sum(self, **offset)

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
            return type(self)(diagram_type=self.dtype, nv=self.null_value, depth=depth)
        else:
            return type(self)(diagram_type=self.dtype, nv=self.null_value)

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

    def to_matrix(self, rows=1, cropping=False):
        """
        This method returns the matrix represented by the diagram
        :param rows:
        :param cropping:
        """
        import numpy as np

        # covering zero-matrices
        if self.child_nodes == {}:
            return np.array([self.null_value])

        def to_mat_rec(node, offset, nv):
            # making sure the node exists
            if not node:
                return None, 0
            # checking whether the node is a leaf
            elif node.is_leaf():
                return node.dtype.to_mat(node, offset), 1
            else:
                # the recursive call
                mat_shape = node.dtype.base**node.d
                base_mat = np.ones(mat_shape)*nv
                if self.offsets == {}:
                    pos_counter = 0
                    for edge_name in node.child_nodes:
                        base_mat[pos_counter*mat_shape/node.dtype.base:(pos_counter+1)*mat_shape/node.dtype.base], _ = \
                            to_mat_rec(node.child_nodes[edge_name], node.dtype.to_mat(node, 0, 0), nv)
                        pos_counter += 1
                else:
                    pos_counter = 0
                    for edge_name in node.child_nodes:
                        base_mat[pos_counter*mat_shape/node.dtype.base:(pos_counter+1)*mat_shape/node.dtype.base], _ = \
                            to_mat_rec(node.child_nodes[edge_name],
                                       node.dtype.to_mat(node, node.offsets[edge_name], offset), nv)
                        pos_counter += 1

                return base_mat, mat_shape

        result, shape = to_mat_rec(self, None, self.null_value)
        rows = self.dtype.base**np.ceil(np.log(rows)/np.log(self.dtype.base))
        result = np.reshape(result, (rows, shape/rows))
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

    def __hash__(self):
        """
        A test function to make nodes hashable. The hash is the address of the python object.
        """
        if not self.hash_value is None:
            return self.hash_value
        elif isinstance(self, Leaf):
            self.hash_value = Hash.leaf_hash(self)
            return self.hash_value
        else:
            self.hash_value = Hash.node_hash(self)
            return self.hash_value


class Hash:
    @staticmethod
    def leaf_hash(leaf):
        return hash(leaf.value)

    @staticmethod
    def node_hash(node):
        return hash(str([edge for edge in node.child_nodes]) + str(node.offsets) \
                          + ''.join([repr(abs(node.child_nodes[i].__hash__())) for i in node.child_nodes]))


class BNode(Node):

    """
    This class extends Node for binary graphs
    """
    from diagram_binary import MTBDD

    def __init__(self, denominator='', diagram_type=MTBDD, depth=None, nv=0, mat=None, var=None):
        """

        :param mat:
        denominator:    the name of the node (str)
        variable:       the variable represented by the node (str)
        """
        Node.__init__(self, denominator, diagram_type, depth, nv, mat, var)
        self.edge_values = {}
        self.leaf_type = BLeaf

    @property
    def p(self):
        """
        A property for the positive fork, for easier access to the only forks available
        :return:
        """
        if 1 in self.child_nodes:
            return self.child_nodes[1]
        else:
            return False

    @p.setter
    def p(self, child):
        """
        A setter for the positive fork, for easier access to the only forks available
        :param child:
        """
        try:
            self.add_child(self, child[0], 1)
            self.child_nodes[1].append(child[1])
        except TypeError:
            Node.add_child(self, child, 1)

    @property
    def n(self):
        """
        A property for the negative fork, for easier access to the only forks available
        :return:
        :type return: BNode
        """
        if 0 in self.child_nodes:
            return self.child_nodes[0]
        else:
            return False

    @n.setter
    def n(self, child):
        """
        A setter for the positive fork, for easier access to the only forks available
        :param child:
        """
        try:
            self.add_child(self, child[0], 0)
            self.offsets[0] = child[1]
        except TypeError:
            Node.add_child(self, child, 0)

    @property
    def po(self):
        """
        A property for the positive fork offset
        :return:
        """
        if 1 in self.offsets:
            return self.offsets[1]
        else:
            return False

    @po.setter
    def po(self, offset):
        """
        A setter for the positive fork offset
        :param offset:
        """
        self.offsets[1] = offset

    @property
    def no(self):
        """
        A property for the negative fork offset
        :return:
        :type return: BNode
        """
        if 0 in self.offsets:
            return self.offsets[0]
        else:
            return False

    @no.setter
    def no(self, offset):
        """
        A setter for the positive fork offset
        :param offset:
        """
        self.offsets[0] = offset

    def plot(self, name):
        """
        This function plots the diagram using GraphViz and DOT (https://en.wikipedia.org/wiki/DOT_language)
        """
        import pydot as pd
        graph = pd.Dot('diagram', graph_type='digraph')
        for node in self.nodes:
            if node.is_leaf():
                graph.add_node(pd.Node(str(node.__hash__()), label=str(node.value), shape='box'))
            else:
                graph.add_node(pd.Node(str(node.__hash__()), label='L'+str(node.d)))
                if node.n:
                    if node.no:
                        graph.add_edge(pd.Edge(str(node.__hash__()), str(node.n.__hash__()), label=str(node.no), style='dashed'))
                    else:
                        graph.add_edge(pd.Edge(str(node.__hash__()), str(node.n.__hash__()), style='dashed'))
                if node.p:
                    if node.po:
                        graph.add_edge(pd.Edge(str(node.__hash__()), str(node.p.__hash__()), label=str(node.po)))
                    else:
                        graph.add_edge(pd.Edge(str(node.__hash__()), str(node.p.__hash__())))

        file_name = './plots/' + name + '.png'
        graph.write_png(file_name)


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

    def add_child(self, child, number, offset=None):
        """
        This method overrides the add_child method of Node, to prevent a leaf with a child
        :param offset:
        :param number:
        :raise Exception:
        """
        raise TerminalNode('Trying to add a child to a leaf node.')

    def to_matrix(self, shape=(1, 1), resize=False):
        """
        Returns the value of the leaf in numpy-matrix form
        """
        print('This is a leaf. The shape ('+str(shape)+') and resize ('+str(resize)+') parameters are ignored')
        import numpy as np
        return np.array(self.value)


class BLeaf(Leaf):
    """
    A special class for leaves_array in binary diagrams
    """
    from diagram_binary import MTBDD

    def __init__(self, denominator, val, diagram_type=MTBDD):
        Leaf.__init__(self, denominator, val, diagram_type=diagram_type)


class NoSuchNode(Exception):
    """
    A slightly more fitting Exception for this usecase :-)
    """
    pass


class TerminalNode(Exception):
    """
    A slightly more fitting Exception for this usecase :-)
    """
    pass