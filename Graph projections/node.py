# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:23:43 2014

@author: clemens
"""
from diagram_initialization import initialize_diagram


class Node(object):
    """
    The generic node class, which should not be used directly in most cases
    (think of it as being like an interface. The Objects to be used should be optimized
    for the specific case, i.e. binary nodes (see class BNode))
    """
    properties = {}
    
    def __init__(self, denominator, depth=None, nv=0, mat=None, var=None):
        """        
        all required information are the name of the node
        :param mat:
        """
        self.name = denominator
        self.d = depth
        self.child_nodes = {}
        self.leaves_set = set()
        self.leaf_type = Leaf
        self.null_value = nv
        self.shape = (0, 0)
        if not var is None:
            self.variable = var
        if not mat is None:
            self.shape = initialize_diagram(self, mat, nv)

    def add_child(self, child, number):
        """        
        setting a node as a child node, where number denotes an internal reference
        """
        if isinstance(child, Node):
            self.child_nodes[number] = child
        elif isinstance(child, Leaf):
            self.child_nodes[number] = child
            self.leaves_set.add(child)
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
        # is the current node a leaf?
        if self.is_leaf():
            return {self}
        # if not, recursively return all children
        else:
            childrens_leaves = set()
            for child in self.child_nodes:
                childrens_leaves = childrens_leaves.union(self.child_nodes[child].leaves)
            # storing it for later use
            self.leaves_set = childrens_leaves
            return childrens_leaves

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


class BNode(Node):
    """
    This class extends Node for binary graphs
    """
    def __init__(self, denominator, depth=None, nv=0, mat=None, var=None):
        """

        :param mat:
        denominator:    the name of the node (str)
        variable:       the variable represented by the node (str)
        """
        Node.__init__(self, denominator, depth, nv, mat, var)
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
        Node.add_child(self, child, 1)

    @property
    def n(self):
        """
        A property for the negative fork, for easier access to the only forks available
        :return:
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
        Node.add_child(self, child, 0)

    def to_matrix(self, rows=1, cropping=False):
        """
        This method returns the matrix represented by the diagram
        :param rows:
        :param cropping:
        """
        import numpy as np

        def to_mat_rec(node, nv):
            # making sure the node exists
            if not node:
                return None, 0
            # checking whether the node is a leaf
            if node.is_leaf():
                return np.array(node.value)[None], 1
            else:
                # the recursive call
                nfork, n_cshape = to_mat_rec(node.n, nv)
                pfork, p_cshape = to_mat_rec(node.p, nv)
                # getting the size for a missing fork
                mat_shape = 2*max(n_cshape, p_cshape)
                if pfork is None:
                    pfork = np.ones(n_cshape)*nv
                if nfork is None:
                    nfork = np.ones(p_cshape)*nv
                # deciding whether the matrices shall be horizontally or vertically concatenated
                return np.concatenate((nfork, pfork), 1), mat_shape
        result, shape = to_mat_rec(self, self.null_value)
        rows = 2**np.ceil(np.log2(rows))
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


class Leaf(Node):
    """
    This special node-type is reserved for modeling the leaves_array of the diagram
    """
    def __init__(self, denominator, val):
        """
        Simply calles the super method and sets the special attribute "value"
        :param denominator:
        :param val:
        """
        Node.__init__(self, denominator, 0)
        self.child_nodes = None
        self.value = val
        self.shape = [1, 1]

    def add_child(self, node, number):
        """
        This method overrides the add_child method of Node, to prevent a leaf with a child
        :param node:
        :param number:
        :raise Exception:
        """
        raise Exception('[ERROR] Trying to add a child to a leaf node.')


class BLeaf(Leaf):
    """
    A special class for leaves_array in binary diagrams
    """
    def __init__(self, denominator, val):
        Leaf.__init__(self, denominator, val)


class NoSuchNode(Exception):
    """
    A slightly more fitting Exception for this usecase :-)
    """
    pass
    #def __init__(self, message):
    #    Exception.__init__(self, message)