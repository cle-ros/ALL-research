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
    
    def __init__(self, denominator, nv=0, mat=None, var=None):
        """        
        all required information are the name of the node
        :param mat:
        """
        self.name = denominator
        self.child_nodes = {}
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
            if isinstance(child, Leaf):
                self.leaves_array.append(child)
            else:
                self.leaves_array.extend(child.leaves_array)
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
            return [self], self.value
        # or does it already have leaf-entries?
        elif not self.leaves_array is []:
            return self.leaves_array
        # if not, recursively return all children
        else:
            children = []
            for child in self.child_nodes:
                children.extend(self.child_nodes[child].leaves_array)
            # storing it for later use
            self.leaves_array = children
            return children

    @leaves.setter
    def leaves(self, leaves_array):
        """
         The leaf - setter function.
        """
        self.leaves_array = leaves_array

    def get_leaf(self, leaf):
        """
        This method returns a boolean value symbolizing whether the leaf is in the leaves_array
        """
        # do we have a node-object passed?
        if isinstance(leaf, Node):
            return leaf in self.leaves_array, leaf
        # if not, look for the name/value
        else:
            for leaf in self.leaves_array:
                if leaf.value == leaf:
                    return True, leaf
        #raise NoSuchNode('The leaf '+str(leaf)+' is not a leaf of node ' + self.name)
        raise NoSuchNode('The leaf '+str(leaf)+' is not a leaf of node ' + self.name)

    def has_child(self,node):
        """
        This function checks whether a node already is a child of the given node
        """
        if isinstance(node, Node):
            if node in self.child_nodes.values():
                return True
            else:
                return False
        else:
            raise Exception('Unrecognized node-reference')

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
        if isinstance(self, Leaf):
            print 'isleaf'
            print self.value
            return True
        else:
            return False


class BNode(Node):
    """
    This class extends Node for binary graphs
    """
    def __init__(self, denominator, nv=0, mat=None, var=None):
        """

        :param mat:
        denominator:    the name of the node (str)
        variable:       the variable represented by the node (str)
        """
        Node.__init__(self, denominator, nv, mat, var)
        self.leaf_type = BLeaf

    # a @property for easy navigation in binary diagrams
    @property
    def p(self):
        if 1 in self.child_nodes:
            return self.child_nodes[1]
        else:
            return False
    @p.setter
    def p(self, child):
        Node.add_child(self, child, 1)
    # a @property for easy navigation in binary diagrams
    @property
    def n(self):
        if 0 in self.child_nodes:
            return self.child_nodes[0]
        else:
            return False
    @n.setter
    def n(self, child):
        Node.add_child(self, child, 0)

    def to_matrix(self):
        """
        This method returns the matrix represented by the diagram
        """
        import numpy as np
        def to_mat_rec(node, depth, shape, nv):
            # making sure the node exists
            if not node:
                return None
            # checking whether the node is a leaf
            if node.is_leaf():
                print node.value
                return np.array(node.value)[None]
            else:
                # the recursive call
                nfork = to_mat_rec(node.n, depth+1, shape, nv)
                pfork = to_mat_rec(node.p, depth+1, shape, nv)
                # getting the size for a missing fork
                try:
                    shape = nfork.shape
                except AttributeError:
                    shape = pfork.shape
                print shape
                if pfork is None:
                    pfork = np.ones(shape)*nv
                if nfork is None:
                    nfork = np.ones(shape)*nv
                # deciding whether the matrices shall be horizontally or vertically concatenated
                if depth > shape:
                    print 'concatenating horizontally'
                    print nfork
                    print pfork
                    return np.concatenate((nfork, pfork))
                else:
                    print 'concatenating vertically'
                    return np.array([nfork, pfork])
        return to_mat_rec(self, 0, self.shape, self.null_value)


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
        Node.__init__(self, denominator)
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