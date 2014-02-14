# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:39:27 2014

@author: clemens
"""
import numpy as np
from node import *
from diagram_initialization import *

class Diagram(object):
    def __init__(self, ntype, ltype, mat, nv):
        """
        Initializing the Object.
        :param ntype:   type of the nodes (i.e. BNode for binary)
        :param ltype:   type of the leaves
        :param mat:     the data matrix the diagram shall be constructed from
        :param nv:      null value, i.e. the most uninformative entry in the matrix
        """
        self.node_type = ntype
        self.leaf_type = ltype
        self.nodes = {}
        self.leaves = {}
        self.matrix = np.array(mat)
        self.null_value = nv
        self.shape = mat.shape
        self.root = self.initialize()

        
    def initialize(self):
        """
        This function initializes the graph, i.e. creates the diagram from
        the given matrix
        :type self: object
        """
        root = initialize_diagram(self, self.matrix, self.null_value)
        self.set_root(root)
        return root

    def add_node(self, denominator, parent=None):
        """
        This method adds a node to the given diagram, and should only be called
        through the different Diagram-class extensions (like BDiagram)
        node_type specifies the type of the Diagram
        :param denominator:
        :param parent:
        """
        # checking whether node already exists in the diagram
        if denominator in self.nodes:
            raise Exception('Leaf already present in Diagram')
        else:
            # creating the node object
            new_node = self.node_type(denominator,parent)
            # adding the node to the list of nodes in the diagram
            self.nodes[denominator] = new_node
            return  new_node
    
    def set_root(self, ref):
        """
        This function adds the root node to the diagram
        :param ref:
        """
        if type(ref) == 'str':
            node = self.add_node(ref, None)
            self.root = node
            return node
        elif isinstance(ref, self.node_type):
            self.root = ref
        else:
            raise Exception('Trying to access nodes by unknown reference type.')
        
    def add_leaf(self, denominator, value):
        """
        :param denominator:
        :param parent:
        :return: :raise Exception:
        """
        if denominator in self.nodes:
            raise Exception('Leaf already present in Diagram')
        else:
            # creating the node object
            new_node = self.leaf_type(denominator, value)
            # adding the node to the list of nodes in the diagram
            self.leaves[denominator] = new_node
            return  new_node
    
    def add_edge(self, node1, node2):
        """
        This method adds an edge from node1 (parent) to node2 (child)
        :param node1:
        :param node2:
        """
        node2.add_parent(node1)
        node1.add_child(node2)
        return

    def has_leaf(self, ref):
        """
        This method checks whether a leave denoted by a given denominator 
        already exists in the diagram
        :param ref:
        """
        if isinstance(ref,np.string_) or isinstance(ref,str):
            if ref in self.leaves:
                return True
            else:
                return False
        elif isinstance(ref, self.leaf_type):
            if ref in self.nodes.values():
                return True
            else:
                return False
        else:
            raise Exception('Trying to access leaves by unknown reference type.')
            
    def has_node(self, ref):
        """
        This method checks whether a node denoted by a given denominator 
        already exists in the diagram
        :param ref:
        """
        if type(ref) == 'str':
            if ref in self.nodes:
                return True
            else:
                return False
        elif isinstance(ref,self.node_type):
            if ref in self.nodes.values():
                return True
            else:
                return False
        else:
            raise Exception('Trying to access nodes by unknown reference type.')

    def has_entry(self, denominator):
        """
        This function checks if a node or leaf denoted by the given denominator
        already exists in the diagram
        :param denominator:
        """
        return self.has_node(denominator) or self.has_leaf(denominator)
    
    def has_edge(self, node1, node2):
        """
        Returns true if there is an edge from node1 to node2
        :param node1:
        :param node2:
        """
        if node1.has_child(node2):
            return True
        else:
            return False


class BDiagram(Diagram):
    def add_edge(self, node1, node2, bin_type):
        """
        This method adds an edge from node1 (parent) to node2 (child)
        :param node1:
        :param node2:
        """
        if bin_type == 'p':
            node2.add_parent(p=node1)
        elif bin_type == 'n':
            node2.add_parent(n=node1)
        else:
            raise Exception('Unknown edge/child type for a binary diagram.')
        return


    def add_p_edge(self, node1, node2):
        """
        A wrapper for add_edge and positive edges, for pure convenience
        :param node1:
        :param node2:
        """
        self.add_edge(node1, node2, 'p')


    def add_n_edge(self, node1, node2):
        """
        A wrapper for add_edge and negative edges, for pure convenience
        :param node1:
        :param node2:
        """
        self.add_edge(node1, node2, 'n')


mat = np.random.random_integers(0,5,[3,3])
diag = BDiagram(BNode, BLeaf, mat,0)
import code;code.interact(local=dict(locals().items() + globals().items()))
#a=BNode('hallo','x1')
#b=BNode('hallo1','x2',p=a)
#c=BNode('hallo1','x2',p=b)
#d=BNode('hallo1','x2',n=b)