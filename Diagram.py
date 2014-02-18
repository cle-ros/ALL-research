# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:39:27 2014

@author: clemens
"""
import numpy as np
from Node import *
from computeDiagram import *

class Diagram(object):
    def __init__(self,ntype,ltype,mat,nv):
        """
        Initializing the Object.
        Args:
            ntype:  type of the nodes (i.e. BNode for binary)
            ltype:  type of the leaves
            mat:    the data matrix the diagram shall be constructed from
            nv:     null value, i.e. the most uninformative entry in the matrix
        """
        self.node_type = ntype
        self.leaf_type = ltype
        self.matrix = np.array(mat)
        self.null_value = nv
        self.shape = mat.shape
        self.root = self.initialize()
        self.entries = {}
        self.leaves = {}
        
    def initialize(self):
        """
        This function initializes the graph, i.e. creates the diagram from
        the given matrix
        """
        root = computeDiagram(self,self.matrix,self.nullValue)
        self.add_root(root)
        return root

    def add_node(self,denominator,parent=None):
        """
        This method adds a node to the given diagram, and should only be called
        through the different Diagram-class extensions (like BDiagram)
        node_type specifies the type of the Diagram
        """
        # checking whether node already exists in the diagram
        if denominator in self.entries:
            raise Exception('[ERROR] Leaf already present in Diagram')
        else:
            # creating the node object
            new_node = self.leaf_type(denominator,parent)
            # adding the node to the list of nodes in the diagram
            self.entries[denominator] = new_node
            return  new_node
    
    def add_root(self,denominator):
        """
        This function adds the root node to the diagram
        """
        node = self.add_node(denominator, None)
        self.root = node
        return node
        
    def add_leaf(self,denominator,parent):
        if denominator in self.entries:
            raise Exception('[ERROR] Leaf already present in Diagram')
            return
        else:
            # creating the node object
            new_node = self.leaf_type(denominator,parent)
            # adding the node to the list of nodes in the diagram
            self.leaves[denominator] = new_node
            return  new_node
    
    def add_edge(self,node1,node2):
        """
        This method adds an edge from node1 (parent) to node2 (child)
        """
        node2.add_parent(node1)
        node1.add_child(node2)

    def has_leave(self,denominator):
        """
        This method checks whether a leave denoted by a given denominator 
        already exists in the diagram
        """
        if denominator in self.leaves:
            return True
        else:
            return False
            
    def has_node(self,denominator):
        """
        This method checks whether a node denoted by a given denominator 
        already exists in the diagram
        """
        if denominator in self.leaves:
            return True
        else:
            return False

    def has_entry(self,denominator):
        """
        This function checks if a node or leaf denoted by the given denominator
        already exists in the diagram
        """
        return self.has_node(denominator) or self.has_leave(denominator)
    
    def has_edge(self,node1,node2):
        """
        Returns true if there is an edge from node1 to node2
        """
        if node1.child_exists(node2):
            return True
        else:
            return False

diag = Diagram
a=BNode('hallo','x1')
b=BNode('hallo1','x2',p=a)
c=BNode('hallo1','x2',p=b)
d=BNode('hallo1','x2',n=b)