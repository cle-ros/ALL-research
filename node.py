# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:23:43 2014

@author: clemens
"""

class Node(object):
    """
    The generic node class, which should not be used directly (it is more like
    an interface)
    """
    properties = {}
    
    def __init__(self, denominator, var):
        """        
        all required information are the name of the node
        """
        self.name = denominator
        self.child_nodes = {}
        self.parent_nodes = []
        self.variable = var

    def set_child(self,child,number):
        """        
        setting a node as a child node, where number denotes an internal reference
        """
        if isinstance(child, Node):
            self.child_nodes[number] = child
        else:
            raise Exception('Trying to add a non-node object as child node')
    
    def add_parent(self,parent):
        """
        setting a parent
        """
        if isinstance(parent, Node):
            self.parent_nodes.append(parent)
        else:
            raise Exception('Trying to add a non-node object as parent node')
    
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

    def has_parent(self,node):
        """
        This function checks whether a node already is a parent of the given node
        """
        if isinstance(node, Node):
            if node in self.parent_nodes:
                return True
            else:
                return False
        else:
            raise Exception('Unrecognized node-reference')

class BNode(Node):
    """
    This class extends Node for binary graphs
    """
    def __init__(self,denominator,variable,**parents):
        """
        denominator:    the name of the node (str)
        variable:       the variable represented by the node (str)
        **parents:      a dictionary of parents; p if the node is the positive
                        child, n if the node is the negative child
        """
#        super(BNode, self).__init__(denominator,variable)
        Node.__init__(self,denominator,variable)
        if not parents == {}:
            self.add_parent(parents)

    def add_parent(self, **parents):
        """
        A function overwriting the add_parent method of Node, and extending
        it to include special operations for binary diagrams
        """
        # creating a positive edge from the parent
        if 'p' in parents:
            Node.add_parent(self, parents['p'])
            parents['p'].p = self
        elif 'n' in parents:
            # creating a negative edge from the parent 
            Node.add_parent(self, parents['n'])
            parents['n'].n = self
        else:
            raise Exception('unknown parent type for a binary node')
    # a @property for easy navigation in binary diagrams        
    @property
    def p(self):
        return self.child_nodes[1]
    @p.setter
    def p(self,child):
        Node.set_child(self,child,1)
    # a @property for easy navigation in binary diagrams
    @property
    def n(self):
        return self.child_nodes[0]
    @n.setter
    def n(self,child):
        Node.set_child(self, child, 0)

class Leaf(Node):
    """
    This special node-type is reserved for modeling the leaves of the diagram
    """
    def __init__(self, denominator, val):
        Node.__init__(self,denominator,'Value')
        self.child_nodes = None
        self.value = val
    def set_child(self):
        raise Exception('[ERROR] Trying to add a child to a leaf node.')
        return None

class BLeaf(Leaf):
    """
    A special class for leaves in binary diagrams
    """
    def add_parent(self, **parents):
        """
        A function overwriting the add_parent method of Node, and extending
        it to include special operations for binary diagrams
        """
        # creating a positive edge from the parent
        if 'p' in parents:
            Node.add_parent(self, parents['p'])
            parents['p'].p = self
        elif 'n' in parents:
            # creating a negative edge from the parent
            Node.add_parent(self, parents['n'])
            parents['n'].n = self
        else:
            raise Exception('unknown parent type for a binary node')