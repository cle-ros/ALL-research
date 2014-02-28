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
    
    def __init__(self, denominator, var, **parents):
        """        
        all required information are the name of the node
        """
        self.name = denominator
        self.child_nodes = {}
        self.parent_nodes = []
        self.variable = var
        for p in parents:
            self.add_parent(parents[p])

    def set_child(self, child, number):
        """        
        setting a node as a child node, where number denotes an internal reference
        """
        if isinstance(child, Node):
            self.child_nodes[number] = child
        else:
            raise Exception('Trying to add a non-node object as child node')

    def add_parent(self, parent):
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

    def remove_parent(self, node):
        """
        Removes the specified node as parent
        """
        if node in self.parent_nodes:
            self.parent_nodes.remove(node)
            return
        else:
            raise Warning('Trying to remove parent '+node.name+', which is not parent of node '+self.name)

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
    def __init__(self, denominator, variable, **parents):
        """
        denominator:    the name of the node (str)
        variable:       the variable represented by the node (str)
        **parents:      a dictionary of parents; p if the node is the positive
                        child, n if the node is the negative child
        """
#        super(BNode, self).__init__(denominator,variable)
        Node.__init__(self, denominator, variable)
        self.add_parent(parents)

    def add_parent(self, parents):
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
        elif parents == {}:
            return
            #raise Warning('Method add_parent called, but list of parents empty.')
        else:
            raise Exception('unknown parent type for a binary node')
    # a @property for easy navigation in binary diagrams        
    @property
    def p(self):
        if 1 in self.child_nodes:
            return self.child_nodes[1]
        else:
            return False
    @p.setter
    def p(self,child):
        Node.set_child(self,child,1)
    # a @property for easy navigation in binary diagrams
    @property
    def n(self):
        if 0 in self.child_nodes:
            return self.child_nodes[0]
        else:
            return False
    @n.setter
    def n(self,child):
        Node.set_child(self, child, 0)

class Leaf(Node):
    """
    This special node-type is reserved for modeling the leaves of the diagram
    """
    def __init__(self, denominator, val, **parents):
        Node.__init__(self, denominator, 'Value')
        self.child_nodes = None
        self.value = val
        self.add_parent(parents)
    def set_child(self):
        raise Exception('[ERROR] Trying to add a child to a leaf node.')
        return None

class BLeaf(Leaf):
    """
    A special class for leaves in binary diagrams
    """
    def add_parent(self, parents):
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
        elif parents == {}:
            return
        else:
            raise Exception('unknown parent type for a binary node')