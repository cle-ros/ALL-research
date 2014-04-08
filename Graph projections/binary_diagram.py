# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""


class BinaryDiagram:
    def __init__(self, nt, lt):
        self.node_type = nt
        self.leaf_type = lt
        self.paths = self.get_path()

    def get_path(self):
        raise NotImplementedError

    def create_leaf(self, value):
        raise NotImplementedError

    def create_tuple(self, nodes, known_paths):
        raise NotImplementedError

    def compute_edge_values(self, nodes):
        raise NotImplementedError

    def create_tuple(self, nodes, known_paths, null_value=0):
        """
        This method defines the general framework to branch. It relies on specific implementations for the different
         binary diagram types
        """
        # creating the node
        node = self.node_type('')
        index = 0
        # creating a container to reference all encountered nodes
        existing_child_nodes = {}
        exists = False
        # loop through all new children
        for child in nodes:
            # if they are leaves, they are denominated by floats
            if isinstance(child, float):
                # does the path exist?
                leaf, new_node_den, offset = self.create_leaf(child, index)
                if new_node_den in known_paths[0]:
                    leaf = known_paths[1][known_paths[0] == new_node_den][0]
                    exists = True
                else:
                    # is the path 0? we have zero-suppressed diagrams, so ignore it then
                    if not child == null_value:
                        leaf = self.leaf_type(child, child)
                        exists = True
                        known_paths[1].append(leaf)
                        known_paths[0].append(leaf.get_paths())
                if exists:
                    # add the child
                    node.add_child(leaf, index, None)
                    existing_child_nodes[child] = leaf
            # if both children are nodes already
            else:
                exists = True
                if child.get_path() in known_paths:
                    node.add_child(known_paths[child], index, None)
                else:
                    node.add_child(child, index, None)
                existing_child_nodes[index] = child
            index += 1
            if exists:
                known_paths[node] = node
        node.edgevalues = self.compute_edge_values(node, existing_child_nodes)
        return node

class MTBDD(BinaryDiagram):
    def __init__(self, nt, lt):
        super(nt, lt)

    def create_leaf(self, value):
        return self.leaf_type(value, value), denominator, offset

    def compute_edge_values(self, nodes):
        if nodes[0]:
            if not isinstance(nodes[0], self.leaf_type):
                #compute the different values using the offset of the nodes
                # and the parent nodes given
                return

    def create_path(self, name_of_node):
        """
        This method defines the create_path to the node
        """
        raise NotImplementedError

    def create_tuple(self, nodes, known_paths):
        BinaryDiagram.create_tuple(self, nodes, known_paths)
