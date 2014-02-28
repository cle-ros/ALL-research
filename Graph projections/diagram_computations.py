# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:22 2014

@author: clemens
"""
from node import *
from diagram import *
import copy

def diagram_shallow_copy(diagram):
    """
    This function creates a new diagram based on an existing one as "blueprint".
    I.e. the result will have the same shape
    """
    new_diag = type(diagram)(diagram.node_type, diagram.leaf_type, diagram.null_value)
    new_diag.shape = diagram.shape
    new_diag.no_variables = diagram.no_variables
    return new_diag


def create_target_diagram(dia1, dia2, **options):
    """
    This function first checks whether two diagrams, dia1 and dia2, are compatible, i.e. whether they represent to
    matrices of the same dimension.
    If so, the function creates a diagram of the same dimension, which can be populated with some operation on the two
    operands.
    :param dia1: one diagram (in order)
    :param dia2: another diagram (in order)
    :param options:
        diagram in_place    if a diagram is specified via the in_place parameter, that diagram will be overwritten
                            by the new diagram
        double null_value   specifies the null_value for the new diagram; defaults to the null_value of dia1 and dia2
    :return: :raise Exception:
    """
    null_value = None
    in_place = False
    # option checking
    if 'null_value' in options:
        null_value = options['null_value']
    if 'in_place' in options:
        if isinstance(options['in_place'],Node):
            in_place = options['in_place']
        else:
            raise Exception('Cannot create the resulting diagram in the place of '+options['in_place'])

    # making sure the two diagrams each have the same shape
    if not dia1.shape == dia2.shape:
        raise Exception('Can only add two diagrams if the underlying data is of the same dimensionality.')

    # checking for the same nullValue
    if not dia1.null_value == dia2.null_value and not null_value is None:
        raise Exception('The null-value of the two graphs differ. Adjust the null-value of one of the graphs first or '
                        'force an override with the null_value option.')

    result = object
    if in_place:
        result = in_place
        in_place.nodes = {}
        in_place.leaves = {}
        in_place.root = None
        result = in_place
    else:
        result = diagram_shallow_copy(dia1)
    return result

def add_diagrams(dia1, dia2, **options):
    """
    This function adds two graphs. The underlying logic represents matrix addition, and therefore requires the same
     shape of the two diagrams.

    This function does a bottom-up addition
    # the addition sequence:
    1. Go down the diagram, starting with the same variable for each.
    2. for each edge, do the following:
        a. if the edge exists in both diagrams, add the corresponding node to the new diagram and look at both
            subdiagrams for that new node, and goto 2
        b. if the edge does only exist in one diagram, add the corresponding node to the new diagram and ignore the
            subdiagram (i.e. deepcopy the whole subdiagram)
    3. clean up
    """
    result = create_target_diagram(dia1, dia2, **options)
    root = add_diagrams_rec(result, dia1.root, dia2.root)
    result.set_root(root)
    return result

def add_diagrams_rec(diagram, node1, node2, parents={}):
    """
    This function adds two subdiagrams specified by the respective root_node, node1/node2
    """
    # checking for the type of node:
    if node1.is_leaf() and node2.is_leaf():
        value = node1.value+node2.value
        print 'leaf'
        print value
        if value == 0:
            return False
        leaf = diagram.add_leaf(str(value), value)
        if 'p' in parents:
            diagram.add_p_edge(parents['p'], leaf)
        if 'n' in parents:
            diagram.add_n_edge(parents['n'], leaf)
        return leaf
    else:
        # checking for the cases in which a fork exists in both diagrams
        node = diagram.add_node(node1.name, node1.variable)
        node.add_parent(parents)
        # checking for the positive fork:
        if node1.p and node2.p:
            p_edge = add_diagrams_rec(diagram, node1.p, node2.p, {'p': node})
            if p_edge:
                print 'p_edge'
                node.p = p_edge
            else:
                return False
        # checking for the negative fork:
        if node1.n and node2.n:
            n_edge = add_diagrams_rec(diagram, node1.n, node2.n, {'n': node})
            if n_edge:
                print 'n_edge'
                node.n = n_edge
            else:
                return False
        # checking for forks off node1 and not off node2
        if node1.p and not node2.p:
            node.p = copy.deepcopy(node1.p)
        if node1.n and not node2.n:
            node.n = copy.deepcopy(node1.n)
        # checking for forks off node2 and not off node1
        if node2.p and not node1.p:
            node.p = copy.deepcopy(node2.p)
        if node2.n and not node1.n:
            node.n = copy.deepcopy(node2.n)
        return node

def elementwise_multiply_diagrams(dia1, dia2, **options):
    """
    This method multiplies two diagrams element-wise, i.e. the MATLAB .* operation.
    :param dia1:
    :param dia2:
    :return: a diagram representing the element-wise matrix product
    """
#    opt = convert_options(options)
    result = create_target_diagram(dia1, dia2, **options)
    result.root = elementwise_multiply_diagrams_rec(result, dia1.root, dia2.root)
    return result

def elementwise_multiply_diagrams_rec(diagram, node1, node2, parent={}):
    """
    This function multiplies two subdiagrams specified by the respective root_node, node1/node2
    """
    # checking for the type of node:
    if node1.is_leaf() and node2.is_leaf():
        value = node1.value*node2.value
        leaf = diagram.add_leaf(str(value), value)
        if 'p' in parent:
            diagram.add_p_edge(parent['p'], leaf)
        if 'n' in parent:
            diagram.add_p_edge(parent['n'], leaf)
        return leaf
    else:
        # checking for the cases in which a fork exists in both diagrams
        node_p = node1.p and node2.p
        node_n = node1.n and node2.n
        if node_p or node_n:
            node = diagram.add_node(node1.name, node1.variable)
            node.add_parent(parent)
            # checking for the positive fork:
            if node_p:
                p_edge = elementwise_multiply_diagrams_rec(diagram, node1.p, node2.p, {'p':node})
                if p_edge:
                    node.p = p_edge
                else:
                    return False
            # checking for the negative fork:
            if node_n:
                n_edge = elementwise_multiply_diagrams_rec(diagram, node1.n, node2.n, {'n':node})
                if n_edge:
                    node.n = n_edge
                else:
                    return False
            return node
        else:
            return False


def skalar_multiply_diagram(diag1, skalar):
    """
    This function multiplies all leaves in the diagram by a skalar
    :param diag1:
    :param skalar:
    :return:
    """
    result = copy.deepcopy(diag1)
    for leaf in result.leaves:
        result.leaves[leaf].value = result.leaves[leaf].value * skalar
    return result


def get_subdiagrams(diagram, n):
    """
    This function takes a diagram and returns a list of all the subdiagrams at the nth level
    :param diagram:
    :param n:
    """
    subdiags = []
    def get_subdiags_rec(node, level):
        # a nested function for the recursive call
        if level == n:
            subdiags.append(diagram)
        else:
            for child in node.child_nodes:
                get_subdiags_rec(node.child_nodes[child])
    return subdiags


def transpose_diagram(diagram):
    """
    This transposes the underlying matrix. The basic operation is to exchange the horizontal with the vertical variables,
    or, in diagram-terms, to exchange the upper half of the diagram representing the vertical variables with the lower
    half, representing the horizontal variables.
    :param diagram:
    """
    result = create_target_diagram(diag1, diag1)
    steps_down = diagram.shape[0]
    subdiagrams = get_subdiagrams(diagram.root, steps_down-1)
    #UNFINISHED
    return


def multiply_diagram(diag1, diag2, **options):
    """
    This function multiplies two diagrams by the logic of the matrix multiplication
    """
    result = create_target_diagram(diag1, diag2, **options)
    #UNFINISHED
    return


if __name__ == "__main__":
    #mat1 = np.random.random_integers(0,5,[3,3])
    #mat2 = np.random.random_integers(-5,0,[3,3])
    mat1 = np.array([[1,2,0],[0,2,0],[0,2,1]])
    mat2 = np.array([[0,-2,0],[0,-2,0],[0,-2,-1]])
    diag1 = BDiagram(BNode, BLeaf, 0, mat1)
    diag2 = BDiagram(BNode, BLeaf, 0, mat2)
    print mat1
    print mat2
    print mat1+mat2
    diag3 = add_diagrams(diag1, diag2)
    print diag3.to_matrix()
    import code;code.interact(local=dict(locals().items() + globals().items()))
    #a=BNode('hallo','x1')
    #b=BNode('hallo1','x2',p=a)
    #c=BNode('hallo1','x2',p=b)
    #d=BNode('hallo1','x2',n=b)