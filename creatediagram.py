# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:25:35 2014

@author: clemens
"""

import networkx as nx

import matplotlib.pyplot as plt

try:
    from networkx import graphviz_layout
except ImportError:
    raise ImportError("This example needs Graphviz and either PyGraphviz or Pydot")
#import pygraphviz as pgv
import numpy as np
import re

from matrixAndVariableOperations import *

def getConjunctions(leaves):
    # convert the list of leaves to a dictionary for the different values
    formula = {}
    for comp in leaves:
        if not comp[1] in formula:
            formula[comp[1]] = []
        formula[comp[1]].append(comp[0])
    return formula

def getNonNullComponents(matrix,nullValue,noVars):
    """
    This function gets all indices for all non-null values inside of a given
    matrix
    """
    mat = matrix - np.ones(matrix.shape)*nullValue
    # getting all non-null indices
    row,col = np.nonzero(mat)
    leaves = []
    indices = []
    # cycling through those indices
    for i in range(row.shape[0]):
        # converting the integer indices to boolean ones
        r = re.sub('0b','',bin(row[i]))
        c = re.sub('0b','',bin(col[i]))
        ind = list('0'*(noVars[1]-len(r)) +r+ '0'*(noVars[2]-len(c))+c)
        indices.append(ind)
        # appending the according matrix entry
        leaves.append(matrix[row[i],col[i]])
    return np.array(indices,dtype=int),np.array(leaves)

def sortVariables(indices,variables):
    """
    This method sorts the variables of a given boolean function
    DEPRECATED, should not be used
    """
    # the sorting is rather simple - it sorts by the order of variables having
    # one value

    # Sorting the variables by occurrances of 0 or 1, i.e. by "randomness"
    order = np.argsort(np.maximum(sum(indices,axis=0),sum(np.logical_not(indices),axis=0)))
    variables=np.array(variables)
#    return indices.T[order].T,variables[order],order
    return indices,variables,order


#def appendNodes(tree,leaves,parentNode,varNames,step):
#    posLeaves = []
#    negLeaves = []
#    for leaf in leaves:
#        if len(leaf[0]) >1:
#            if leaf[0][0] == '1':
#                posLeaves.append([leaf[0][1:],leaf[1])
#            else:
#                negLeaves.append([leaf[0][1:],leaf[1])
#        else:
#            
#    if posLeaves != []:
#        node = varNames[step]+'('+parentNode+')'
#        if not node in tree.nodes():
#            tree.add_node(node)
#        tree.add_edge(parentNode,node)
#        appendNodes(tree,posChildren,node,varNames,step+1)
#    if negLeaves != []:
#        node = varNames[step]+'('+'-'+parentNode+')'
#        if not node in tree.nodes():
#            tree.add_node(node)
#        tree.add_edge(parentNode,node)
#        appendNodes(tree,posChildren,node,varNames,step+1)

def computeDiagram(tree,boolMat,leaves,varNames,matSize):
    """ 
    This function computes a diagram from a given tree-object, a matrix
    containing all paths to non-null values in a boolean form, the according 
    leaves and the name of the variables.
    """
    # concatenating the boolean matrix, the leaves and the variable names
    # to one large, and easier to handle, matrix
    tmpMat = np.append(varNames[None],boolMat,axis=0)
    tmpMat = np.append(tmpMat,np.append([-1],leaves,axis=1)[None].T,axis=1)
    tree.add_node(varNames[0])
#    tree.set_node_attributes(tree,size,)
#    tree.get_node(varNames[0])['Size'] = matSize
    # calling the function which builds the tree recursively
    tree = appendNodesRecursively(tree,tmpMat,varNames[0])
    return tree

def appendNodesRecursively(tree,children,parent):
    """
    This function appends nodes to a given parent recursively, based on the 
    specifications given in the 'children' matrix
    """
    # appending leaves
    if children.shape[1] == 2:
        for i in range(children.shape[0]-1):
            # does the leave exist?
            if not tree.has_node(children[i+1,1]):
                tree.add_leaf(children[i+1,1])
            # deciding on the edge to the leave (false/pos)
            if children[i+1,0] == '0':
                tree.add_edge(parent,children[i+1,1],value=0)
            if children[i+1,0] == '1':
                tree.add_edge(parent,children[i+1,1],value=1)
        return tree
    # appending nodes
    else:
        # dividing the matrix into the positive and the negative child-submatrices
        posChildren = np.append(children[0,1:][None],children[children[:,0]=='1'][:,1:],axis=0)
        negChildren = np.append(children[0,1:][None],children[children[:,0]=='0'][:,1:],axis=0)
        # creating the 'positive' subdiagram
        if posChildren.shape[0] >1:
            # creating the name of the node, denoted by the path leading to it            
            nodeName = children[0,1]+'.'+parent
            # adding the node
            tree.add_node(nodeName)
            # adding the edge
            tree.add_edge(parent,nodeName,value=1)
            # doing the recursive call
            tree = appendNodesRecursively(tree,posChildren,nodeName)
        # creating the 'negative' subdiagram
        if negChildren.shape[0] >1:
            # creating the name of the node, denoted by the path leading to it
            nodeName = children[0,1]+'.-'+parent
            # adding the node
            tree.add_node(nodeName)
            # adding the edge
            tree.add_edge(parent,nodeName,value=0)
            # doing the recursive call
            tree = appendNodesRecursively(tree,negChildren,nodeName)
        return tree

def getLeaves(tree,node):
    """
    This function iterates recurively over a path and collects all subsequent
    leaves 
    """
    leaves = []
    leaves = getLeavesRec(node,leaves)
    return leaves

def getLeavesRec(tree,node,leaves):
    if tree.get_node(node)[isLeave]:
        leaves.append(node)
        return leaves
    for child in tree.get_nodes(node,child):
        leaves.append(getLeavesRec(tree,child,leaves))
    return leaves

def addDiagrams(dia1, dia2, nullValue):
    """
    this function adds two graphs. The underlying logic represents the
    matrix addition. 
    FOR THE TIME BEING, NO SIZE-CHECK IS PERFORMED! SO MAKE SURE THE DIAGRAMS
    ADDED COME FROM A DATASOURCE OF THE SAME SIZE!!!
    This function does a bottom-up addition
    """
    

def reduceDiagram(tree):
    return

def computeTree(matrix,nullValue):
    # getting the number of required vars
    noVars = getReqVars(matrix)
    print noVars
    matrix = expandMatrix2n(matrix,noVars[1:],nullValue)
    print matrix
    # constructing the tree
    bTree = nx.DiGraph()
    # adding the nodes, named x0,x1,...xn, where n is the number of required vars
#    varNames = getVarNames(noVars[0])
    # computing the nodes on a copy of the matrix
#    computeNodes(np.array(matrix,dtype=int),bTree,varNames)
    indices,leaves = getNonNullComponents(matrix,nullValue,noVars)
#    print leaves
#    conjs = getConjunctions(leaves)
#    print conjs
    varNames = getVarNames(noVars[0])
    sIndices,sVarNames,sOrder = sortVariables(indices,varNames)
    tree = computeDiagram(bTree,sIndices,leaves,sVarNames,noVars)
    
    nx.draw(tree)
    return tree,sVarNames[0]
#    bTree.add_node('Null')
#    bTree.add_node(varNames[i])
#    for i in range(varNames.shape[0]):
#        # negative branch of the parent
#        bTree.add_node('-'+varNames[i+1])
#        # positive branch of the parent
#        bTree.add_node(varNames[i+1])
#        for leaf in leaves:
#            if leaf[0][0] == '1':
                


a = np.random.random_integers(0,5,[3,3])
#print a
computeTree(a,0)