# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:50:44 2014

@author: clemens
"""
#look into the following graph libraries:
# NetworkX
#   http://networkx.github.io/
# iGraph
#   http://igraph.wikidot.com/

# Import graphviz
#import sys
#sys.path.append('..')
#sys.path.append('/usr/lib/graphviz/python/')
#sys.path.append('/usr/lib64/graphviz/python/')
#import gv

# Import pygraph
#from pygraph.classes.graph import graph
#from pygraph.classes.digraph import digraph
#from pygraph.algorithms.searching import breadth_first_search
#from pygraph.readwrite.dot import write

import networkx as nx

import matplotlib.pyplot as plt

try:
    from networkx import graphviz_layout
except ImportError:
    raise ImportError("This example needs Graphviz and either PyGraphviz or Pydot")

import numpy as np
import re

# in a later version, this function should be used to reduce the matrix
# dimensionality to 2
def flattenMatrix(matrix):
    if len(matrix.shape) > 2:
        print 'Please reduce matrix to be 2-dimensional. Aborting'
        return
    return matrix

def getReqVars(mat):
    # checking the matrix dimension 
    # 1. getting a binary representation of the shape
    vSize = re.sub("0b0*",'',bin(mat.shape[0]))
    hSize = re.sub("0b0*",'',bin(mat.shape[1]))
    # 2. computing the number of variables required for a matrix of this size
    return [len(vSize)+len(hSize),len(vSize),len(hSize)]

# this function expands a given matrix to be of size 2^n x 2^m, filling the 
# unknown values with the specified nullValue
def expandMatrix2n(matrix,demSize,nullValue):
    mat = np.ones([2**demSize[0],2**demSize[1]]) * nullValue
    mat[0:matrix.shape[0],0:matrix.shape[1]] = matrix
    return mat

# this function computes the names of the boolean variables    
def getVarNames(noVars):
    varNames = []
    for i in range(noVars):
        varNames.append('x'+str(i+1))
    return varNames

def computeNodes(matrix,bTree,varNames,value=True,parentName='',step=0):
#    print '~~ new loop ~~'
#    print step
    if matrix.shape[0] > 1:
#        print 'loop1'
        pName = parentName
        if not value:
            pName = pName+'-' 
        curName = pName+varNames[step]
        #print matrix
        bTree.add_node(curName)
        if not parentName is '':
            bTree.add_edge(curName,parentName)
#        print matrix.shape
#        print 'loop1'
#        print matrix
        # upper half, i.e. boolean false
        computeNodes(matrix[:matrix.shape[0]/2,:],bTree,varNames,False,curName,step+1)
        # lower half, i.e. boolean true
        computeNodes(matrix[matrix.shape[0]/2:,:],bTree,varNames,True,curName,step+1)
        return
    elif matrix.shape[1] > 1:
        pName = parentName
        if not value:
            pName = pName+'-'
        curName = pName+varNames[step]
        bTree.add_node(curName)
        if not parentName is '':
            bTree.add_edge(curName,parentName)
#        print matrix.shape
        if len(matrix.shape) < 2:
            matrix=matrix[None]
        # left half, i.e. boolean false
#        print 'loop2'
#        print matrix
        computeNodes(matrix[:,:matrix.shape[1]/2],bTree,varNames,False,curName,step+1)
        # right half, i.e. boolean true
        computeNodes(matrix[:,matrix.shape[1]/2:],bTree,varNames,True,curName,step+1)
        return
    elif np.prod(matrix.shape)==0:
        return
    else:
#        print 'loop else'
        leafName = str(matrix[0,0])
#        while leafName in bTree:
#            leafName = leafName + '`'
        bTree.add_edge(leafName,parentName)
        return
    return bTree

def computeTree(matrix,nullValue):
    # getting the number of required vars
    noVars = getReqVars(matrix)
    matrix = expandMatrix2n(matrix,noVars[1:],nullValue)
    # constructing the tree
    bTree = nx.Graph()
    # adding the nodes, named x0,x1,...xn, where n is the number of required vars
    varNames = getVarNames(noVars[0])
    # computing the nodes on a copy of the matrix
    computeNodes(np.array(matrix,dtype=int),bTree,varNames)
    # printing a picture
    nx.draw(bTree)
#    pos=nx.graphviz_layout(bTree,prog='twopi',args='')
#    plt.figure(figsize=(8,8))
#    nx.draw(bTree,pos,node_size=20,alpha=0.5,node_color="blue", with_labels=False)
#    plt.axis('equal')
#    plt.savefig('circular_tree.png')
#    plt.show()
#    dot = write(bTree)
#    gvv = gv.readstring(dot)
#    gv.layout(gvv,'dot')
#    gv.render(gvv,'png','europe.png')
    

def sortVars(tree)

# this function computes the DD from a given matrix. The 'nullValue' defines
# the fields which should be regarded as a boolean 0    
def computeDiagram(matrix,nullValue=0):
    mat = np.array(matrix)
    # checking for matrix size; if not 2-dimensional, project to 2-dimensions
    mat = flattenMatrix(mat)
    
a = np.random.random_integers(0,5,[3,3])
print a
computeTree(a,0)