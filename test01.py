# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:27:17 2014

@author: clemens
"""

# Import graphviz
import sys
sys.path.append('..')
sys.path.append('/usr/lib/graphviz/python/')
sys.path.append('/usr/lib64/graphviz/python/')
import gv

import re

# Import pygraph
from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.searching import breadth_first_search
from pygraph.readwrite.dot import write

# This function parses DNF formulas, as "a+a.-b", where . is an 
# AND, and + an OR and - a NOT
def parseDependencies(function):
    # 1. parse ORs
    conjs   = re.split('\+',function)
    # Graph creation
    gr      = graph()
    deps    = {}
    # cycling through ands
    for conj in conjs:
        # parsing components
        variables   = re.split('\.',conj)
        # cycling through variables in conjunctions
        for var in variables:
            # removing nots
            nnVar       = re.sub('-','',var)
            # making sure the dictionary entry exists
            if not nnVar in deps and nnVar != '':
                deps[nnVar] = ''
            # cycling through the conjunction again to add dependencies
            for var1 in variables:
                nnVar1      = re.sub('-','',var1)
                # checking that one does not depend on itself and that 
                # the variable is not added repeatedly
                if (nnVar != nnVar1 and nnVar1 not in deps[nnVar]):
                    deps[nnVar] = deps[nnVar] + nnVar1
    return deps

# this function computes the dependency ordering
def sortDependencies(dependencies):
    # creating a list containing the variables
    variables = []
    # iterating through the list and projecting onto a number
    sortVars = sorted(dependencies.items(),key=len)
    print sortVars
    #for dep in dependencies:
    #    variables.append(len(dep)+dep)

a = parseDependencies('a.b+a.-b+a.c.f+g')
sortDependencies(a)
    