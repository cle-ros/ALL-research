# -*- coding: utf-8 -*-
"""
@author: Clemens Rosenbaum (cgbr@cs.umass.edu)
"""


def d_apply(outer_node1, outer_node2, operation):
    """
    This is the canonical apply algorithm as introduced by Bryant in 1988
    """

    def apply_rec(node1, node2):
        return
    return


def evbdd(low, high):
    from node import BNode, BLeaf
    # the container collecting a list of already encountered paths/leaves
    encountered = {}
    if isinstance(low, int) and isinstance(high, int):
        # in case we are initializing at the leaves
        if encountered[high-low]:
            return [encountered[high-low][0], low-encountered[high-low][1]]
        else:
            leaf = BLeaf(low, low)
            node = BNode()
            node.p = leaf
            node.n = leaf
            node.edge_values[0] = 0
            node.edge_values[1] = high - low
            encountered[high-low] = [node, low]
            return [node, 0]
    # in case we are at a higher level, and are not initializing the leaves
    else:
        return


# function Apply(v1, v2: vertex; <op>: operator): vertex
# var T: array[1..|G 1 |, 1..|G 2 |] of vertex;
# {Recursive routine to implement Apply}
# function Apply-step(v1, v2: vertex): vertex;
# begin
# u := T[v1.id, v2.id];
# if u ≠ null then return(u); {have already evaluated}
# u := new vertex record; u.mark := false;
# T[v1.id, v2.id] := u; {add vertex to table}
# u.value := v1.value <op> v2.value;
# if u.value ≠ X
# then begin {create terminal vertex}
# u.index := n+1; u.low := null; u.high := null;
# end
# else begin {create nonterminal and evaluate further down}
# u.index := Min(v1.index, v2.index);
# if v1.index = u.index
# then begin vlow1 := v1.low; vhigh1 := v1.high end
# else begin vlow1 := v1; vhigh1 := v1 end;
# if v2.index = u.index
# then begin vlow2 := v2.low; vhigh2 := v2.high end
# else begin vlow2 := v2; vhigh2 := v2 end;
# u.low := Apply-step(vlow1, vlow2);
# u.high := Apply-step(vhigh1, vhigh2);
# end;
# return(u);
# end;
# begin {Main routine}
# Initialize all elements of T to null;
# u := Apply-step(v1, v2);
# return(Reduce(u));
# end;