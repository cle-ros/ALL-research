__author__ = 'clemens'

import numpy as np
from marl.ToolsGeneral import dprint
from marl.Domains.Domain import Domain, Edge


# -~-~-~-~-~-~-~-~ cost functions -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
def linear_cost_function(traffic_flow, base_cost, capacity=1e6, add_cost=.001):
    if traffic_flow < capacity:
        return (base_cost + (traffic_flow * add_cost)) *-1
    else:
        return 1e6


def realistic_cost_function(traffic_flow, base_cost, capacity, mult_offset=.01, exp_offset=1.5):
    """
    This computes the cost of travelling on the road following an empirical model taken from the Bachelor's thesis
    'Variational Inequalities and Complementarity Problems' by Friedemann Sebastian Winkler.
    """
    return -1 * base_cost * (1 + mult_offset * (traffic_flow/capacity)**exp_offset)


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
class NetworkEdge(Edge):
    def __init__(self, origin, target, base_cost, capacity, traffic_flow):
        Edge.__init__(origin, target)
        self.base_cost = base_cost
        self.capacity = capacity
        self.traffic_flow = traffic_flow

    def cost(self, agent, action):
        return self.origin.network.cost_function(self.traffic_flow, self.base_cost, self.capacity)


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
class NetworkNode:
    """
    One element of the network; resembling a cross section of the traffic map
    """
    def __init__(self, ident, network):
        self.id = ident
        self.network = network
        self.layer_from_start = -1
        # the successors should have the format
        #   id: [(node) successor, base_cost, capacity, traffic_flow, cost, (opt) offs1, (opt) offs2]
        self.is_final = False
        self.successors = []

    @property
    def reward(self):
        return np.zeros((len(self.successors)))

    def ne_l2error(self):
        pass

    @property
    def dim(self):
        return len(self.successors)

    def next_stage(self, policy=None, action=None):
        """
        This method returns the next stage, either by playing a specified policy or by taking a specified action.
        """
        if policy is None and action is None:
            raise ValueError('Please specify a policy or an action.')

        if policy is not None:
            action = self.play(policy)
        return action, self.successors[action].target

    def play(self, policy):
        """
        Play the game (i.e. select an action) according to the specified policy.
        """
        import scipy.stats
        # print('in the play functino', self.id, self.is_final, policy)
        dist = scipy.stats.rv_discrete(values=([i for i in range(len(policy))], policy))
        action = dist.rvs()
        self.successors[action].traffic_flow += 1
        return action

    def action(self, policy):
        return self.play(policy)

    def __hash__(self):
        return self.id


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
class Network(Domain):
    """
    the class storing the information on the traffic network problem
    """
    def __init__(self, no_nodes, no_agents, max_path_length, max_edge_length, sparsity=.5, max_capacity=1e3, b_dead_end=True,
                 b_one_way=True, cf=linear_cost_function, dual_conn=2):
        """
        This function initializes the network for the traffic problem
        :param max_path_length:
        :param no_nodes:
        :param max_edge_length:
        :param sparsity: = ignored =
        :param max_capacity:
        :param b_dead_end:
        :param b_one_way:
        :return:
        """
        self.no_stages = no_nodes
        self.max_path_length = max_path_length
        self.max_edge_length = max_edge_length
        self.cost_function = cf
        self.adjacency_matrix = 1e6*np.ones((self.no_stages, self.no_stages, 4))
        self.players = no_agents
        self.sparsity = sparsity
        self.max_capacity = max_capacity
        self.has_dead_ends = b_dead_end
        self.has_dual_conn = dual_conn
        self.is_reversible = not b_one_way
        # initialize the set of nodes
        self.stages = [NetworkNode(i, self) for i in range(self.no_stages)]
        self.stages[-1].is_final = True
        self.start_stage = self.stages[0]
        # initializing the layer of the first node
        self.stages[0].layer_from_start = 0
        self.nodex_by_id = {}
        for i in range(self.no_stages):
            self.nodex_by_id[i] = self.stages[i]
        self.create_connections()
        # each edge stores length, capacity, usage, cost

    @property
    def reward(self):
        return self.adjacency_matrix

    def resolve_target_to_action(self, origin, target):
        if isinstance(target, int):
            ident = target
        else:
            ident = target.id

        for i in range(len(origin.successors)):
            if origin.successors[i].target.id == ident:
                return i

    def reward_min_max(self):
        return 0, self.max_edge_length*self.max_path_length

    def compute_nash_eq_value(self):
        pass

    def ne_l2error(self):
        pass

    def compute_path_cost(self, origin, action=None, successor=None):
        """
        This method computes the cost of taking one of the paths.
        :type origin: NetworkNode
        :type successor: NetworkNode
        """
        # making sure the origin is of the right type
        orig = origin
        if type(origin) is int:
            orig = self.stages[origin]

        if action is None != successor is None:
            raise AttributeError

        if action is None:
            for i in range(len(origin.successors)):
                if origin.successors[i].target.id == successor.id:
                    action = i
                    break
        edge = origin.successors[action]
        # traffic_flow, base_cost, capacity=1e6, add_cost=.001
        # id: [(node) successor, base_cost, capacity, traffic_flow, cost, (opt) offs1, (opt) offs2]
        dprint('compute path cost: ', origin, action, orig.successors.keys())
        return self.cost_function(edge.traffic_flow, edge.base_cost)

    def unravel(self):
        """
        This function computes a list of all stages, from the connected list.
        :return:
        :rtype: list
        """
        return self.stages

    def init_single_connection(self, origin, target, length=None, capacity=None, usage=0):
        """
        This method creates a single connection
        :param origin:
        :param target:
        :param length:
        :param capacity:
        :return:
        """
        s = self.stages
        if length is None:
            length = self.max_edge_length*np.random.random()
        if capacity is None:
            capacity = np.rint(self.max_capacity*np.random.random()).astype(int)
        else:
            capacity = np.rint(capacity).astype(int)
        cost = self.cost_function(length, capacity, usage)
        s[origin].successors.append(Edge(s[origin], s[target], length, capacity, usage))
        self.adjacency_matrix[origin, target, :] = [length, capacity, usage, cost]
        if self.stages[target].layer_from_start == -1:
            self.stages[target].layer_from_start = self.stages[origin].layer_from_start + 1

    def exec_chosen_action(self, origin, action):
        """
        A wrapper to generalize the different domains to be sequential.
        :param origin:
        :param agent:
        :param action:
        :return:
        """
        return self.update_single_connection(origin, self.stages[origin].successors[action].target.id, diff=1)

    def reset_path_count(self):
        """
        This method will reset all the path counts, after each play.
        :return:
        """
        for origin in self.stages:
            for s_action in range(len(self.stages[origin.id].successors)):
                edge = self.stages[origin.id].successors[s_action]
                self.update_single_connection(origin, edge.target, new_total=0, action=s_action)

    def update_single_connection(self, origin, target, diff=0, new_total=None, update_all=False, action=None):
        """
        adf
        :param diff:
        :param new_total:
        :return:
        """
        if action is None:
            action = self.resolve_target_to_action(origin, target.id)
        if new_total is None:
            self.adjacency_matrix[origin, action, 2] += diff
            origin.successors[action].traffic_flow += diff
        else:
            origin.successors[action].traffic_flow = new_total
        self.adjacency_matrix[origin.id, target.id, 3] = self.stages[origin.id].successors[action].cost

        if update_all:
            length, capacity, usage = self.adjacency_matrix[origin, action, :-1]
            new_cost = self.cost_function(length, capacity, usage)
            self.stages[origin].successors[action].cost = new_cost
            self.adjacency_matrix[origin, action, 3] = new_cost

    def update_all_costs(self):
        """
        This function updates the costs for all the edges.
        :return:
        """
        for stage in self.stages:
            for edge in stage.successors:
                self.adjacency_matrix[stage.id, edge.target.id, 3] = edge.cost

    def create_connections(self):
        """
        The method constructing the paths (i.e. "streets")
        """
        # initializing with one direct path from start to finish, to guarantee one path (however, this path has minimal
        # capacity). The path may be overridden later on.
        # self.init_single_connection(0, self.no_nodes-1, self.no_nodes*self.max_edge_length, 1e-2)

        # for the special case of two nodes only.
        if self.no_stages == 2:
            start = self.stages[0]
            goal = self.stages[1]
            for i in range(self.has_dual_conn):
                base_cost = np.random.random() * self.max_edge_length
                capacity = np.random.random() * self.max_capacity
                start.successors.append(Edge(start, goal, base_cost, capacity, 0))
        else:
            if not self.has_dead_ends:
                for i_node in range(1, self.no_stages, 1):
                    self.init_single_connection(i_node, self.no_stages-1, self.max_edge_length*self.max_path_length)
            for i_node in range(self.no_stages):
                if self.is_reversible:
                    target_nodes = range(self.no_stages)
                else:
                    target_nodes = range(i_node + 1, self.no_stages, 1)
                for in_node_2 in target_nodes:
                    if i_node != in_node_2 and self.stages[i_node].layer_from_start < self.max_path_length:
                                    # np.random.random() > self.sparsity and \
                        self.init_single_connection(i_node, in_node_2)

    def plot(self, name):
        """
        This function plots the network using GraphViz and DOT (https://en.wikipedia.org/wiki/DOT_language)
        """
        import pydot as pd
        graph = pd.Dot('diagram', graph_type='digraph')
        graph.add_node(pd.Node('0', label='Start', shape='box'))
        graph.add_node(pd.Node(str(self.stages[-1].id), label='Goal', shape='box'))
        for node in self.stages[1: -1]:
            graph.add_node(pd.Node(str(node.id), label=str(node.id)))

        for node in self.stages:
            for edge_id in range(len(node.successors)):
                no = str(np.round(node.successors[edge_id].target.id, 2)) + ';\n ' + \
                     str(node.successors[edge_id].base_cost) + '/'  + \
                     str(np.round(node.successors[edge_id].capacity, 2)) + ';\n '
                if node.successors[edge_id].cost < 1:
                    no += '-'
                else:
                    no += str(node.successors[edge_id].cost)
                graph.add_edge(pd.Edge(str(node.id), str(node.successors[edge_id].target.id), label=no))
        file_name = './plots/' + name + '.png'
        graph.write_png(file_name)


# class TrafficNetwork(Domain):
#     """
#     The matching pennies domain (http://en.wikipedia.org/wiki/Matching_pennies)
#     """
#     def __init__(self):
#         self.players = 2
#         self.reward_range = [-1, 1]
#         self.dim = 2
#         self.r_reward = np.array([[1., -1.], [-1., -1]])
#         self.c_reward = np.array([[1., -1.], [-1., -1.]])
#         self.u = self.u()
#         self.uprime = self.uprime()
#         self.A = np.array([[0., self.u], [self.uprime, 0.]])
#         self.b = np.array(
#             [-(self.r_reward[1, 1] - self.r_reward[0, 1]), -(self.c_reward[1, 1] - self.c_reward[1, 0])])
#         self.A_curl = np.array(
#             [[2. * self.uprime ** 2., 0], [0, 2. * self.u ** 2.]])
#         self.b_curl = np.array([-2. * self.uprime * (self.c_reward[1, 1] - self.c_reward[1, 0]),
#                                 -2. * self.u * (self.r_reward[1, 1] - self.r_reward[0, 1])])
#         self.NE = np.array([[1., .0],
#                             [1., .0]])  # 1 mixed NE
#
#     def u(self):
#         return (self.r_reward[0, 0] + self.r_reward[1, 1]) - (self.r_reward[1, 0] + self.r_reward[0, 1])
#
#     def uprime(self):
#         return (self.c_reward[0, 0] + self.c_reward[1, 1]) - (self.c_reward[1, 0] + self.c_reward[0, 1])
#
#     def f(self, data):
#         return self.A.dot(data) + self.b
#
#     def f_curl(self, data):
#         return 0.5 * self.A_curl.dot(data) + self.b_curl
#
#     def ne_l2error(self, data):
#         return np.linalg.norm(data - self.NE)
#
#     def compute_value_function(self, policy, policy_approx=None):
#         if policy_approx is None:
#             policy_approx = policy
#         value = [0., 0.]
#         # computing the first player's value function -> relying on estimates for the second player's strategy
#         value[0]= self.r_reward[0][0]*policy[0][0]*policy_approx[1][0]\
#                 + self.r_reward[1][1]*policy[0][1]*policy_approx[1][1]\
#                 + self.r_reward[0][1]*policy[0][0]*policy_approx[1][1]\
#                 + self.r_reward[1][0]*policy[0][1]*policy_approx[1][0]
#
#         # computing the second player's value function -> relying on estimates for the first player's strategy
#         value[1]= self.c_reward[0][0]*policy_approx[0][0]*policy[1][0]\
#                 + self.c_reward[1][1]*policy_approx[0][1]*policy[1][1]\
#                 + self.c_reward[0][1]*policy_approx[0][0]*policy[1][1]\
#                 + self.c_reward[1][0]*policy_approx[0][1]*policy[1][0]
#         return value
#
#     def compute_nash_eq_value(self):
#         policy = [[self.NE[0], 1-self.NE[0]],
#                   [self.NE[1], 1-self.NE[1]]]
#         return self.compute_value_function(policy)
#
#     @staticmethod
#     def action(policy):
#         ind = np.random.rand()
#         try:
#             if ind <= policy[0]:
#                 return 0
#             else:
#                 return 1
#         except ValueError:
#             if ind <= policy[0][0]:
#                 return 0
#             else:
#                 return 1
