__author__ = 'clemens'

import numpy as np
import random
from marl.Domains.Domain import Domain
from marl.ToolsGeneral import dprint


class TaskAllocation(Domain):
    """
    This domain is the task allocation problem as discussed by Abdallah and Lesser
    :param no_players: the number of players
    :param no_providers: the number of providers in the system
    :param factor_range_per_provider: the range of cost for each task (between 1 and x)
    :type no_players: int
    :type no_providers: int
    :type factor_range_per_provider: list
    :return:
    """
    def __init__(self, no_players, no_providers, factor_range_per_provider=None):
        if factor_range_per_provider is None:
            self.frpp = [1., 1.]
        else:
            self.frpp = factor_range_per_provider
        self.players = no_players
        self.dim = no_providers
        self.b = np.zeros(self.players)
        self.providers = [Provider(random.uniform(self.frpp[0], self.frpp[1])) for _ in range(no_providers)]
        self.stored_reward = None

    def compute_nash_eq_value(self):
        return 1/self.dim * np.ones(self.dim)

    def reward(self, actions_taken, player=None):
        """
        This method computes the rewards for the different agents, given the choice of actions made by those agents.
        :param actions_taken: a list/array of the actions taken, one for each agent
        :param player: if specified, only the value for the player will be returned
        :type actions_taken: numpy.array
        :type player: int
        :return: a numpy array containing the rewards for each agent, in order
        :rtype: numpy.array
        """
        at = np.array(actions_taken)
        cost = np.zeros(self.dim)
        reward = np.zeros(self.players)
        for i in range(self.dim):
            cost[i] = self.providers[i].cost(np.sum(at == i))
        for i in range(len(at)):
            reward[i] = cost[at[i]]
        # reward += self.players
        # dprint(actions_taken, reward)
        self.stored_reward = reward
        if player is not None:
            return reward[player]
        return reward

    def reward_min_max(self):
        """
        This method returns the min and max values of the reward.
        :return: [minvalue, maxvalue]
        """
        return -1. * self.players * self.frpp[1], 0.

    def compute_value_function(self, policies, comp_type='estimation', sampling_iters=100):
        """
        This method computes the analytical value of the policies played
        :param comp_type: specifies the computation; estimation or analytic
        :param policies: all the player's policies
        :param sampling_iters: the number of samples for the estimator
        :return: the analytical value for each player
        :comp_type policies: list
        :comp_type estimation: str
        :rtype: list
        """
        '''
        The analytical value function can be computed as follows:
            1. compute the value for each provider, by:
                a. computing the probability that it will be chosen 0 times, 1 times, 2 times ... depending on the
                   policies of the different players
                b. adding those probabilities, weighted by the cost following from the number of people playing
            2. computing the dot product between these costs and the players' policies.
        '''
        # dprint(policies.shape, self.stored_reward.shape)
        if comp_type == 'estimation':
            # another sampling approach
            estimation = np.zeros(self.players)
            for i in range(sampling_iters):
                actions = np.zeros(self.players)
                for i_player in range(self.players):
                    actions[i_player] = self.action(policies[i_player])
                rewards = self.reward(actions)
                # probs = np.array([policies[i][actions[i]] for i in range(self.players)])
                # estimation += np.multiply(probs, rewards)
                estimation += rewards
            # dprint(estimation / sampling_iters)
            return estimation / sampling_iters
        else:
            from marl.ToolsSpecific import combinations
            # 1. compute the different combinations
            combs = combinations(range(self.players), range(1, self.players+1, 1))
            # dprint(len(combs))
            # 2. for each possible action ...
            stage_values = []
            for act in range(self.dim):
                # 3. compute the resulting probabilities
                ind_values = []
                # iterating over the combinations of players to choose the specific action
                for comb in combs:
                    probs = []
                    # iterating over the players in the combination
                    for index in comb:
                        # picking the corresponding probability
                        probs.append(policies[index][act])
                    # computing the probability that all the other ones are not in the same stage
                    indices_no_dim = list(range(self.dim))
                    indices_no_dim.remove(act)
                    indices_no_players = list(range(self.players))
                    for c in comb:
                        indices_no_players.remove(c)
                    for player in indices_no_players:
                        probs.append(np.sum(policies[player][indices_no_dim]))
                    # computing the joint probability
                    jprob = np.product(probs)
                    # and the resulting value
                    ind_values.append(jprob * self.providers[act].cost(len(comb)))
                # appending to the values for each provider
                stage_values.append(np.sum(ind_values))
            # computing the resulting value of the policies of the different players
            values = []
            for player in range(self.players):
                values.append(np.dot(np.array(stage_values), np.array(policies[player])).tolist())
            return values


class Provider:
    """
    This class provides the provider objects, representing the agents solving the assigned tasks.
    :param base_factor: a factor representing the providers average performance
    :type base_factor: float
    """

    def __init__(self, base_factor=1.):
        self.base_factor = base_factor

    def cost(self, base_costs):
        """
        The function computing the cost of several tasks assigned to this provider. ATM, the cost is computed as
        a simple overall sum.
        :param base_costs: a list of the individual tasks' base cost
        :type base_costs: list
        :return: the cost of completing the tasks
        :rtype: float
        """
        try:
            return -1. * self.base_factor * sum(base_costs)
        except TypeError:
            return -1. * self.base_factor * base_costs
