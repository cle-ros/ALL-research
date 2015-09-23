from __future__ import print_function
from marl.ToolsGeneral import dprint


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class Edge:
    def __init__(self, origin, target):
        self.origin = origin
        self.target = target

    def cost(self, agent, action):
        return self.origin.reward(agent=agent, actions=action)


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class Domain:
    """
    The base class for all domains.
    """

    def __init__(self):
        raise NotImplementedError('This is a generic domain object.  You need to pick a specific domain to use.')

    @staticmethod
    def action(policy):
        """
        Given a policy, this method will return an action following the policy.
        :param policy: the policy (i.e. probability distribution over the actions)
        :type policy: list
        :return: the action played by the agent
        :rtype: int
        """
        import scipy.stats
        dist = scipy.stats.rv_discrete(values=([i for i in range(len(policy))], policy))
        action = dist.rvs()
        return action

    def reward(self, actions_taken, player=None):
        """
        a wrapper for the different shapes the reward may have
        :return: the reward
        :rtype: numpy.array
        """
        if isinstance(self, RowColumnGame):
            a1 = actions_taken[0]
            a2 = actions_taken[1]
            if player is None:
                return [self.r_reward[a1][a2], self.c_reward[a1][a2]]
            else:
                if player == 0:
                    return self.r_reward[a1][a2]
                else:
                    return self.c_reward[a1][a2]
        else:
            import numpy as np
            reward_shape = [self.players]
            for i in range(self.players):
                reward_shape.append(self.dim)
            return np.zeros(reward_shape)
        # try:
        #     a1 = actions_taken[0]
        #     a2 = actions_taken[1]
        #     if player is None:
        #         return [self.r_reward[a1][a2], self.c_reward[a1][a2]]
        #     else:
        #         if player == 0:
        #             return self.r_reward[a1][a2]
        #         else:
        #             return self.c_reward[a1][a2]
        # except AttributeError:
        #     import numpy as np
        #     reward_shape = [self.players]
        #     for i in range(self.players):
        #         reward_shape.append(self.dim)
        #     return np.zeros(reward_shape)

    def reward_min_max(self):
        """
        This method computes the min and max values of the reward.
        :return:
        """
        raise NotImplementedError

    def u(self):
        """
        Computes the u-value
        :return:
        """
        pass

    def uprime(self):
        """
        Computes the u'-value
        :return:
        """
        pass

    def f(self, data):
        """
        ... deprecated ...
        :return:
        """
        pass

    def f_curl(self, data):
        """
        ... deprecated ...
        :return:
        """
        pass

    def ne_l2error(self, data):
        """
        Computes the error of the policies (i.e. the L_2 distance to the equilibrium strategy)
        :param data:
        :return:
        """
        pass

    def compute_value_function(self, policy):
        """
        Computes the analytical value of the policy given
        :param policy:
        :param policy_approx:
        :return:
        """
        pass

    def compute_nash_eq_value(self):
        """
        (analytically) computes the Nash-policy tuple.
        :return:
        """
        pass

    def reward_min_max(self):
        return self.reward()

    def unravel(self):
        return [self]

    @property
    def start_stage(self):
        return self

    @property
    def id(self):
        return 0

    @property
    def successors(self):
        try:
            return self.succ
        except AttributeError:
            self.succ = [Edge(self, self) for _ in range(self.dim)]
            return self.succ

    @property
    def is_final(self):
        try:
            self.stage_counter = self.stage_counter
        except AttributeError:
            self.stage_counter = -1
        self.stage_counter += 1
        return self.stage_counter >= 1

    def next_stage(self, policy=None, action=None, agent=None):
        if policy is not None:
            return self.action(policy=policy), self
        try:
            self.agent_actions[agent] = action
        except AttributeError:
            self.agent_actions = {agent: action}
        return action, self

    def update_all_costs(self):
        pass


class RowColumnGame(Domain):
    def __init__(self):
        pass

    def reward(self, actions=None, agent=None):
        """
        The funcion computing the reward for a specific player, given all the actions taken.
        :param args:
        :return:
        """
        return Domain.reward(self, actions, agent)
        # if agent == 0:
        #     return self.r_reward[actions[0]][actions[1]]
        # elif agent == 1:
        #     return self.c_reward[actions[0]][actions[1]]
        # else:
        #     return [self.r_reward, self.c_reward]
            # raise ValueError('Agent specified does not exist.')