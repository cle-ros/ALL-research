import numpy as np

from marl.Domains.Domain import Domain
from marl.ToolsGeneral import dprint


class ShapleysGame(Domain):
    """
    Shapley's  Game domain (http://en.wikipedia.org/wiki/Matching_pennies)
            R   0 1 1  C   1 0 1
                1 0 1      1 1 0
                1 1 0      0 1 1
    """
    def __init__(self):
        self.players = 2
        self.reward_range = [0., 1.]
        self.dim = 3
        # self.r_reward = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        # self.c_reward = np.array([[1., 0., 1.], [1., 1., 0.], [0., 1., 1.]])
        self.r_reward = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        self.c_reward = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        self.rewardf = np.array([self.r_reward.tolist(), self.c_reward.tolist()])
        self.u = self.u()
        self.uprime = self.uprime()
        self.A = np.array([[0., self.u], [self.uprime, 0.]])
        self.b = np.array(
            [-(self.r_reward[1, 1] - self.r_reward[0, 1]), -(self.c_reward[1, 1] - self.c_reward[1, 0])])
        self.A_curl = np.array(
            [[2. * self.uprime ** 2., 0], [0, 2. * self.u ** 2.]])
        self.b_curl = np.array([-2. * self.uprime * (self.c_reward[1, 1] - self.c_reward[1, 0]),
                                -2. * self.u * (self.r_reward[1, 1] - self.r_reward[0, 1])])
        self.NE = np.array([.33, .33, .33])  # 1 mixed NE

    def reward_min_max(self):
        """
        This method computes the min and max values of the reward.
        :return:
        """
        return [-1., 1.]

    def u(self):
        return (self.r_reward[0, 0] + self.r_reward[1, 1]) - (self.r_reward[1, 0] + self.r_reward[0, 1])

    def uprime(self):
        return (self.c_reward[0, 0] + self.c_reward[1, 1]) - (self.c_reward[1, 0] + self.c_reward[0, 1])

    def f(self, data):
        return self.A.dot(data) + self.b

    def f_curl(self, data):
        return 0.5 * self.A_curl.dot(data) + self.b_curl

    def ne_l2error(self, data):
        return np.linalg.norm(data[0] - self.NE)

    def compute_value_function(self, policy, policy_approx=None):
        value = np.zeros(self.players)
        # computing the first player's value function -> relying on estimates for the second player's strategy
        for pv in range(self.players):
            value[pv] = 0
            for d1 in range(self.dim):
                for d2 in range(self.dim):
                    value[pv] += self.rewardf[pv][d1][d2] * policy[1][d2] * policy[0][d1]
        return value

    def compute_nash_eq_value(self):
        policy = [self.NE, self.NE]
        return self.compute_value_function(policy)