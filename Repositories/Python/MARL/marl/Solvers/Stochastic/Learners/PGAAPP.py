"""
The PGA-APP gradient update function.
"""
import numpy as np

from marl.Projection import BoxProjection, ErrorProjection
from marl.Estimators import q_learning_estimator
from marl.Solvers.Stochastic.Learners.GradientLearner import GradientLearner


class PGAAPP(GradientLearner):
    """
    The PGA-APP Learner (Policy Gradient Ascent with Approximate Policy Prediction).
    """
    def __init__(self,
                 domain,
                 value_function=q_learning_estimator,
                 simp_projection=BoxProjection(),
                 error_projection=ErrorProjection()):
        """
        :param domain:
        :param value_function:
        :param simp_projection:
        :param error_projection:
        """
        GradientLearner.__init__(self, domain, value_function, simp_projection, error_projection)
        # defining optional data entries
        self.optional_data = {'Iteration':              [0],
                              'Value':                  [[0. for _ in range(self.domain.dim)]],
                              'Min Reward':             [0.]}

    def policy_update(self, policy, action, reward, learning_setting):
        """
        The PGA-APP policy update.
        :param policy:
        :param action:
        :param reward:
        :param learning_setting:
        :return:
        """
        value_old = self.optional_data['Value'][-1]
        iteration = float(self.optional_data['Iteration'][-1] + 1)
        learning_rate = learning_setting[2]

        # update the value function
        reward_min = min(0., reward, self.optional_data['Min Reward'][-1])
        reward_c = reward - reward_min
        value_new = q_learning_estimator(self.domain.dim, action, reward_c, value_old, learning_rate=.01)

        # the derivative prediction length - gamma
        # derivative_prediction_length = 5000/(1000 + t)
        derivative_prediction_length = 3.
        # derivative_prediction_length =-1

        av = np.multiply(value_new, policy)
        unit = np.ones(len(value_new))

        if max(policy) < 1:
            delta = np.divide((value_new - av), (unit - policy))
        else:
            delta = (value_new - av)

        for i in range(len(delta)):
            delta[i] -= derivative_prediction_length*policy[i] * abs(delta[i])

        self.optional_data['Value'].append(value_new),
        self.optional_data['Iteration'].append(iteration),
        self.optional_data['Min Reward'].append(reward_min),
        updated_policy = self.simplex_projection(policy, learning_rate, delta)
        return updated_policy
