"""
The WoLF GIGA update function (stochastic).
"""
import numpy as np

from marl.Projection import BoxProjection, ErrorProjection
from marl.Estimators import q_learning_estimator
from marl.Solvers.Stochastic.Learners.GradientLearner import GradientLearner


class WoLFGIGA(GradientLearner):
    """
    The WoLF-GIGA Learner (Win or Learn Fast - Generalized Infinitesimal Gradient Ascent).
    """
    def __init__(self, domain,
                 value_function=q_learning_estimator,
                 simp_projection=BoxProjection(),
                 error_projection=ErrorProjection()):
        GradientLearner.__init__(self, domain, value_function, simp_projection, error_projection)

        # defining optional data entries
        self.optional_data = {'z Policy':    [[0. for _ in range(self.domain.dim)]],
                              'Value':       [[0. for _ in range(self.domain.dim)]],
                              'Min Reward':  [0.]}

    def policy_update(self, policy, action, reward, learning_setting, optional_data=None):
        """
        This method computes an update for the forecaster, using the WoLF-GIGA update.
        :param policy:
        :param action:
        :param reward:
        :param learning_setting:
        :param optional_data:
        :return:
        """
        # checking on the reward-correction
        learning_rate = learning_setting[2]
        x_policy = policy
        z_policy = self.optional_data['z Policy'][-1]
        value_old = self.optional_data['Value'][-1]
        # computing the value function
        reward_min = min(0., reward, self.optional_data['Min Reward'][-1])
        reward_c = reward - reward_min
        value_new = q_learning_estimator(self.domain.dim, action, reward_c, value_old, learning_rate=.01)
        policy_gradient = value_new
        # perform update on the policies and project them into the feasible space
        updated_x_policy = self.simplex_projection(x_policy, learning_rate, policy_gradient)
        updated_z_policy = self.simplex_projection(z_policy, learning_rate/3., policy_gradient)

        # computing the second update learning rate
        dist_lentgh = np.linalg.norm(updated_z_policy - updated_x_policy)
        if dist_lentgh == 0.:
            dist_lentgh = 1.
        lr_difference = min(1., np.linalg.norm(updated_z_policy - z_policy) / dist_lentgh)

        # computing the final update for the x policy
        updated_policy = updated_x_policy + lr_difference * (updated_z_policy - updated_x_policy)
        self.optional_data['z Policy'].append(updated_z_policy)
        self.optional_data['Value'].append(value_new)
        self.optional_data['Min Reward'].append(reward_min)

        return updated_policy