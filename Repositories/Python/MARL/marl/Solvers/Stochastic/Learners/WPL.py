"""
The WPL update function.
"""
import numpy as np

from marl.Projection import BoxProjection, ErrorProjection
from marl.Estimators import q_learning_estimator
from marl.Solvers.Stochastic.Learners.GradientLearner import GradientLearner


class WPL(GradientLearner):
    """
    The Weighted Policy Learner.
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
        self.optional_data = {'WPL Policy Gradient':    [[0. for _ in range(domain.dim)]],
                              'Value Function':          [0.]}

    def policy_update(self, policy, action, reward, learning_setting):
        """
        This method computes an update for the forecaster, using the WPL update.
        :param policy:
        :param action:
        :param reward:
        :param learning_setting:
        :return:
        """
        value = self.compute_value_function(len(policy),
                                            action,
                                            reward,
                                            self.optional_data['Value Function'][-1],
                                            flatten=True)
        policy_gradient = np.zeros(np.array(policy).shape)
        policy_error = reward - value

        # WPL projections
        policy_gradient[action] = self.project_error(policy_error, learning_setting[0], learning_setting[1])
        if policy_error < 0:
            policy_gradient[action] *= policy[action]
        else:
            policy_gradient[action] *= 1 - policy[action]

        # computing the policy gradient and the learning rate
        self.optional_data['WPL Policy Gradient'].append(policy_gradient)
        self.optional_data['Value Function'].append(value)
        return self.simplex_projection(policy, learning_setting[2], policy_gradient)
