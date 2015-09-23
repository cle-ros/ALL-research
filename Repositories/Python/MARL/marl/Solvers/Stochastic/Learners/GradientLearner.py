"""
The Gradient Learner Base Class.
"""
__author__ = 'clemens'


class GradientLearner:
    """
    The super class for all gradient learners.
    """
    def __init__(self,
                 domain,
                 value_function,
                 simp_projection,
                 error_projection):
        self.domain = domain
        self.compute_value_function = value_function
        self.simplex_projection = simp_projection.p
        self.project_error = error_projection.p
        # defining optional data entries
        self.optional_data = None

    def policy_update(self, policy, action, reward, learning_setting):
        """
        The policy update function.
        :param policy:
        :param action:
        :param reward:
        :param learning_setting:
        :return:
        """
        raise NotImplementedError
