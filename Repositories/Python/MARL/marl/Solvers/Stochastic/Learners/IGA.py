__author__ = 'clemens'

from marl.Projection import *
from marl.Solvers.Solver import Solver
from marl.ToolsSpecific import *


class IGA(Solver):
    def __init__(self, domain, P=IdentityProjection(), delta0=1e-2, growth_limit=2, min_step=-1e10, max_step=1e10):

        self.reward = [domain.r_reward, domain.c_reward]
        self.domain = domain
        self.Proj = P
        self.StorageSize = 3
        self.temp_storage = {}

    def init_temp_storage(self, start, domain, options):

        self.temp_storage['Policy'] = self.StorageSize * [start]
        self.temp_storage['Value Function'] = self.StorageSize * [np.zeros(domain.r_reward.shape)]
        self.temp_storage['Policy Gradient (dPi)'] = self.StorageSize * [np.zeros(domain.b.shape)]
        self.temp_storage['Policy Learning Rate'] = self.StorageSize * [options.init.step]
        self.temp_storage['Projections'] = self.StorageSize * [0]
        # self.temp_storage['Value'] = self.storage_size * [0]

        return self.temp_storage

    def update(self, record):
        # settings
        play = False

        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        policy_grad = record.temp_storage['Policy Gradient (dPi)'][-1]
        pol_learning_rate = record.temp_storage['Policy Learning Rate'][-1]

        # Initialize Storage
        temp_data = {}

        # estimate the gradient
        policy_gradient = [ policy[1]*self.domain.u-     (self.reward[0][1][1]-self.reward[0][0][1]),
                            policy[0]*self.domain.uprime-(self.reward[1][1][1]-self.reward[1][1][0])]

        # perform update on the policies and project them into the feasible space
        updated_policy = [  self.Proj.p(policy[0], pol_learning_rate, policy_gradient[0]),  # Player 1
                            self.Proj.p(policy[1], pol_learning_rate, policy_gradient[1])]  # Player 2

        # compute the value of the current strategy
        value = self.domain.compute_value_function(updated_policy)

        # Record Projections
        temp_data['Projections'] = 1 + self.temp_storage['Projections'][-1]

        # Store Data
        temp_data['Policy'] = updated_policy
        temp_data['Value Function'] = value
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Policy Learning Rate'] = pol_learning_rate
        self.book_keeping(temp_data)

        return self.temp_storage