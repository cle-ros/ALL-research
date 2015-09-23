__author__ = 'clemens'

from marl.Projection import *
from marl.Solvers.Solver import Solver
from marl.ToolsSpecific import *
from marl.Options import Reporting
from marl.Estimators import q_learning_estimator
from marl.Estimators import decaying_average_estimator


class WoLFIGA(Solver):
    def __init__(self, domain, P=IdentityProjection(), delta0=1e-2, growth_limit=2, min_step=5e-5, max_step=2e-3):

        # self.reward = [domain.r_reward, domain.c_reward]
        self.domain = domain
        self.nash_eq_value = domain.compute_nash_eq_value()
        self.Proj = P
        self.storage_size = 3
        self.temp_storage = {}
        self.MinStep = min_step
        self.MaxStep = max_step

    def init_temp_storage(self, start, domain, options):

        self.temp_storage['Policy'] = self.storage_size * [start]
        self.temp_storage['Value Function'] = np.zeros((self.storage_size,
                                                        self.domain.players,
                                                        )).tolist()
        self.temp_storage['True Value Function'] = np.zeros((self.storage_size,
                                                             self.domain.players,
                                                             )).tolist()
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros(domain.b.shape)]
        self.temp_storage['Policy Learning Rate'] = self.storage_size * [[options.init.step], [options.init.step]]
        self.temp_storage['Projections'] = self.storage_size * np.zeros(self.domain.players).tolist()
        self.temp_storage['Reward'] = self.storage_size * np.zeros(self.domain.players).tolist()
        self.temp_storage['Action'] = self.storage_size * np.zeros(self.domain.players).tolist()

        return self.temp_storage

    def reporting_options(self):
        return Reporting(requests=[self.domain.ne_l2error,
                                   'Policy',
                                   'Policy Gradient (dPi)',
                                   'Policy Learning Rate',
                                   'Reward',
                                   'Value Function',
                                   'True Value Function',
                                   'Action',
                                   ])

    def update(self, record):
        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        value_old = np.array(record.temp_storage['Value Function'][-1])
        learn_min = self.MinStep
        learn_max = self.MaxStep

        # Initialize Storage
        temp_data = {}

        # playing the game
        actions = np.zeros(self.domain.players)
        for p in range(self.domain.players):
            actions[p] = self.domain.action(policy[p])
        rewards = self.domain.reward(actions)

        # estimate the gradient
        value_new = np.zeros(self.domain.players)
        for p in range(self.domain.players):
            # value_new[p] = q_learning_estimator(self.domain.dim, actions[p], reward[p], value_old[p], .1)
            try:
                all_actions = record.temp_storage['Action'][:][p] + [actions[p]]
            except TypeError:
                all_actions = [actions[p]]
            value_new[p] = decaying_average_estimator(all_actions, .9)
        policy_gradient = value_new

        # decide on the learning rate
        learning_rate = np.zeros(self.domain.players)
        for p in range(self.domain.players):
            learning_rate[p] = learn_min if np.sum(value_new[p]) > np.sum(value_old[p]) else learn_max

        # perform update on the policies and project them into the feasible space
        updated_policy = [self.Proj.p(policy[0], learning_rate[0], policy_gradient[0]),
                          self.Proj.p(policy[1], learning_rate[1], policy_gradient[1])]

        # Record Projections
        temp_data['Projections'] = 1 + self.temp_storage['Projections'][-1]

        # Store Data
        temp_data['Policy'] = updated_policy
        temp_data['Value Function'] = value_new
        temp_data['True Value Function'] = self.domain.compute_value_function(updated_policy)
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Policy Learning Rate'] = learning_rate
        temp_data['Reward'] = rewards
        temp_data['Action'] = actions
        self.book_keeping(temp_data)

        return self.temp_storage

