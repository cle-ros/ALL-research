"""
The MATH (Multi Agent Testing of Hypotheses Paradigm) wrapper for Gradient learners.
"""
from __future__ import print_function
import copy

import numpy as np

from marl.Projection import *
from marl.Solvers.Solver import Solver
from marl.Options import Reporting
from marl.Solvers.Stochastic.Learners.PGAAPP import PGAAPP
from marl.Solvers.Stochastic.Learners.WoLFGIGA import WoLFGIGA
from marl.Solvers.Stochastic.Learners.WPL import WPL
import marl.config

__author__ = 'clemens'


class MATH(Solver):
    """
    The MATH (Multi Agent Testing of Hypotheses Paradigm) wrapper for Gradient learners.
    """
    def __init__(self, domain,
                 p=IdentityProjection(),
                 learning_rate=0.06,
                 averaging_window=10,
                 estimator_decay=.8,
                 max_no_hypotheses=0,
                 lfea_strategy='linear-average',
                 hypo_eval_type='decaying-reward',
                 solvers=None,
                 q_learning_value_decay=.01):
        Solver.__init__(self)
        self.is_stochastic = True
        self.iteration = 0
        self.domain = domain
        self.nash_eq_value = domain.compute_nash_eq_value()
        self.Proj = p
        self.storage_size = 150
        self.temp_storage = {}
        self.averaging_window = averaging_window
        self.lr = learning_rate
        self.estimator_decay = estimator_decay
        # initializing the solvers
        overview = parse_solvers(domain, solvers)
        self.solvers = list(overview[:, 0])
        self.learning_settings = list(overview[:, 1])
        self.max_no_hypo = max_no_hypotheses
        self.no_forecasters = len(overview)
        self.lfea_strategy = lfea_strategy
        self.hypo_evaluation_type = hypo_eval_type
        self.init_lfea_options = {'Value': np.zeros(self.no_forecasters)}
        self.q_learning_value_decay = q_learning_value_decay

    def init_temp_storage(self, start, domain, options):
        """
        This method initializes the storage keeping track of information needed over different plays.
        :param start:
        :param domain:
        :param options:
        :return:
        """
        self.temp_storage['Learning Settings'] = [[self.learning_settings] * self.domain.players,
                                                  [self.learning_settings] * self.domain.players]
        self.temp_storage['Action'] = np.zeros((self.storage_size,
                                                self.domain.players,
                                                )).tolist()
        self.temp_storage['Policy'] = self.storage_size * [start]
        self.temp_storage['Reward'] = np.zeros((self.storage_size,
                                                self.domain.players,
                                                )).tolist()
        self.temp_storage['True Value Function'] = np.zeros((self.storage_size,
                                                             self.domain.players,
                                                             )).tolist()
        self.temp_storage['Forecaster Policies'] = np.zeros((self.storage_size,
                                                             self.domain.players,
                                                             self.no_forecasters,
                                                             self.domain.dim,
                                                             ))
        for i in range(self.domain.players):
            for j in range(self.no_forecasters):
                self.temp_storage['Forecaster Policies'][:, i, j, :] = np.array(start[i])
        self.temp_storage['Forecaster Policies'] = self.temp_storage['Forecaster Policies'].tolist()
        self.temp_storage['Forecaster Settings'] = [[copy.deepcopy(self.init_lfea_options)
                                                     for _ in range(self.domain.players)]]
        return self.temp_storage

    @staticmethod
    def reporting_options():
        """
        Defines the reporting options for this particular solver.
        :return:
        """
        return Reporting(requests=[
                                   'Policy',
                                   'Reward',
                                   'True Value Function',
                                   'Action',
                                   'Forecaster Policies',
                                   ])

    def hypotheses_projection(self,
                              fc_policies,
                              reward_history,
                              action_history,
                              averaging_type=None,
                              lfea_settings=None):
        """
        This method does the projection from several hypotheses to one policy, following different strategies from
         the learning from expert advice paradigm.
        :param fc_policies:
        :param reward_history:
        :param action_history:
        :param averaging_type:
        :param lfea_settings:
        :return:
        """
        # another approach: follow-the-best
        # 1. compute the "value" for each of the policies
        value_type = self.hypo_evaluation_type
        normalized_reward = reward_history / max(reward_history.max(), 1.)
        normalized_reward -= np.mean(normalized_reward)
        # normalized_reward[normalized_reward<np.mean(normalized_reward)] *= 20
        pol_values = lfea_settings['Value']

        if value_type == 'decaying-reward':
            normalized_reward = np.array([normalized_reward[i]*(.98**(len(normalized_reward)-i-1))
                                          for i in range(len(normalized_reward))])
            pol_values = np.zeros(len(fc_policies[-1]))
            for i in range(len(fc_policies)):
                pol_values[action_history[i]] += np.sum(np.multiply(fc_policies[i, :, action_history[i]],
                                                                    normalized_reward[i]))
        elif value_type == 'q-learning':
            for i in range(len(pol_values)):
                new_val = normalized_reward[-1] * fc_policies[-1][i][action_history[-1]]
                pol_values[i] += self.q_learning_value_decay * (new_val - pol_values[i])

        if averaging_type == 'follow-the-best':
            best_estimate = fc_policies[-1][np.argmax(pol_values)] / self.no_forecasters

        elif averaging_type == 'exponential-averaging':
            # calculating the reward-action space
            weight = np.sqrt(8.*np.log(self.no_forecasters))
            best_estimate = np.zeros(self.domain.dim)
            adj_pol_values = np.array(pol_values)
            adj_pol_values = np.exp(weight * adj_pol_values)
            for i in range(len(adj_pol_values)):
                best_estimate += fc_policies[-1, i, :] * adj_pol_values[i]
            best_estimate /= np.sum(adj_pol_values)
        else:
            # This defaults to linear averaging, the best performing option.
            # elif averaging_type == 'linear-averaging':
            if max(np.abs(pol_values)) == 0.:
                pol_values = np.ones(pol_values.shape)
            best_estimate = np.zeros(self.domain.dim)
            # pol_values -= pol_values.min()
            for i in range(len(pol_values)):
                if pol_values[i] == 0:
                    dprint('upd')
                best_estimate += pol_values[i] * fc_policies[-1, i, :]
            best_estimate /= np.sum(pol_values)
        lfea_settings['Value'] = pol_values
        return self.Proj.p(best_estimate, 0., 0.), lfea_settings

    def update(self, record):
        """
        This method does the actual update.
        :param record:
        :return:
        """
        # Retrieve Necessary Data
        iteration = len(record.perm_storage['Policy'])
        policy = record.temp_storage['Policy'][-1]
        tmp_policy = self.temp_storage['Policy']
        tmp_reward = np.array(self.temp_storage['Reward'][-1*iteration:])
        tmp_action = np.array(self.temp_storage['Action'][-1*iteration:])
        tmp_forecaster_policies = np.array(self.temp_storage['Forecaster Policies'][-1*iteration:])
        tmp_learning_settings = self.temp_storage['Learning Settings'][-1]
        lfea_settings = self.temp_storage['Forecaster Settings'][-1]

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # Initialize Storage
        temp_data = {}
        actions = [0 for _ in range(self.domain.players)]
        policy_taken = np.array(policy)
        updated_forecaster_policies = np.array(tmp_forecaster_policies[-1])

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # 1. start by calculating the forecaster's policy/recommendation
        for player in range(self.domain.players):
            policy_taken[player],\
                lfea_settings[player] = self.hypotheses_projection(tmp_forecaster_policies[:, player],
                                                                   tmp_reward[:, player],
                                                                   tmp_action[:, player],
                                                                   averaging_type=self.lfea_strategy,
                                                                   lfea_settings=lfea_settings[player])
        # policy_taken = tmp_forecaster_policies[0, :, 0]
        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # 2. then play the game
        for player in range(self.domain.players):
            # playing the game
            actions[player] = self.domain.action(policy_taken[player])

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # 3. compute the reward and the resulting value function:
        reward = self.domain.reward(actions)

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # 4. updating the policies
        if marl.config.debug_output_level >= 1:
            print ('-~-~-~-~-~-~ new iteration (', iteration, ') ~-~-~-~-~-~-~-~-~-~-~-~-')
        # perform the update on the policies:
        for player in range(self.domain.players):
            # iterating over the different forecasters
            for i in range(self.no_forecasters):
                # compute the value of the current strategy
                updated_forecaster_policies[player][i] = self.solvers[i].policy_update(
                                                                         tmp_forecaster_policies[-1][player][i],
                                                                         actions[player],
                                                                         reward[player],
                                                                         self.learning_settings[i])

            # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
            if marl.config.debug_output_level >= 1:
                dprint(player, tmp_policy[-1][player][0], policy_taken[player])

        # Store Data
        temp_data['Policy'] = policy_taken
        temp_data['True Value Function'] = self.domain.compute_value_function(policy_taken)
        temp_data['Reward'] = reward
        temp_data['Action'] = actions
        temp_data['Forecaster Policies'] = updated_forecaster_policies
        temp_data['Learning Settings'] = tmp_learning_settings
        temp_data['Forecaster Settings'] = lfea_settings
        self.book_keeping(temp_data)
        self.iteration += 1
        return self.temp_storage


def parse_solvers(domain, solvers):
    """
    A function for parsing the solvers option, for better readability of the code.
    :param domain:
    :param solvers:
    :return:
    """
    if solvers is None:
        overview = np.array([
                            [WPL(domain), [1., 1., .01]],
                            # [WPL(domain), [1.5, 3., .008]],
                            # [PGAAPP(domain, [1., 1., .3]],
                            # [PGAAPP(domain, [1., 1., .05]],
                            # [PGAAPP(domain, [1., 1., .15]],
                            # [PGAAPP(domain, [1., 1., .1]],
                            [PGAAPP(domain), [1., 1., .01]],
                            # [WoLFGIGA(domain, [1., 1., .001]],
                            [WoLFGIGA(domain), [1., 1., .05]],
                            ])
    elif solvers == 'pga-pga':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .05]],
                            [PGAAPP(domain), [1., 1., .005]],
                            ])
    elif solvers == 'wpl-pga-giga':
        overview = np.array([
                            [WPL(domain), [1., 1., .01]],
                            [PGAAPP(domain), [1., 1., .01]],
                            [WoLFGIGA(domain), [1., 1., .05]],
                            ])
    elif solvers == 'pga-giga':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .01]],
                            [WoLFGIGA(domain), [1., 1., .05]],
                            ])
    elif solvers == 'pga-giga-2':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .1]],
                            [WoLFGIGA(domain), [1., 1., .05]],
                            ])
    elif solvers == 'pga-pga-giga':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .02]],
                            [PGAAPP(domain), [1., 1., .005]],
                            [WoLFGIGA(domain), [1., 1., .05]],
                            ])
    elif solvers == 'pga-giga-giga':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .01]],
                            [WoLFGIGA(domain), [1., 1., .08]],
                            [WoLFGIGA(domain), [1., 1., .008]],
                            ])
    elif solvers == 'wpl-0.01':
        overview = np.array([
                            [WPL(domain), [1., 1., .01]],
                            ])
    elif solvers == 'pga-0.05':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .1]],
                            ])
    elif solvers == 'pga-0.01':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .01]],
                            ])
    elif solvers == 'pga-0.02':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .02]],
                            ])
    elif solvers == 'pga-0.005':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .005]],
                            ])
    elif solvers == 'pga-0.0001':
        overview = np.array([
                            [PGAAPP(domain), [1., 1., .0001]],
                            ])
    elif solvers == 'giga-0.05':
        overview = np.array([
                            [WoLFGIGA(domain), [1., 1., .05]],
                            ])
    elif solvers == 'giga-0.08':
        overview = np.array([
                            [WoLFGIGA(domain), [1., 1., .08]],
                            ])
    elif solvers == 'giga-0.008':
        overview = np.array([
                            [WoLFGIGA(domain), [1., 1., .008]],
                            ])
    else:
        overview = solvers
    return overview
