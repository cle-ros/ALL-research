from __future__ import print_function
__author__ = 'clemens'

import numpy as np
import copy

from marl.Projection import *
from marl.Solvers.Solver import Solver
from marl.ToolsSpecific import *
from marl.Options import Reporting
from marl.Estimators import decaying_average_estimator, q_learning_estimator

import marl.config


class MATH(Solver):
    def __init__(self, domain,
                 p=IdentityProjection(),
                 learning_rate=0.06,
                 min_step=1e-5,
                 max_step=1e-3,
                 averaging_window=10,
                 exploration_trials=50,
                 averaging_meta_window=5,
                 estimator_decay=.8,
                 max_no_hypotheses=0,
                 lfea_strategy='linear-average',
                 hypo_eval_type='decaying-reward',
                 solvers=None,
                 q_learning_value_decay=.01):

        self.iteration = 0
        self.reward = domain.reward
        self.domain = domain
        self.nash_eq_value = domain.compute_nash_eq_value()
        self.Proj = p
        self.storage_size = 150
        self.temp_storage = {}
        self.min_step = min_step
        self.max_step = max_step
        self.averaging_window = averaging_window
        self.exploration_trials = exploration_trials
        self.amw = averaging_meta_window
        self.lr = learning_rate
        self.estimator_decay = estimator_decay
        self.value_approx_range = np.array([1./(51-1)*i for i in range(51)])
        self.additive_ = compute_increase_vector(3, .7, .3)
        # self.ne_hypotheses = np.zeros(self.value_approx_range.shape)
        wpl_solver_options = {'WPL Policy Gradient':    [0. for _ in range(self.domain.dim)],
                              'Value':                  0.}
        pga_app_solver_options = {'Iteration':              0,
                                  'Value':                  [0. for _ in range(self.domain.dim)],
                                  'Min Reward':  0.}
        wolfgiga_solver_options = {'z Policy':    [0. for _ in range(self.domain.dim)],
                                   'Value':       [0. for _ in range(self.domain.dim)],
                                   'Min Reward':  0.}
        # [[1., .75, .08], [1., .5, .13], [1., 1., .03], [1.5, 2.5, .01], [1.5, 3., .008]]
        if solvers is None:
            overview = np.array([
                                [self.singular_wpl_update, copy.deepcopy(wpl_solver_options), [1., 1., .01]],
                                # [self.singular_wpl_update, copy.deepcopy(wpl_solver_options), [1.5, 3., .008]],
                                # [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .3]],
                                # [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .05]],
                                # [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .15]],
                                # [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .1]],
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .01]],
                                # [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .001]],
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .05]],
                                ])
        elif solvers == 'pga-pga':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .05]],
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .005]],
                                ])
        elif solvers == 'wpl-pga-giga':
            overview = np.array([
                                [self.singular_wpl_update, copy.deepcopy(wpl_solver_options), [1., 1., .01]],
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .01]],
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .05]],
                                ])
        elif solvers == 'pga-giga':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .01]],
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .05]],
                                ])
        elif solvers == 'pga-giga-2':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .1]],
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .05]],
                                ])
        elif solvers == 'pga-pga-giga':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .02]],
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .005]],
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .05]],
                                ])
        elif solvers == 'pga-giga-giga':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .01]],
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .08]],
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .008]],
                                ])
        elif solvers == 'wpl-0.01':
            overview = np.array([
                                [self.singular_wpl_update, copy.deepcopy(wpl_solver_options), [1., 1., .01]],
                                ])
        elif solvers == 'pga-0.05':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .1]],
                                ])
        elif solvers == 'pga-0.01':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .01]],
                                ])
        elif solvers == 'pga-0.02':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .02]],
                                ])
        elif solvers == 'pga-0.005':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .005]],
                                ])
        elif solvers == 'pga-0.0001':
            overview = np.array([
                                [self.singular_pga_app_update, copy.deepcopy(pga_app_solver_options), [1., 1., .0001]],
                                ])
        elif solvers == 'giga-0.05':
            overview = np.array([
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .05]],
                                ])
        elif solvers == 'giga-0.08':
            overview = np.array([
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .08]],
                                ])
        elif solvers == 'giga-0.008':
            overview = np.array([
                                [self.singular_wolfgiga_update, copy.deepcopy(wolfgiga_solver_options), [1., 1., .008]],
                                ])
        else:
            overview = solvers
        self.solvers = list(overview[:, 0])
        self.init_solver_options = list(overview[:, 1])
        self.learning_settings = list(overview[:, 2])
        self.max_no_hypo = max_no_hypotheses
        self.no_forecasters = len(overview)
        self.lfea_strategy = lfea_strategy
        self.hypo_evaluation_type = hypo_eval_type
        self.init_lfea_options = {'Value': np.zeros(self.no_forecasters)}
        self.q_learning_value_decay = q_learning_value_decay
        # self.lfea_strategy = 'follow-the-best'
        # self.lfea_strategy = 'linear-averaging'
        # self.lfea_strategy = 'exponential-averaging'

    def init_temp_storage(self, start, domain, options):
        self.temp_storage['Learning Settings'] = [self.learning_settings] * self.domain.players
        self.temp_storage['Learning Settings'] = [self.temp_storage['Learning Settings'],
                                                  self.temp_storage['Learning Settings']]
        self.temp_storage['Action'] = np.zeros((self.storage_size,
                                                self.domain.players,
                                                )).tolist()
        self.temp_storage['Policy'] = self.storage_size * [start]
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros((self.domain.players,
                                                                                    self.domain.dim))]
        self.temp_storage['Reward'] = np.zeros((self.storage_size,
                                                self.domain.players,
                                                )).tolist()
        self.temp_storage['True Value Function'] = np.zeros((self.storage_size,
                                                             self.domain.players,
                                                             )).tolist()
        self.temp_storage['Forecaster Policies'] = np.zeros((self.storage_size,
                                                             self.domain.players,
                                                             self.get_forecaster_no(option='m'),
                                                             self.domain.dim,
                                                             ))
        self.temp_storage['Optional Settings'] = [copy.deepcopy(self.init_solver_options)
                                                  for _ in range(self.domain.players)]
        for i in range(self.domain.players):
            for j in range(len(self.init_solver_options)):
                if 'z Policy' in self.temp_storage['Optional Settings'][i][j]:
                    self.temp_storage['Optional Settings'][i][j]['z Policy'] = start[i]
        self.temp_storage['Optional Settings'] = [copy.deepcopy(self.temp_storage['Optional Settings'])
                                                  for _ in range(self.storage_size)]
        for i in range(self.domain.players):
            for j in range(self.get_forecaster_no(option='w')):
                self.temp_storage['Forecaster Policies'][:, i, j, :] = np.array(start[i])
        self.temp_storage['Forecaster Policies'] = self.temp_storage['Forecaster Policies'].tolist()
        self.temp_storage['Forecaster Settings'] = [[copy.deepcopy(self.init_lfea_options)
                                                     for _ in range(self.domain.players)]]
        return self.temp_storage

    def reporting_options(self):
        return Reporting(requests=[#self.domain.ne_l2error,
                                   'Policy',
                                   'Policy Gradient (dPi)',
                                   'Reward',
                                   'True Value Function',
                                   'Action',
                                   'Forecaster Policies',
                                   # 'Policy Estimates',
                                   ])

    def singular_wpl_update(self, policy, action, reward, learning_setting, optional_data=None):
        """
        This method computes an update for the forecaster, using the WPL update.
        :param policy:
        :param action:
        :param reward:
        :param learning_setting:
        :param optional_data:
        :return:
        """
        # compute the policy gradient:
        value = self.compute_value_function(reward, optional_data['Value'])
        policy_gradient = np.zeros(np.array(policy).shape)
        policy_error = reward - value
        # WPL projections
        policy_gradient[action] = self.project_error(policy_error, learning_setting[0], learning_setting[1])
        if policy_error < 0:
            policy_gradient[action] *= policy[action]
        else:
            policy_gradient[action] *= 1 - policy[action]
        # computing the policy gradient and the learning rate
        optional_data = {'WPL Policy Gradient':    policy_gradient,
                         'Value':                  value}
        return self.Proj.p(policy, learning_setting[2], policy_gradient), optional_data

    def singular_wolfgiga_update(self, policy, action, reward, learning_setting, optional_data=None):
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
        z_policy = optional_data['z Policy']
        value_old = optional_data['Value']
        # computing the value function
        reward_min = min(0., reward, optional_data['Min Reward'])
        reward_c = reward - reward_min
        value_new = q_learning_estimator(self.domain.dim, action, reward_c, value_old, learning_rate=.01)
        policy_gradient = value_new
        # perform update on the policies and project them into the feasible space
        updated_x_policy = self.Proj.p(x_policy, learning_rate, policy_gradient)
        updated_z_policy = self.Proj.p(z_policy, learning_rate/3., policy_gradient)

        # computing the second update learning rate
        dist_lentgh = np.linalg.norm(updated_z_policy - updated_x_policy)
        if dist_lentgh == 0.:
            dist_lentgh = 1.
        lr_difference = min(1., np.linalg.norm(updated_z_policy - z_policy) / dist_lentgh)

        # computing the final update for the x policy
        updated_policy = updated_x_policy + lr_difference * (updated_z_policy - updated_x_policy)
        optional_data = {'z Policy':    updated_z_policy,
                         'Value':       value_new,
                         'Min Reward':  reward_min}

        return updated_policy, optional_data

    def singular_pga_app_update(self, policy, action, reward, learning_setting, optional_data=None):
        """
        The PGA-APP policy update.
        :param policy:
        :param action:
        :param reward:
        :param learning_setting:
        :param optional_data:
        :return:
        """
        value_old = optional_data['Value']
        iteration = float(optional_data['Iteration'] + 1)
        learning_rate = learning_setting[2]

        # update the value function
        reward_min = min(0., reward, optional_data['Min Reward'])
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

        # dprint(delta, derivative_prediction_length, policy, delta)
        for i in range(len(delta)):
            delta[i] -= derivative_prediction_length*policy[i] * abs(delta[i])

        optional_data = {'Value':       value_new,
                         'Iteration':   iteration,
                         'Min Reward':  reward_min}
        # dprint(policy, learning_rate, delta, (5000./(5000.+iteration)))
        updated_policy = self.Proj.p(policy, learning_rate, delta)#*(5000./(5000. + iteration)))
        return updated_policy, optional_data

    def update(self, player, record):
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
        optional_settings = self.temp_storage['Optional Settings'][-1]
        lfea_settings = self.temp_storage['Forecaster Settings'][-1]

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # Initialize Storage
        temp_data = {}
        policy_gradient = np.zeros((self.domain.players, self.domain.dim))
        value = np.zeros((self.domain.players, self.no_forecasters)).tolist()
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
            for i in range(self.get_forecaster_no(option='w')):
                # compute the value of the current strategy
                updated_forecaster_policies[player][i], \
                    optional_settings[player][i] = self.solvers[i](tmp_forecaster_policies[-1][player][i],
                                                                   actions[player],
                                                                   reward[player],
                                                                   self.learning_settings[i],
                                                                   optional_settings[player][i])

            # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
            if marl.config.debug_output_level >= 1:
                print ('-> player', player)
                print ('   - which update is performed?', policy_gradient[player][actions[player]] < 0)
                print ('   - temp policies last round:  %.4f' % tmp_policy[-1][player][0])
                print ('   - the policy gradient:       ', policy_gradient[player])
                print ('   - the resulting policy:      ', policy_taken[player][0])

        # Store Data
        temp_data['Policy'] = policy_taken
        temp_data['Value Function'] = value
        temp_data['True Value Function'] = self.domain.compute_value_function(policy_taken)
        # temp_data['True Value Function'] = np.array(reward)
        # temp_data['True Value Function'] = np.sum(np.array(policy_taken), axis=0)
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Reward'] = reward
        temp_data['Action'] = actions
        temp_data['Forecaster Policies'] = updated_forecaster_policies
        temp_data['Learning Settings'] = tmp_learning_settings
        temp_data['Optional Settings'] = optional_settings
        temp_data['Forecaster Settings'] = lfea_settings
        self.book_keeping(temp_data)
        self.iteration += 1
        return self.temp_storage
