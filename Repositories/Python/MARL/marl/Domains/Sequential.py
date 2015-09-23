"""
This file defines the sequential wrapper, and its requirements (class Agent).
"""
from __future__ import print_function
from marl.ToolsGeneral import dprint
from marl.Storage import Storage
import numpy as np

__author__ = 'clemens'


class Agent:
    """
    The agent class; it is the 'active' part of this MARL implementation, combining the play and update methods.
    :param ident:
    :param sequential_instance: an instance of the seqential class
    :param solver_class:
    :param solver_instances:
    """

    def __init__(self, ident, sequential_instance, solver_class=None, solver_instances=None):
        self.id = ident
        self.seq_wrapper = sequential_instance
        self.stages = sequential_instance.stages
        self.start_stage = sequential_instance.domain.start_stage
        self.wrapper_domain = sequential_instance.domain
        self.solver_class = solver_class
        self.records = {}
        self.solvers = solver_instances
        self.paths_taken = []
        self.rewards = []
        self.init_records_solvers()

    def init_records_solvers(self):
        """
        This method initializes the record and solver objects to go to each stage.
        :rtype : None
        """
        # iterating over the stages
        solvers_given = True
        if self.solvers is None:
            self.solvers = {}
            solvers_given = False
        for stage in self.wrapper_domain.unravel():
            if not solvers_given:
                method = self.solver_class(stage, *self.seq_wrapper.method_base_options[1:])
                self.solvers[stage.id] = method
            else:
                method = self.solvers[stage.id]
            from marl.Options import (DescentOptions, Miscellaneous, Termination)
            # initializing the options for the solver
            terminal_conditions = Termination(max_iter=self.seq_wrapper.max_iterations)
            reporting_options = method.reporting_options()
            whatever_this_does = Miscellaneous()
            options = DescentOptions(None, terminal_conditions, reporting_options, whatever_this_does)
            # initializing the start conditions
            start = self.seq_wrapper.initialize_start_strategies(len(stage.successors))
            # creating the record object
            self.records[stage.id] = Storage(start, stage, method, options)

    def play_all(self, start_stage=None):
        """
        This method plays all stages, given the agent's current strategies, by:
            1. generating an action for all stages accessible in this run
            2. storing the resulting (stage, action, stage) pairs
        :param start_stage: the stage the chain of actions starts from
        """
        if start_stage is None:
            start_stage = self.start_stage
        stage = start_stage
        current_path = []
        while not stage.is_final:
            action, next_stage = stage.next_stage(policy=self.records[stage.id].perm_storage['Policy'][-1])
            current_path.append((stage, action, next_stage))
            stage = next_stage
        self.paths_taken.append(current_path)

    def collect_all_rewards(self):
        """
        This method collects all rewards stored in the domain object, assuming that the costs have been updated.
        """
        # iterating over the different stages
        dprint(self.paths_taken[-1])
        for i_origin, action, i_target in self.paths_taken[-1]:
            self.rewards.append(self.stages[i_origin.id].successors[action].cost(self.id, action))
            dprint(self.rewards, self.rewards[0])

    def get_policy(self, stage):
        """
        This method extracts the policy for the specified stage from the agent.
        :param stage: the stage object (i.e. current domain)
        """
        if not isinstance(stage, int):
            stage = stage.id
        return self.records[stage].perm_storage[-1]['Policy']

    def update_policy(self, stage_id, action, reward):
        """
        This method updates a specified policy according to the result of the previous play.
        :param reward: the reward received
        :param action: the action taken
        :param stage_id: the stage's (i.e. current domain) id
        """
        self.solvers[stage_id].update(self.records[stage_id], action, reward)

    def update_all_policies(self):
        """
        Updates all the policies corresponding to the path actually taken.
        """
        for i in range(len(self.paths_taken[-1])):
            i_origin, action, i_target = self.paths_taken[-1][i]
            self.update_policy(i_origin.id, action, sum(self.rewards[-1][i:]))
        dprint('policy updated')


class Sequential:
    """
    This class is meant as a wrapper for problems that have a sequential component.
    """
    def __init__(self,
                 domain,
                 method,
                 method_params,
                 initial_conditions,
                 max_iterations,
                 *domain_options,
                 **kw_domain_options):
        """
        Well, a constructor :-)
        :param game: The game played at each stage.
        :type domain: class Domain
        :type method: marl.Solvers.Solver.Solver
        :type method_params: tuple
        :type initial_conditions:
        :type max_iterations: int
        :return:
        """
        self.domain = domain(*domain_options, **kw_domain_options)
        # try:
        #     self.reward = self.domain.reward
        # except AttributeError:
        self.reward = self.domain.reward_min_max()
        self.max_iterations = max_iterations
        try:
            self.no_stages = self.domain.no_stages
        except AttributeError:
            self.no_stages = 1
        self.stages = self.domain.unravel()
        self.players = self.domain.players
        self.method = method
        self.method_base_options = method_params
        self.init_cond = initial_conditions
        self.stage_action_counts = np.zeros((self.no_stages, self.no_stages))
        self.agents = [Agent(i, self, solver_class=self.method) for i in range(self.domain.players)]

    def initialize_start_strategies(self, length):
        """
        This method computes a valid start distribution over all of the stages
        :param length: the number of outcomes of the distribution
        :return:.temp_storage['Policy'][-1]
        """
        start = np.ones((np.ceil(length), ))
        if self.init_cond == 'random':
            start = np.random.random((length, ))
        elif self.init_cond == 'uniform':
            # start = np.ones((length, ))
            pass
        start = start / start.sum()
        return start

    def update(self, record=None):
        """
        The method playing and updating the policies.
        :param record: The record object (unused for the sequential wrapper)
        :return:
        """
        '''
        2. each player selects the sequence of actions it takes until termination
        3. the rewards/costs for each stage are determined from the player's actions
        4. the costs are used to update the policies of each player at each stage
        5. reset the stages
        '''
        # 2. -~-~-~-~-~-~-~-~-~-~-
        for agent in self.agents:
            agent.play_all(start_stage=self.domain.start_stage)

        # 3. -~-~-~-~-~-~-~-~-~-~-
        self.domain.update_all_costs()
        for agent in self.agents:
            agent.collect_all_rewards()
            # 4. -~-~-~-~-~-~-~-~-~-~-
            agent.update_all_policies()

        # 5. -~-~-~-~-~-~-~-~-~-~-
        self.domain.reset_path_count()

    # def init_player_stage_alignment(self):
    #     """
    #     This function initializes the player-stage matrix, by picking a sequence of actions for each player,
    #     and computing the resulting sequence of stages.
    #     :return:
    #     """
    #     # initializing some variables
    #     no_agents = self.domain.players
    #     alignment_matrix = np.ones((no_agents, self.domain.max_path_length)) * -1
    #     for i_agent in range(no_agents):
    #         for i_dom in alignment_matrix[i_agent, :]:
    #             self.stages[i_agent][i_dom].update()
    #
    #
    #
    # def initialize_records_solvers(self):
    #     """
    #     This function initializes the record and solver-objects for each of the player-stage tuples.
    #     :return:
    #     """
    #     records = []
    #     solvers = []
    #     # iterating over the agents
    #     for agent in self.agents:
    #         # print('generating the start strategies for player ', agent)
    #         agent_stage_records = []
    #         agent_stage_solvers = []
    #         # iterating over the stages
    #         for i_stage in range(self.no_stages):
    #             if not self.stages[i_stage].is_final:
    #                 # initializing the solver object
    #                 # computing the three remaining options for the record object
    #                 start = self.initialize_start_strategies(len(self.stages[i_stage].successors))
    #                 domain = self.stages[i_stage]
    #                 print(self.method_base_options)
    #                 method = self.method(domain, *self.method_base_options[1:])
    #                 from marl.Options import (DescentOptions, Miscellaneous, Termination, Initialization)
    #                 terminal_conditions = Termination(max_iter=self.max_iterations)
    #                 reporting_options = method.reporting_options()
    #                 whatever_this_does = Miscellaneous()
    #                 options = DescentOptions(None, terminal_conditions, reporting_options, whatever_this_does)
    #                 # creating the record object
    #                 cur_record = Storage(start,
    #                                      domain,
    #                                      method,
    #                                      options)
    #                 agent_stage_records.append(cur_record)
    #                 agent_stage_solvers.append(method)
    #         agent.policies = agent_stage_records
    #         records.append(agent_stage_records)
    #         solvers.append(agent_stage_solvers)
    #
    #     return records, solvers
# def solve(start, method, domain, options):
#     # Record Data Dimension
#     domain.Dim = start.size  # is this necessary?
#
#     # Check Validity of options
#     options.check_options(method, domain)
#
#     # Create Storage Object for Record Keeping
#     record = Storage(start, domain, method, options)
#
#     # Begin Solving
#     while not options.term.is_terminal(record):
#         # Compute New Data Using update Method
#         temp_storage = method.update(record)  # should also report projections
#
#         # Record update Stats
#         record.book_keeping(temp_storage)
#
#     return record
# i.  update the policies
