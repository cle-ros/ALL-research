import time
from optparse import OptionParser

# sys.path.append(dirname('~/remote-working-dirs/home/pydd'))

# from Domains.DummyMARL import *
# from Domains.DummyMARL2 import *

from marl.Solvers.Solver import solve
from marl.Domains.Stochastic.MatchingPennies import MatchingPennies
from marl.Domains.Stochastic.Tricky import Tricky
from marl.Domains.Stochastic.PureCollaborative import PureCollaborative
from marl.Domains.Stochastic.BachStravinski import BachStravinski

from marl.Solvers.Stochastic.MATH import MATH
from marl.Projection import *
from marl.Options import (
    DescentOptions, Miscellaneous, Termination, Initialization)
from marl.Log import print_sim_results
from marl.ToolsGeneral import plot_results
from marl.ToolsSpecific import *
import marl.config as config


def demo(domain, method, iterations=500, start_strategies=None):
    # __DUMMY_MARL__##################################################

    # Set Method
    # box = np.array(Domain.reward_range)
    box = np.array(domain.reward_min_max())
    epsilon = np.array([-0.01, 0.01])

    # Initialize Starting Point
    # Start = np.array([0,1])
    if start_strategies is None or start_strategies == 'random':
        start_strategies = np.random.random((domain.players, domain.dim))
        for i in range(start_strategies.shape[0]):
            start_strategies[i] /= np.sum(start_strategies[i])
    elif start_strategies == 'deterministic-s':
        ind_start_strat = np.ones(domain.dim)*.01
        ind_start_strat[0] = 1. - (domain.dim-1)*.01
        start_strategies = np.array([ind_start_strat for _ in range(domain.players)])
    elif start_strategies == 'deterministic-s-i':
        ind_start_strat = np.ones(domain.dim)*.01
        ind_start_strat[0] = 1. - (domain.dim-1)*.01
        start_strategies = 1.-np.array([ind_start_strat for _ in range(domain.players)])
    elif start_strategies == 'deterministic-a':
        start_strategies = np.zeros((domain.players, domain.dim))
        for i in range(domain.players):
            cur_start_strat = np.ones(domain.dim)*.01
            cur_start_strat[i % domain.dim] = 1. - (domain.dim-1)*.01
            cur_start_strat = method.Proj.p(cur_start_strat, 0., 0.)
            start_strategies[i] = cur_start_strat
    elif start_strategies == 'deterministic-a-i':
        start_strategies = np.zeros((domain.players, domain.dim))
        for i in range(domain.players):
            cur_start_strat = np.ones(domain.dim)*.01
            cur_start_strat[i % domain.dim] = 1. - (domain.dim-1)*.01
            cur_start_strat = method.Proj.p(cur_start_strat, 0., 0.)
            start_strategies[i] = cur_start_strat
        start_strategies = 1.- start_strategies
    elif start_strategies == 'lopsided':
        start_strategies = np.zeros((domain.players, domain.dim))
        for i in range(domain.players):
            cur_start_strat = np.ones(domain.dim)*.12
            cur_start_strat[i % domain.dim] = .5
            cur_start_strat = method.Proj.p(cur_start_strat, 0., 0.)
            start_strategies[i] = cur_start_strat
    elif start_strategies == 'lopsided-i':
        start_strategies = np.zeros((domain.players, domain.dim))
        for i in range(domain.players):
            cur_start_strat = np.ones(domain.dim)*.12
            cur_start_strat[i % domain.dim] = .5
            cur_start_strat = method.Proj.p(cur_start_strat, 0., 0.)
            start_strategies[i] = cur_start_strat
        start_strategies = 1 - start_strategies
    # dprint(start_strategies)
    # Set Options
    initialization_conditions = Initialization(step=1e-4)
    # terminal_conditions = Termination(max_iter=iterations, tols=[(domain.ne_l2error, 1e-20)])
    terminal_conditions = Termination(max_iter=iterations)
    reporting_options = method.reporting_options()
    whatever_this_does = Miscellaneous()
    options = DescentOptions(initialization_conditions, terminal_conditions, reporting_options, whatever_this_does)

    # Start Solver
    tic = time.time()
    # set random seed
    # np.random.seed(0)
    # dprint(start_strategies.shape)
    marl_results = solve(start_strategies, method, domain, options)
    toc = time.time() - tic

    # Print Results
    if config.debug_output_level != -1:
        print_sim_results(options, marl_results, method, toc)

    # computing the statistics
    # reward = np.array(marl_results.perm_storage['Reward'])
    # win_ratio = np.mean(.5 + .5*reward, axis=0).tolist()#.round(2)
    # print_exception = False
    # if win_ratio[1] < .4 and type(method) is LEAP:
    #     print 'Problem:', win_ratio[1]
        # print_exception = True

    # creating plots, if desired:
    if config.show_plots:# or print_exception:
        policies = np.array(marl_results.perm_storage['Policy'])  # Just take probabilities for first action

        if 'Forecaster Policies' in marl_results.perm_storage:
            forecaster = np.array(marl_results.perm_storage['Forecaster Policies'])[:, :, :, 0]
            # dprint(policies[:, 0, 0].shape, forecaster[:,0,0].shape, forecaster[:,1,0].shape)
            ne = np.ones(policies[:, 0, 0].shape)#*.33
            thingy = np.vstack((policies[:, 0, 0], forecaster[:, 0, 0], forecaster[:, 0, 1], ne))
            # thingy = np.vstack((policies[:, 0, 0], ne))#, forecaster[:, 0, 0], forecaster[:, 0, 1], ne))

        # true_val_fun = np.array(marl_results.perm_storage['True Value Function'][1:])

        printing_data = {}
        printing_data[''] = {
            'axesLabels':   ['Iterations/Number of plays', 'Probability of first action'],
            'values':       thingy,
            'yLimits':      [0., 1.] + epsilon,
            'smooth':       -1,
            # 'labels':       ['Policy', 'Nash Eq.'],
            # 'colors':       [None, 'black'],
            # # 'markers':  ['o', '^', 'v'],
            # 'linestyles':   ['-', ':'],
            # 'linewidths':   [3, 3]
            # 'labels':       ['Projection', 'Hypothesis 1', 'Hypothesis 2', 'Nash Eq.'],
            'labels':       ['Projection', 'Hypothesis 1 (PGA)', 'Hypothesis 2 (GIGA)', 'Nash Eq.'],
            'colors':       [None, None, None, 'black'],
            # 'markers':  ['o', '^', 'v'],
            'linestyles':   ['-', '-.', '--', ':'],
            'linewidths':   [3, 3, 3, 3]
        }
        # printing_data['Expected Reward of policies played'] = {'values': true_val_fun,
        #                                                        'yLimits': np.array(domain.reward_min_max()) + epsilon,
        #                                                        'smooth': -1}
        # for p in range(policies.shape[1]):
        #     printing_data['The policies: '+str(p)] = {'values': policies[:,p,:], 'yLimits': np.array([0., 1.]) + epsilon, 'smooth': -1}
        #     if 'Forecaster Policies' in marl_results.perm_storage:
        #         printing_data['Hypotheses - P'+str(p)] = {'values': forecaster[:, p, :], 'yLimits': np.array([0, 1]) + epsilon, 'smooth': -1}
        plot_results(printing_data, legend_location=1)

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
    else:
        policies = np.array(marl_results[0].perm_storage['Policy'])[1:, 0, 0]
        forecaster = np.array(marl_results[0].perm_storage['Forecaster Policies'])[1:, 0, :, 0]
        true_val_fun = np.array(marl_results[0].perm_storage['True Value Function'][1:])
        return policies, forecaster, true_val_fun
    return


def wrapper_batch_testing_helper(domains, trials, iterations, bt_type):
    for dom in domains:
        domain = domains[dom]
        config.batch_testing = bt_type
        results = np.zeros((len(methods), trials, iterations, 2))
        for trial in range(trials):
            start_st = np.random.random((domain.players, domain.dim))
            for k in range(start_st.shape[0]):
                start_st[k] /= np.sum(start_st[k])
            start_st = np.array([[.5, .5], [.99, .01]])
            for method in range(len(methods)):
                results[method, trial, :, :] = demo(domain, methods[method], iterations, start_st)[1:]

        # store the results
        np.save(dom+'.WPL-MATH-LEAP.npy', results)

def wrapper_batch_testing():
    # batch testing global settings
    config.show_plots = False
    config.debug_output_level = -1

    # method = IGA(domain, P=BoxProjection())
    # method = WoLFIGA(domain, P=BoxProjection(), min_step=1e-4, max_step=1e-3 )
    # method = MySolver(domain, P=BoxProjection())
    # method = MyIGA(domain, P=BoxProjection())
    # method1 = WPL(domain, P=BoxProjection(low=.001))
    # method = AWESOME(domain, P=LinearProjection())
    # method = PGA_APP(domain, P=LinearProjection())
    # method = MultiAgentVI(domain, P=LinearProjection())
    # method2 = LEAP(domain, P=LinearProjection())

    domains = {'tricky':    Tricky(),
               'mp':        MatchingPennies(),
               'battle':    BachStravinski(),
               'pure':      PureCollaborative()}
    domains_b = {'deficientMP': MatchingPennies()}
    # results = [[] for _ in range(len(methods))]
    wrapper_batch_testing_helper(domains, 200, 200, 2)
    wrapper_batch_testing_helper(domains_b, 200, 8000, 1)


def wrapper_batch_testing_2(domain, start_st, no_trials, no_it):
    import copy
    config.show_plots = False
    config.debug_output_level = -1
    # method_settings = [['pga-pga', 2], ['wpl-pga-giga', 3], ['pga-giga', 2], ['pga-pga-giga', 3], ['pga-giga-giga', 3]]
    # method_settings = [['wpl-0.01',1],
    #                    ['pga-0.01',1],
    #                    ['pga-0.02',1],
    #                    ['pga-0.005',1],
    #                    ['pga-0.0001',1],
    #                    ['giga-0.05',1],
    #                    ['giga-0.08',1],
    #                    ['giga-0.008',1]]
    # method_settings = [['pga-giga-2',2]]
    method_settings = [['pga-0.01',1], ['giga-0.05',1], ['pga-giga', 2]]
    # method_settings = [['pga-0.05',1], ['pga-giga', 2]]
    # method_settings = [['pga-0.1',1]]
    # method_settings = [['pga-giga', 2], ['pga-0.01',1], ['giga-0.05',1]]
    for ms in method_settings:
        mms = [['follow-the-best', .01], ['linear-averaging', .0001], ['exponential-averaging', .01]]
        # mms = [['follow-the-best', .01]]
        # mms = [['exponential-averaging', .0001]]
        # mms = [['linear-averaging', .0001]]
        for cmms in mms:
            policies = np.empty((no_trials, no_it))
            forecasters = np.empty((no_trials, no_it, ms[1]))
            true_val_fun = np.empty((no_trials, no_it, domain.players))
            method = MATH(copy.deepcopy(domain), p=LinearProjection(), lfea_strategy=cmms[0],
                          hypo_eval_type='q-learning', q_learning_value_decay=cmms[1], solvers=ms[0])

            for i in range(no_trials):
                # print('Executing trial no %i of projection %s of %s.' % (i, cmms[0], ms[0]))
                p, f, t = demo(method.domain, method, no_it, start_st)
                policies[i, :] = p
                forecasters[i, :, :] = f
                true_val_fun[i, :, :] = t

            policies = np.array(policies)
            forecasters = np.array(forecasters)
            true_val_fun = np.array(true_val_fun)
            np.save('./results/'+ms[0]+'_'+cmms[0]+'_'+'policies', policies)
            np.save('./results/'+ms[0]+'_'+cmms[0]+'_'+'forecasters', forecasters)
            np.save('./results/'+ms[0]+'_'+cmms[0]+'_'+'true_val_fun', true_val_fun)

def wrapper_singular_runs():

    # start_st = np.array([[.5, .5], [.01, .99]])
    # start_st = 'random'
    # start_st = 'deterministic-s-i'
    start_st = 'deterministic-s'
    # start_st = 'deterministic-a'
    # start_st = 'lopsided'
    # start_st = 'lopsided-i'

    # domain = PureCollaborative()
    # domain = PureStrategy()
    # domain = YoungestSiblingDilemma()
    # domain = ShapleysGame()
    # domain = TaskAllocation(no_players=4, no_providers=2)
    # domain = BachStravinski(version='easy')
    domain = MatchingPennies()
    # domain = Tricky()
    # domain = PrisonersDilemma()

    # method = WoLFGIGA(domain, P=LinearProjection())
    # method = MATH(domain, p=LinearProjection(), lfea_strategy='follow-the-best')
    # method = MATH(domain, p=LinearProjection(), lfea_strategy='linear-averaging')
    # method = MATH(domain, p=LinearProjection(), lfea_strategy='exponential-averaging')
    # method = MATH(domain, p=LinearProjection(), lfea_strategy='follow-the-best', hypo_eval_type='q-learning',
    #               q_learning_value_decay=.01, solvers='pga-giga')
    method = MATH(domain, p=LinearProjection(), lfea_strategy='linear-averaging', hypo_eval_type='q-learning',
                  q_learning_value_decay=.0001, solvers='pga-giga')
    # method = MATH(domain, p=LinearProjection(), lfea_strategy='exponential-averaging', hypo_eval_type='q-learning',
    #               q_learning_value_decay=.001, solvers='pga-giga')
    # method = WoLFIGA(domain, P=BoxProjection())
    # method = LEAP(domain, P=LinearProjection())
    # method = WPL(domain, P=BoxProjection())

    # config.batch_testing = 1
    # wrapper_batch_testing_2(method, start_st, 400, 2)

    # demo(domain, method, 50000, start_strategies=start_st)
    # demo(domain, method, 10000, start_strategies=start_st)
    # demo(domain, method, 5000, start_strategies=start_st)
    # demo(domain, method, 2000, start_strategies=start_st)
    demo(domain, method, 1000, start_strategies=start_st)
    # demo(domain, method, 500, start_strategies=start_st)
    # demo(domain, method, 250, start_strategies=start_st)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--suppress-plots', action='store_false', dest='plot', default=True, help='suppress plots')
    parser.add_option('-v', '--show-output', type='int', dest='debug', default=0, help='debug level')
    (options, args) = parser.parse_args()
    config.debug_output_level = options.debug
    config.show_plots = options.plot

    domain = MatchingPennies()
    # domain = ShapleysGame()
    # domain = PureCollaborative()
    # domain = TaskAllocation(no_players=50, no_providers=25)
    start_st = 'lopsided'
    # start_st = 'deterministic-s-i'

    # wrapper_batch_testing_2(domain, start_st, 10, 10000)
    # wrapper_batch_testing_2(domain, start_st, 100, 5000)
    wrapper_singular_runs()
    # import Analyses as ana
    # ana.do_analysis()
    # ana.do_analysis_math_policies()
    # wrapper_batch_testing()
    # ana.do_analysis_math_hypotheses()