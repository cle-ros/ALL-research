__author__ = 'clemens'
import numpy as np
import os
import re

from marl.ToolsGeneral import dprint, ralen


def load_data(dir, cont=None):
    data = {}
    direct = dir
    for res in os.listdir(direct):
        if res.endswith('.npy'):
            if cont is not None:
                if cont in res:
                    data[res.split('.npy')[0]] = np.load(direct+res)
            else:
                data[res.split('.npy')[0]] = np.load(direct+res)
            # if 'deficient' in res or 'mp' in res:
            #     print 'hi'
            #     data[res.split('.')[0]] *= .5
            #     data[res.split('.')[0]] += .5
            # print data[res.split('.')[0]].shape
    return data


def do_analysis_math_policies():
    # data = load_data('./results/', 'pga-0.1')
    data = load_data('./results/', 'policies')
    results = {}
    for key in data:
        average_convergence = [[.2, .1, .05, .025], [[], [], [], []]]
        results[key] = {}
        for c in range(len(average_convergence[0])):
            for policy in data[key]:
                norm_policy = np.abs(policy - .5) < average_convergence[0][c]
                average_convergence[1][c].append(len(norm_policy)-np.argmin(norm_policy[::-1]))
            results[key][average_convergence[0][c]] = np.mean(average_convergence[1][c])
    for key in sorted(results.keys()):
        key_comp = ''
        for key2 in sorted(results[key])[::-1]:
            key2_str = ' '*(5-len(str(key2)))
            key2_str += str(key2)
            key_comp += key2_str +', '+ str(int(np.ceil(results[key][key2])))+ ', '
        # print('%s: %s: %i' % (key_str, key2_str, results[key][key2]))
        print(key+', '+key_comp)


def do_analysis_math_hypotheses():
    data = load_data('./results/', 'forecasters')
    results = {}
    for key in data:
        average_convergence = [[.2, .1, .05, .025], [[], [], [], []]]
        results[key] = {}
        for c in range(len(average_convergence[0])):
            for policy in data[key]:
                norm_policy = np.abs(policy - .5) < average_convergence[0][c]
                average_convergence[1][c].append(norm_policy.shape[0]-np.argmin(norm_policy[::-1], axis=0))
            results[key][average_convergence[0][c]] = np.mean(np.array(average_convergence[1][c]), axis=0)
    for key in sorted(results.keys()):
        key_comp = ''
        for key2 in sorted(results[key])[::-1]:
            # dprint(key2)
            key2_str = ' '*(5-len(str(key2)))
            key2_str += str(key2)
            hypo_str = ''
            for hyp in np.ceil(results[key][key2]):
                hypo_str += str(int(hyp)) + ', '
            key_comp += key2_str +', ' + hypo_str
        # print('%s: %s: %i' % (key_str, key2_str, results[key][key2]))
        print(key+', '+key_comp)


def do_analysis_leap():
    data = load_data()
    averages ={}
    for key in data:
        averages_tmp = np.zeros((3,200))
        for i in range(3):
            for j in range(200):
                tmpdat = (np.array(data[key]) + data[key].min())/(data[key].max()-data[key].min())
                # tmpdat = np.array(data[key])
                averages_tmp[i, j] = np.mean(tmpdat[i, j, :100, 1])
        averages[key] = averages_tmp
        print '\n -~-~-~-~-~-~-~-~-~-~-'
        print key
        print 'Max values:      ', averages_tmp.max(axis=1)
        print 'Min values:      ', averages_tmp.min(axis=1)
        print 'Average values:  ', averages_tmp.mean(axis=1)