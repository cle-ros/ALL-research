"""
This file defines some minor tools, mostly for debugging and inspection purposes.
"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy
import numpy as np

__author__ = 'clemens'


def dprint(*options):
    """
    A special print function, that prints the call to the function as well; includes calling function, file, line
    and the name of the variable printed.
    :param options: The options to be printed.
    :return:
    """
    import inspect
    import re
    # the inspect stack
    st = inspect.stack()
    var_names = re.split('dprint *\(', str(st[1][4][0]))[1]
    var_names = re.split('\) *\n', var_names)[0]
    var_names = re.split(' *, *', var_names)
    len_of_name_str = 12
    for i in range(len(var_names)):
        vn = var_names[i]
        appendix = ' '*(len_of_name_str-len(vn))
        var_names[i] = vn + appendix
    function_name = st[1][3]
    function_path = st[1][1]
    function_file = re.split('/', function_path)[-1]
    print(' + The calling function:', function_name, '(', function_file, 'L', st[1][2], ')')
    for i in range(len(options)):
        if isinstance(options[i], list) or isinstance(options[i], numpy.ndarray):
            if len(var_names[i]) <= len_of_name_str + 10:
                print(' |   * ', var_names[i], '    ->  ', options[i][0])
            else:
                print(' |   * ', var_names[i])
                print(' |     '+' '*(len_of_name_str+6)+'->  ', options[i][0])
            for j in range(1, len(options[i]), 1):
                print(' |     '+' '*(len_of_name_str+10), options[i][j])
        else:
            if len(var_names[i]) <= len_of_name_str:
                print(' |   * ', var_names[i], ' '*(len_of_name_str - len(var_names[i])), '    ->  ', options[i])
            else:
                print(' |   * ', var_names[i])
                print(' |', ' '*(len_of_name_str + 5), '    ->  ', options[i])
    print(' +-- End of output of:  ', function_name, '(', function_file, 'L', st[1][2], ')')


def ralen(*args):
    """
    A shorthand for range(len(variable))
    :param args:
    :return:
    """
    if len(args) == 1:
        st = 0
        end = len(args[0])
        inc = 1
    else:
        st = args[0]
        end = len(args[1])
        inc = args[2]
    return range(st, end, inc)


def smooth_1d_sequence(sequence, sigma=15):
    """
    This function smoothes a 1-dimensional sequence, using a gaussian filter.
    :param sequence:
    :param sigma:
    :return:
    """
    # smoothing functions for more readable plotting
    from scipy.ndimage import gaussian_filter1d
    sequence = np.array(sequence)
    assert len(sequence.shape) <= 2, 'Cannot interpret an array with more than 2 dimensions as a tuple of 1d sequences.'
    # asserting that the data is in the rows and that the array has a second dimension (for the for loop)
    if max(sequence.shape) > min(sequence.shape):
        if sequence.shape[1] > sequence.shape[0]:
            sequence = sequence.T
    else:
        sequence = sequence[None]
    for i in range(sequence.shape[1]):
        val_interpol = np.interp(range(sequence.shape[0]), range(sequence.shape[0]), sequence[:, i])
        sequence[:, i] = gaussian_filter1d(val_interpol, sigma)
    return sequence


def plot_results(data, columns=2, shx=False, shy=False, line_width=2, legend_location=4):
    """
    A dictionary-based plot function, that will plot data in the following form:
    {
        'Variable Name' : {
                            'values':  [array of the data; will plot along the larger dimension]
                            'yLimits': [array of the min and max values on the plot]
                            'smooth':  int specifying interpolation for better looks
                          },
        'Another Name'  : { ... }
    }
    :param data:
    :param columns:
    :param shx:
    :param shy:
    :return:
    """
    if columns > 1 and len(data) > 1:
        a = float(len(data))
        c = float(columns)
        first_dim = int(np.ceil(a/c))
        second_dim = columns
        ax_indices = []
        for j in range(second_dim):
            for i in range(first_dim):
                ax_indices.append([i, j])
    else:
        first_dim = len(data)
        second_dim = 1
        ax_indices = range(first_dim)
    # creating the indices

    def create_plot(ax, name, points, ylimits, smooth, labels, lstyle, marker, colors, lwidth, ax_labels):
        lines = np.array(points)
        if smooth != -1:
            lines = smooth_1d_sequence(points, smooth)
        if lines.shape[0] > lines.shape[1]:
            lines = lines.T
        for i in range(lines.shape[0]):
            ax.plot(lines[i], linewidth=lwidth[i], linestyle=lstyle[i], label=labels[i], marker=marker[i], color=colors[i])
        ax.set_title(name)
        ax.legend(loc=legend_location)
        ax.grid(True)
        ax.set_ylabel(ax_labels[1], fontsize=15, labelpad=-1)
        ax.set_xlabel(ax_labels[0], fontsize=15, labelpad=-2)
        if ylimits is not None:
            ax.set_ylim(ylimits)

    data_processed = []
    for name in data.keys():
        ds = data[name]
        axes_labels = ['Iterations/Number of plays', 'Probability of first action'] \
            if 'axesLabels' not in ds else ds['axesLabels']
        no_lines = min(ds['values'].shape)
        points = np.array(ds['values'])
        ylimits = None if 'yLimits' not in ds else ds['yLimits']
        labels = ['Line '+str(i) for i in range(1, no_lines+1, 1)] if 'labels' not in ds else ds['labels']
        marker = [None for _ in range(no_lines)] if 'markers' not in ds else ds['markers']
        smoothing = -1 if 'smooth' not in ds else ds['smooth']
        linewidths = [line_width for _ in range(no_lines)] if 'linewidths' not in ds else ds['linewidths']
        if 'linestyles' in ds:
            linestyle = ds['linestyles']
            colors = ['black' for _ in range(no_lines)] if 'colors' not in ds else ds['colors']
        else:
            linestyle = ['-' for _ in range(no_lines)]
            colors = [None for _ in range(no_lines)] if 'colors' not in ds else ds['colors']

        data_processed.append({'name':      name,
                               'values':    points,
                               'ylimits':   ylimits,
                               'labels':    labels,
                               'smooth':    smoothing,
                               'linestyle': linestyle,
                               'marker':    marker,
                               'colors':    colors,
                               'linewidths':linewidths,
                               'axlabels':  axes_labels,
                               }
                              )

    fig, ax_all = plt.subplots(first_dim, second_dim, sharex=shx, sharey=shy, figsize=(6, 4), dpi=80)
    for i in range(len(ax_indices)):
        try:
            try:
                axe = ax_all[ax_indices[i][0], ax_indices[i][1]]
            except TypeError:
                try:
                    axe = ax_all[ax_indices[i]]
                except TypeError:
                    axe = ax_all
            dprint(data_processed[i]['name'])
            create_plot(axe,
                        data_processed[i]['name'],
                        data_processed[i]['values'],
                        data_processed[i]['ylimits'],
                        data_processed[i]['smooth'],
                        data_processed[i]['labels'],
                        data_processed[i]['linestyle'],
                        data_processed[i]['marker'],
                        data_processed[i]['colors'],
                        data_processed[i]['linewidths'],
                        data_processed[i]['axlabels'],
                        )
        except IndexError:
            break
    fig.tight_layout()
    # show the plots
    plt.show()


def store_results(result, approach, params, folder='./'):
    """
    This function stores the given result, with the approach and parameters
    specified, in a file following this name pattern:
    '../results/month-day_hour-min-sec_approach_parameters.csv'
    :param result: the matrix
    :param approach: the name of the approach
    :param params: the parameters chosen
    :param folder: the folder the files are stored in. Defaults to ./
    :return: None
    :type result: numpy.ndarray
    :type approach: str
    :type params: str
    :type folder: str
    :rtype : object
    """
    import csv
    from time import gmtime, strftime
    # import numpy
    # if type(result) is numpy.ndarray:
    #     to_be_stored = result.tolist()
    # else:
    #     to_be_stored = result
    # generating a time-string
    time_string = strftime('%m-%d_%H-%M-%S', gmtime())
    # generating the filename
    file_name = folder + time_string + '_' + approach + '_' + params + '.csv'
    # opening the csv file
    o_file = open(file_name, 'w')
    writer = csv.writer(o_file)
    try:
        dprint(result.tolist())
        writer.writerows(result.tolist())
        # writer.writerow(to_be_stored.tolist())
    except AttributeError:
        writer.writerows(result)
        # writer.writerow(to_be_stored)
    # and closing it
    o_file.close()
    print('Result successfully stored in ' + file_name)