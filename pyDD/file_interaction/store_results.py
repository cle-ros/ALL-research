__author__ = 'clemens'


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
        writer.writerows(result.tolist())
        # writer.writerow(to_be_stored.tolist())
    except AttributeError:
        writer.writerows(result)
        # writer.writerow(to_be_stored)
    # and closing it
    o_file.close()
    print('Result successfully stored in ' + file_name)