__author__ = 'clemens'


def store_results(result, approach, params):
    """
    This function stores the given result, with the approach and parameters
    specified, in a file following this name pattern:
    '../results/month-day_hour-min-sec_approach_parameters.csv'
    """
    import csv
    from time import gmtime, strftime
    # generating a time-string
    time_string = strftime('%m-%d_%H-%M-%S', gmtime())
    # generating the filename
    file_name = 'results/' + time_string + '_' + approach + '_' + params + '.csv'
    # opening the csv file
    o_file = open(file_name, 'w')
    writer = csv.writer(o_file)
    try:
        writer.writerow(result.tolist())
    except AttributeError:
        writer.writerow(result)
    # and closing it
    o_file.close()