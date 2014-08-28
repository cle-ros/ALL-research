__author__ = 'clemens'

import numpy as np


def load_data(file_name):
    import csv
    # defining the data structures
    raw_data = []
    file_name = './results/'+file_name+'.csv'

    # opening the file
    df = open(file_name, 'rb')
    try:
        reader_ob = csv.reader(df)
        for row in reader_ob:
        # adding the different dimensions. comment a line out to remove
        # the data from the result
            raw_data.append(row)

    finally:
        df.close()
    return np.array(raw_data, dtype=int)