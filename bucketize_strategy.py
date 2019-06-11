import numpy as np
"""
you can add your own straget for bucketization. here we use uniform
"""


def uniform_bucketize(data_range, n_buckets):
    min_val = data_range[0]
    max_val = data_range[1]
    return np.linspace(min_val,max_val,n_buckets+1)