__author__ = 'clemens'
import pycuda as pc
import numpy as np
import scipy as sp
import multiprocessing as mp


no_iter = 5


def with_cuda():
    pc.tools
    pass


def without_cuda():
    pool = mp.Pool(4) # up to 4 processes at once
    result1 = pool.apply_async(test_function(no_iter))
    result2 = pool.apply_async(test_function(no_iter))
    result1.get()
    result2.get()


def test_function(no_iterations):
    for _ in range(no_iterations):
        s = 5000
        mat1 = np.random.random((s, s))
        mat2 = np.random.random((s, s))
        np.dot(mat1, mat2)
