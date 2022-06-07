import numpy as np
import logging

def shuffle_arrays(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]

def argmax_one_hot(array, len_encoding=10):
    max_val = np.max(array)
    return_array = []
    if len_encoding != len(array):
        raise Exception("The length of the arrays are not equal")
    for i in range(0, len_encoding):
        if (max_val is array[i]):
            return_array.append(1.0)
        else:
            return_array.append(0.0)
    return np.array(return_array)

def to_categorical(array):
    result = np.zeros((array.shape[0],10))
    for i in range(len(result)):
        result[i][array[i]] = 1
    return result

if __name__=="__main__":
    to_categorical(np.array([7,7,6,7]))
