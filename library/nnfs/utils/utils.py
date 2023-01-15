import numpy as np


def shuffle_arrays(X, y):
    """Shuffles the X data and y data, while keeping the corresponding features
    and labels matched together. Goal of this function is to remove the
    influence of the order of the data.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset features
    y : numpy.ndarray
        Dataset labels

    Returns
    -------
    X and y : np.ndarray
        Redordered label and feature data.
    """
    if len(X) == len(y):
        p = np.random.permutation(len(X))
    else:
        raise Exception(f"len(X) == len(y) where X={X} does not equal y={y}")
    return X[p], y[p]


def to_categorical(array):
    """Generates one-hot encoded array based on
    integer value input.

    Parameters
    ----------
    array : numpy.ndarray
        Integer values in a 1-dim array

    Returns
    -------
    result: np.ndarray
        2-dim one-hot encoded arrays based on integer value
        at that index.
    """
    result = np.zeros((array.shape[0], 10))
    for i in range(len(result)):
        result[i][array[i]] = 1
    return result


# if __name__=="__main__":
#     print(to_categorical(np.array([7,7,6,7])))
