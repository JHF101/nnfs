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


def min_max_scaler(data):
    """
    Min-Max scaler function to scale input data between 0 and 1.

    Parameters
    ----------
    data : numpy array or list of lists
        The input data to be scaled.

    Returns
    -------
    numpy array
        Scaled data between 0 and 1.
    """
    # Convert data to numpy array if it's not already
    data = np.array(data)

    # Calculate min and max for each feature (column)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero if min and max are the same
    max_vals[max_vals == min_vals] += 1e-6

    # Scale the data to [0, 1]
    scaled_data = (data - min_vals) / (max_vals - min_vals)

    return scaled_data


def standard_scaler(data):
    """
    Standard scaler function to standardize input data to have mean 0 and standard deviation 1.

    Parameters
    ----------
    data : numpy array or list of lists
        The input data to be standardized.

    Returns
    -------
    numpy array
        Standardized data with mean 0 and standard deviation 1.
    """
    # Convert data to numpy array if it's not already
    data = np.array(data)

    # Calculate mean and standard deviation for each feature (column)
    mean_vals = data.mean(axis=0)
    std_vals = data.std(axis=0)

    # Avoid division by zero if standard deviation is 0
    std_vals[std_vals == 0] = 1e-6

    # Standardize the data
    standardized_data = (data - mean_vals) / std_vals

    return standardized_data


# if __name__=="__main__":
#     print(to_categorical(np.array([7,7,6,7])))
