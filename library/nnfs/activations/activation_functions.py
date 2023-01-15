import numpy as np

"""
Activation function for output layer based on prediction problem:
    - Regression - Linear
    - Binary Classification - Sigmoid
    - Multiclass Classification - Softmax
    - Multilabel Classification - Sigmoid
"""


def sigmoid(x, derivative=0):
    """
    Sigmoid function.

    Note: Suffers from Vanishing gradient problem due to suppression of larger
    values from it's derivative function. Training is more difficult because
    of the asymmetry of the function along the x-axis.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs to be transformed by activation function
    derivative : int, optional
        Specifies whether the derivative function should be used (!=0) else
        use the function, by default 0

    Returns
    -------
    numpy.ndarray
        - Sigmoid output
        - Derivative of output
    """
    # x = np.array(x)
    if derivative == 0:
        return 1 / (1 + np.exp(-x))
    else:
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)


def tanh(x, derivative=0):
    """Tanh function.

    Note: Suffers from Vanishing gradient problem.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs to be transformed by activation function
    derivative : int, optional
        Specifies whether the derivative function should be used (!=0) else
        use the function, by default 0

    Returns
    -------
    numpy.ndarray
        - Tanh output
        - Derivative of tanh
    """
    # x = np.array(x)
    if derivative == 0:
        return np.tanh(x)
    else:
        return 1 - np.tanh(x) ** 2


def relu(x, derivative=0):
    """
    Rectified Linear Unit function.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs to be transformed by activation function
    derivative : int, optional
        Specifies whether the derivative function should be used (!=0) else
        use the function, by default 0

    Returns
    -------
    x : numpy.ndarray
        - ReLu output
        - Derivative of output
    """
    # x = np.array(x)
    if derivative == 0:
        return x * (x > 0)
    else:
        x = np.where(x <= 0, 0, 1)
        return x


def softmax(x, derivative=0):
    """Softmax function.

    Note: The softmax function is usually used in the final layer of the
    network to predict a multinomial distribution.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs to be transformed by activation function
    derivative : int, optional
        Specifies whether the derivative function should be used (!=0) else
        use the function, by default 0

    Returns
    -------
    x : numpy.ndarray
        - Softmax output
        - Derivative of output
    """
    if derivative == 0:
        # -- Not stable -- #
        # e_i = np.exp(x)
        # return e_i / e_i.sum()
        # Numerically stable
        e_i = np.exp(x - np.max(x))
        return e_i / e_i.sum()
    else:
        s = x
        s = s.reshape(-1, 1)
        result = np.diagflat(s) - np.dot(s, s.T)
        result = np.diagonal(result).reshape(1, result.shape[0])
        return np.array(result)  # Extracting the diagonals from the array


def linear(x, derivative=0):
    """Linear activation function (Identity Function).

    Note: Derivative has no relation to the input. If a linear function
    is used the inputs will have a linear relationship to the output.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs to be transformed by activation function
    derivative : int, optional
        Specifies whether the derivative function should be used (!=0) else
        use the function, by default 0

    Returns
    -------
    x : numpy.ndarray
        - Linear output
        - Derivative of output
    """
    if derivative == 0:
        return np.ones(x.shape)
    else:
        return np.array(x)


def leaky_relu(x, derivative=0):
    """Similar to the relu function but does not have the dead
    region like the relu.

    Note: Gradient for negative values is small that makes the
    learning model parameters time-consuming.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs to be transformed by activation function
    derivative : int, optional
        Specifies whether the derivative function should be used (!=0) else
        use the function, by default 0

    Returns
    -------
    x : numpy.ndarray
        - Leaky relu output
        - Derivative of output
    """
    # x = np.array(x)
    alpha = 0.1
    if derivative == 0:
        return np.maximum(alpha * x, x)
    else:
        x = np.where(x <= 0, alpha, 1)
        return x


def elu(x, derivative=0):
    """
    Exponential Linear Unit function.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs to be transformed by activation function
    derivative : int, optional
        Specifies whether the derivative function should be used (!=0) else
        use the function, by default 0

    Returns
    -------
    x : numpy.ndarray
        - elu output
        - Derivative of output
    """
    # x = np.array(x)
    alpha = 0.1
    if derivative == 0:
        x = np.where(x >= 0, x, alpha * (np.exp(x) - 1))
        return x
    else:
        x = np.where(x >= 0, alpha * (np.exp(x)), 1)
        return x


# Cross Entropy
# https://deepnotes.io/softmax-crossentropy


# Types of activations functions
# v7labs.com/blog/neural-networks-activation-functions#:~:text=get%20started%20%3A)-,What%20is%20a%20Neural%20Network%20Activation%20Function%3F,prediction%20using%20simpler%20mathematical%20operations.
# Choosing types of activation functions
# https://machinelearningmastery.com/choose-an-activation-function-for-deep-learnin
# https://www.v7labs.com/blog/neural-networks-activation-functions
