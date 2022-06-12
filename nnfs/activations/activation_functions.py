import numpy as np

#TODO: allow for other types of activation functions, that are not products of themselves
# def rprop_sigmoid(x, derivative=0):

#     x = np.array(x)
#     if derivative==0:
#         # return x/(1+np.abs(x))
#         return np.exp(-x)/(1+np.exp(-x))
#     else:
#         sig = x # 1/(1+np.exp(-x))
#         return -sig*(1-sig)

def sigmoid(x, derivative=0):

    x = np.array(x)
    if derivative==0:
        return 1/(1+np.exp(-x))
    else:
        sig = 1/(1+np.exp(-x)) # x
        return sig*(1-sig)

def tanh(x, derivative=0):
    x = np.array(x)
    if derivative==0:
        return np.tanh(x)
    else:
        return 1-np.tanh(x)**2 #1 - x**2 #

def relu(x, derivative=0):
    """
    Rectified Linear function

    Parameters
    ----------
    x : numpy.ndarray
        input from weights to layers / the output of layers
    derivative : int, optional
        If 0 it is the ReLu function
        else it is the derivative function, by default 0

    Returns
    -------
    numpy.ndarray
        - ReLu output
        - Derivative of input (assuming that we are getting data that has been
        passed through the function)

    References
    ----------
    - https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
    """
    x = np.array(x)
    if derivative==0:
        # return np.maximum(x, 0)
        return x * (x > 0)
    else:
        # return (x > 0) * 1
        x[x<=0] = 0
        x[x>0] = 1
        return x

def softmax(x, derivative=0):
    if derivative == 0:
        # Not stable
        # e_i = np.exp(x)
        # return e_i / e_i.sum()
        # Numerically stable
        e_i = np.exp(x-np.max(x))
        return e_i / e_i.sum()
    else:
        """
        https://stackoverflow.com/questions/54976533/derivative-of-softmax-function-in-python
        """
        s = x
        s = s.reshape(-1,1)
        result = np.diagflat(s) - np.dot(s, s.T)
        result=np.diagonal(result).reshape(1,result.shape[0])
        return np.array(result) # Extracting the diagonals from the array

def linear(x, derivative=0):
    if derivative==0:
        return np.array(x)
    else:
        return np.array(x)

# Cross Entropy
# https://deepnotes.io/softmax-crossentropy


# Types of activations functions
# v7labs.com/blog/neural-networks-activation-functions#:~:text=get%20started%20%3A)-,What%20is%20a%20Neural%20Network%20Activation%20Function%3F,prediction%20using%20simpler%20mathematical%20operations.
# Choosing types of activation functions
# https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/