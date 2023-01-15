import numpy as np


def mse(y_actual, y_out, derivative=0):
    if derivative == 0:
        return np.mean(np.power(y_actual - y_out, 2))
    else:
        y_actual = np.array(y_actual)
        y_out = np.array(y_out)
        dE_total_wrt_dy = 2 * (y_out - y_actual) / y_actual.size
        return dE_total_wrt_dy


def squared_error(y_actual, y_out, derivative=0):
    if derivative == 0:
        return np.sum(np.power(y_actual - y_out, 2))
    else:
        y_actual = np.array(y_actual)
        y_out = np.array(y_out)
        dE_total_wrt_dy = 2 * (y_out - y_actual)
        return dE_total_wrt_dy


def rms(y_actual, y_out, derivative=0):
    if derivative == 0:
        return NotImplementedError
    else:
        return NotImplementedError


# TODO: Test cross entropy
def cross_entropy(y_actual, y_out, derivative=0):
    y_actual = np.array(y_actual)
    y_out = np.array(y_out)

    if derivative == 0:
        network_out = y_out.shape[0]

        exps = np.exp(y_actual - np.max(y_actual))
        softmax_out = exps / np.sum(exps)

        log_likelihood = -np.log(softmax_out[range(network_out), y_out])
        loss = np.sum(log_likelihood) / network_out
        return loss
    else:
        network_out = y_out.shape[0]

        # Getting gradients
        exps = np.exp(y_actual - np.max(y_actual))
        dE_total_wrt_dy = exps / np.sum(exps)

        dE_total_wrt_dy[range(network_out), y_out] -= 1
        dE_total_wrt_dy = dE_total_wrt_dy / network_out
        return dE_total_wrt_dy
