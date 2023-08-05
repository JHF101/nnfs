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
        return np.sqrt(np.mean(np.power(y_actual - y_out, 2)))
    else:
        y_actual = np.array(y_actual)
        y_out = np.array(y_out)
        dE_total_wrt_dy = 2 * (y_out - y_actual) / len(y_actual)
        return dE_total_wrt_dy


def cross_entropy(y_actual, y_out, derivative=0):
    y_actual = np.array(y_actual)
    y_out = np.array(y_out)

    if derivative == 0:
        network_out = y_out.shape[0]

        exps = np.exp(y_out - np.max(y_out))  # Apply softmax to y_out
        softmax_out = exps / np.sum(exps)

        # Convert y_actual to integer type if it's not already
        y_actual = np.array(y_actual, dtype=int)

        log_likelihood = -np.log(
            softmax_out[np.arange(network_out), y_actual]
        )  # Use np.arange for indexing
        loss = np.sum(log_likelihood) / network_out
        return loss
    else:
        network_out = y_out.shape[0]

        exps = np.exp(y_out - np.max(y_out))  # Apply softmax to y_out
        softmax_out = exps / np.sum(exps)

        dE_total_wrt_dy = softmax_out.copy()
        y_actual = np.array(y_actual, dtype=int)
        dE_total_wrt_dy[np.arange(network_out), y_actual] -= 1
        dE_total_wrt_dy = dE_total_wrt_dy / network_out
        return dE_total_wrt_dy
