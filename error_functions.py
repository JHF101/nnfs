import numpy as np

def mse(y_actual, y_out, derivative=0):
    if derivative==0:
        # E_y = []
        # for i in range(len(y_out)):
        #     E_y.append((1/2)*(y_actual[i]-y_out[i])**2)
        # ##print(E_y)

        # E_total = np.sum(E_y)
        # return E_total
        return np.mean(np.power(y_actual-y_out, 2))
    else:
        y_actual = np.array(y_actual)
        y_out = np.array(y_out)
        dE_total_wrt_dy = 2*(y_out- y_actual)/y_actual.size
        # print(f"dE_total/d_y {dE_total_wrt_dy}")
        return dE_total_wrt_dy

def squared_error(y_actual, y_out, derivative=0):
    if derivative==0:
        return np.sum(np.power(y_actual-y_out, 2))
    else:
        y_actual = np.array(y_actual)
        y_out = np.array(y_out)
        dE_total_wrt_dy = 2*(y_out- y_actual)
        # print(f"dE_total/d_y {dE_total_wrt_dy}")
        return dE_total_wrt_dy


def rms(y_actual, y_out, derivative=0):
    if derivative==0:
        # return np.sqrt(np.sum(np.power(y_actual-y_out, 2)))
        return NotImplementedError
    else:
        return NotImplementedError

