from .gradient_optimizer import GradientOptimizer
import logging
import numpy as np

from utils.logs import create_logger
log = create_logger(__name__)
class Rprop(GradientOptimizer):
    def __init__(self, eta_plus, eta_minus, delta_max, delta_min, weights_initialization=None):
        super().__init__()
        self.initialization_method = weights_initialization

        # TODO: Add modification to get RProp+ etc
        self.optimizer_name = 'Rprop'
        log.info(f"Optimizer Name: {self.optimizer_name}")

        self.rprop_init_process = True
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta_max = delta_max
        self.delta_min = delta_min

        self.del_ij_t_w = []
        self.dE_dwij_t_1 = []

        # Bias
        self.del_ij_t_b = []
        self.dE_dbij_t_1 = []


    def optimize(self, **kwargs):
        # --- Weights
        dE_dwij_t = kwargs["dE_dwij_t"]
        weights = kwargs["weights"]

        # --- Bias
        if self.use_bias:
            dE_dbij_t = kwargs["dE_dbij_t"]
            bias = kwargs["bias"]

        # --------------------------------------------------------------------------------- #
        #                                       RPROP                                       #
        # --------------------------------------------------------------------------------- #
        # TODO: Optimization can be done with numpy.where

        if self.rprop_init_process== True:
            # ---- Initialization ---- #
            for w in range(0,len(weights)):
                gradient_multiplication = dE_dwij_t[w]
                if self.use_bias:
                    bias_multiplication = dE_dbij_t[w]

                self.del_ij_t_w.append(np.random.uniform(low=0.05, high=0.2, size=gradient_multiplication.shape))
                self.dE_dwij_t_1.append(np.random.uniform(low=0.05, high=0.2, size=gradient_multiplication.shape))

                if self.use_bias:
                    self.del_ij_t_b.append(np.random.uniform(low=0.05, high=0.2, size=bias_multiplication.shape))
                    self.dE_dbij_t_1.append(np.random.uniform(low=0.05, high=0.2, size=bias_multiplication.shape))

        self.rprop_init_process= False

        if len(dE_dwij_t) != len(self.dE_dwij_t_1):
            raise Exception("The size of the derivatives in RProp do not match")

        if self.use_bias:
            if len(dE_dwij_t) != len(dE_dbij_t):
                raise Exception("Length of bias gradients does not equal the length of the weights gradients")

        for outer in range(0, len(dE_dwij_t)): # Looping through the first set of weights

            if (dE_dwij_t[outer].shape[0] != self.dE_dwij_t_1[outer].shape[0]):
                raise Exception("The element size of derivatives in RProp do not match")

            for p in range(0, dE_dwij_t[outer].shape[0]):

                if (dE_dwij_t[outer].shape[1] != self.dE_dwij_t_1[outer].shape[1]):
                    raise Exception("The row size of derivatives in RProp do not match")

                for q in range(0, dE_dwij_t[outer].shape[1]):

                    # (dE(t)/dw_i(t))*(dE(t-1)/dw_i(t-1))
                    gradient_mult = self.dE_dwij_t_1[outer][p][q] * dE_dwij_t[outer][p][q]
                    # print("Index", outer, p, q)
                    if (gradient_mult > 0.0): # This loop only runs once
                        self.del_ij_t_w[outer][p][q] = np.minimum(self.del_ij_t_w[outer][p][q] * self.eta_plus, self.delta_max)
                        del_w_ij_t = -1.0 * np.sign(dE_dwij_t[outer][p][q]) * self.del_ij_t_w[outer][p][q]
                        weights[outer][p][q] = weights[outer][p][q] + del_w_ij_t
                        self.dE_dwij_t_1[outer][p][q] = dE_dwij_t[outer][p][q]

                    elif (gradient_mult < 0.0):
                        self.del_ij_t_w[outer][p][q] = np.maximum(self.del_ij_t_w[outer][p][q] * self.eta_minus, self.delta_min) # Error was delta min was huge
                        self.dE_dwij_t_1[outer][p][q] = 0.0
                        # if train_err_arr[-1]>train_err_arr[-2]: #Modification for RPROP+
                        #     self.weights[outer][p][q] -= self.learning_rate*dE_dwij_t[outer][p][q]

                    elif (gradient_mult == 0.0):
                        del_w_ij_t = -1.0 * np.sign(dE_dwij_t[outer][p][q]) * self.del_ij_t_w[outer][p][q]
                        weights[outer][p][q] = weights[outer][p][q] + del_w_ij_t
                        self.dE_dwij_t_1[outer][p][q] = dE_dwij_t[outer][p][q]
                    else:
                        raise Exception("There exists no value you are looking for")

            if self.use_bias:
                # Assuming bias and weights arrats are the same length
                for s in range(0, dE_dbij_t[outer].shape[0]):
                    for r in range(0, dE_dbij_t[outer].shape[1]):

                        bias_grad_mult = self.dE_dbij_t_1[outer][s][r] * dE_dbij_t[outer][s][r]

                        if ( bias_grad_mult > 0.0): # This loop only runs once

                            self.del_ij_t_b[outer][s][r] = np.minimum(self.del_ij_t_b[outer][s][r] * self.eta_plus, self.delta_max)
                            del_b_ij_t = -1.0 * np.sign(dE_dbij_t[outer][s][r]) * self.del_ij_t_b[outer][s][r]
                            bias[outer][s][r] = bias[outer][s][r] + del_b_ij_t
                            self.dE_dbij_t_1[outer][s][r] = dE_dbij_t[outer][s][r]

                        elif (bias_grad_mult < 0.0):
                            self.del_ij_t_b[outer][s][r] = np.maximum(self.del_ij_t_b[outer][s][r] * self.eta_minus, self.delta_min) # Error was delta min was huge
                            self.dE_dbij_t_1[outer][s][r] = 0.0
                            # if train_err_arr[-1]>train_err_arr[-2]: #Modification for RPROP+
                            #     self.bias[outer][s][r] -=  self.learning_rate*dE_dbij_t[outer][s][r]

                        elif (bias_grad_mult == 0.0):
                            del_b_ij_t = -1.0 * np.sign(dE_dbij_t[outer][s][r]) * self.del_ij_t_b[outer][s][r]
                            bias[outer][s][r] = bias[outer][s][r] + del_b_ij_t
                            self.dE_dbij_t_1[outer][s][r] = dE_dbij_t[outer][s][r]
                            # print(self.weights[outer][s][q])
                        else:
                            raise Exception("There exists no value you are looking for")

        # --------------------------------------------------------------------------------- #
        #                                   END RPROP                                       #
        # --------------------------------------------------------------------------------- #

        if self.use_bias:
            return weights, bias
        else:
            return weights, 0