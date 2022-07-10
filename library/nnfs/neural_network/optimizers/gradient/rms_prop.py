from nnfs.neural_network.optimizers.gradient.gradient_optimizer import GradientOptimizer
import numpy as np
from nnfs.utils.logs import create_logger

log = create_logger(__name__)

class RMSProp(GradientOptimizer):
    def __init__(self, learning_rate, beta=0.9, weights_initialization=None):
        super().__init__()
        self.initialization_method = weights_initialization
        self.beta = beta
        self.learning_rate = learning_rate
        self.optimizer_name = 'rms prop'
        log.info(f"Optimizer Name: {self.optimizer_name}")

        # Requires a moving average
        self.s_dW = None
        self.s_dB = None
        self.eps = 10**(-8)

    def optimize(self, **kwargs):
        # --- Weights
        dE_dwij_t = kwargs["dE_dwij_t"]
        weights = kwargs["weights"]

        # --- Bias
        if self.use_bias:
            dE_dbij_t = kwargs["dE_dbij_t"]
            bias = kwargs["bias"]

        # Initializing the shape of the weights
        if self.s_dW is None:
            self.s_dW = []
            for i in dE_dwij_t:
                self.s_dW.append(np.zeros(shape=i.shape))

        if self.use_bias:
            if self.s_dB is None:
                self.s_dB = []
                for b in dE_dbij_t:
                    self.s_dB.append(np.zeros(shape=b.shape))

        for w in range(0, len(weights)):
            # --- Weights
            dW = dE_dwij_t[w]
            self.s_dW[w] = (self.beta* self.s_dW[w]) + (1-self.beta)*(np.power(dW, 2)) # Element wise squaring
            weights[w] -= self.learning_rate * (np.divide(dW, np.sqrt(self.s_dW[w])+self.eps))

            # --- Bias
            if self.use_bias:
                dB = dE_dbij_t[w]
                self.s_dB[w] = (self.beta* self.s_dB[w]) + (1-self.beta)*(np.power(dB, 2)) # Element wise squaring
                bias[w] -= self.learning_rate * (np.divide(dB, (np.sqrt(self.s_dB[w])+self.eps)))

        if self.use_bias:
            return weights, bias
        else:
            return weights, 0
