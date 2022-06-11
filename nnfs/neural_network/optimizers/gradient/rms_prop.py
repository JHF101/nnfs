from .gradient_optimizer import GradientOptimizer
import numpy as np
from utils.logs import create_logger

log = create_logger(__name__)

class RMSProp(GradientOptimizer):
    def __init__(self, learning_rate, beta=0.99, weights_initialization=None):
        super().__init__()
        self.initialization_method = weights_initialization
        self.beta = beta
        self.learning_rate = learning_rate
        self.optimizer_name = 'gradient descent with momentum'
        log.info(f"Optimizer Name: {self.optimizer_name}")

        # Requires a moving average
        self.sdW = None
        self.sdB = None

    def optimize(self, **kwargs):
        # --- Weights
        dE_dwij_t = kwargs["dE_dwij_t"] 
        weights = kwargs["weights"]
        mega_delta_weights_array = kwargs["mega_delta_weights_array"]

        # --- Bias
        if self.use_bias:
            dE_dbij_t = kwargs["dE_dbij_t"]
            bias = kwargs["bias"]
            mega_delta_bias_array = kwargs["mega_delta_bias_array"]
        


        for w in range(0, len(weights)):
            if (len(mega_delta_weights_array)>1):
                # --- Weights
                dW = dE_dwij_t[w]
                self.sdW[w] = (self.beta* self.sdW[w]) + (1-self.beta)*(np.power(dW, 2)) # Element wise squaring
                weights[w] -= self.learning_rate * (dW/np.sqrt(self.sdW[w]))

                # --- Bias
                if self.use_bias:
                    dB = dE_dbij_t[w]
                    self.sdB[w] = (self.beta* self.sdB[w]) + (1-self.beta)*(np.power(dB, 2)) # Element wise squaring
                    bias[w] -= self.learning_rate * (dB/np.sqrt(self.sdB[w]))
            else:
                # Initializing the shape of the weights
                if self.sdW is None:
                    self.sdW = []
                    for i in dE_dwij_t:
                        self.sdW.append(np.zeros(shape=i.shape))

                if self.use_bias:
                    if self.sdB is None:
                        self.sdB = []
                        for b in dE_dbij_t:
                            self.sdB.append(np.zeros(shape=b.shape))
                
                # Using normal GD if we cannot look back
                # --- Weights
                weights[w] -= self.learning_rate * dE_dwij_t[w]
                # --- Bias
                if self.use_bias:
                    bias[w] -= self.learning_rate * dE_dbij_t[w]

        if self.use_bias:
            return weights, bias
        else:
            return weights, 0
