from .gradient_optimizer import GradientOptimizer
import logging
log = logging.getLogger(__name__)

class GradientDescentWithMomentum(GradientOptimizer):
    def __init__(self, learning_rate, beta, weights_initialization=None):
        super().__init__()
        self.initialization_method = weights_initialization
        self.beta = beta
        self.learning_rate = learning_rate
        self.optimizer_name = 'gradient descent with momentum'
        log.info(f"Optimizer Name: {self.optimizer_name}")

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

        # --------------------------------------------------------------------------------- #
        #                       Backpropagation + Momentum                                  #
        # --------------------------------------------------------------------------------- #
        for w in range(0,len(weights)):
            if (len(mega_delta_weights_array)>1):
                # --- Weights
                v_dW = mega_delta_weights_array[-1][w] - mega_delta_weights_array[-2][w]
                v_dW = self.beta* v_dW + (1-self.beta)*dE_dwij_t[w]
                weights[w] -= self.learning_rate * v_dW

                # --- Bias
                if self.use_bias:
                    v_dB = mega_delta_bias_array[-1][w] - mega_delta_bias_array[-2][w]
                    v_dB = self.beta* v_dB + (1-self.beta)*dE_dbij_t[w]
                    bias[w] -= self.learning_rate * v_dB
            else:
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
