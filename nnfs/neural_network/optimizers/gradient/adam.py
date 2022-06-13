from nnfs.neural_network.optimizers.gradient.gradient_optimizer import GradientOptimizer
import numpy as np
from nnfs.utils.logs import create_logger

log = create_logger(__name__)

class Adam(GradientOptimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, weights_initialization=None):
        super().__init__()
        self.initialization_method = weights_initialization
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.optimizer_name = 'Adaptive Moment Estimation (adam) optimizer'
        log.info(f"Optimizer Name: {self.optimizer_name}")

        # Requires a moving average
        self.s_dW = None
        self.s_dB = None

        self.v_dW = None
        self.v_dB = None

        self.eps = 10**(-8)
        # Counting variable for number of iterations
        self.t = 0

    def optimize(self, **kwargs):
        # --- Weights
        dE_dwij_t = kwargs["dE_dwij_t"]
        weights = kwargs["weights"]
        # --- Bias
        if self.use_bias:
            dE_dbij_t = kwargs["dE_dbij_t"]
            bias = kwargs["bias"]

        # Initializing the shape of the weights # TODO: Do list comprehension
        if self.s_dW is None:
            self.s_dW = []
            for i in dE_dwij_t:
                self.s_dW.append(np.zeros(shape=i.shape))
        if self.v_dW is None:
            self.v_dW = []
            for i in dE_dwij_t:
                self.v_dW.append(np.zeros(shape=i.shape))

        if self.use_bias:
            if self.s_dB is None:
                self.s_dB = []
                for b in dE_dbij_t:
                    self.s_dB.append(np.zeros(shape=b.shape))
            if self.v_dB is None:
                self.v_dB = []
                for b in dE_dbij_t:
                    self.v_dB.append(np.zeros(shape=b.shape))

        self.t += 1 # TODO: See what can be done about overflow errors
        # Number of iteration of optimizations
        for w in range(0, len(weights)):
            # --- Weights
            dW = dE_dwij_t[w]

            # Momentum
            self.v_dW[w] = self.beta1*self.v_dW[w] + (1-self.beta1)*dW
            # RMS Prop
            self.s_dW[w] = self.beta2*self.s_dW[w] + (1-self.beta2)*np.power(dW, 2)

            # Handling overflow
            beta1_t = 1-self.beta1**self.t
            # beta1_t = beta1_t if beta1_t>self.eps else self.eps

            beta2_t = 1-self.beta2**self.t
            # beta2_t = beta2_t if beta2_t>self.eps else self.eps

            # Adjusting bias
            # Momentum
            self.v_dW[w] = self.v_dW[w]/beta1_t
            # RMSProp
            self.s_dW[w] = self.s_dW[w]/beta2_t

            # Adjusting weights
            denom = np.sqrt(self.s_dW[w])+self.eps
            weights[w] -= self.learning_rate * np.divide(self.v_dW[w], denom)

            # --- Bias
            if self.use_bias:
                dB = dE_dbij_t[w]

                # Momentum
                self.v_dB[w] = self.beta1*self.v_dB[w] + (1-self.beta1)*dB
                # RMS Prop
                self.s_dB[w] = self.beta2*self.s_dB[w] + (1-self.beta2)*(np.power(dB, 2))

                # Adjusting bias
                # Momentum
                self.v_dB[w] = self.v_dB[w]/beta1_t
                # RMSProp
                self.s_dB[w] = self.s_dB[w]/beta2_t

                denom = np.sqrt(self.s_dB[w])+self.eps
                bias[w] -= self.learning_rate * np.divide(self.v_dB[w], denom)

            log.warn(f"Current index is {w}")
            log.info(f"self.s_dW[w] {self.s_dW[w]}")
            log.info(f"self.s_dW[w] {self.s_dB[w]}")
            log.info(f"self.v_dW[w] {self.v_dW[w]}")
            log.info(f"self.v_dB[w] {self.v_dB[w]}")
            log.critical(f"Beta 1 dash t is {beta1_t}")
            log.critical(f"Beta 2 dash t is {beta2_t}")
            log.info(f"The shape of the RMS prop weights are : {self.s_dW[w].shape}")


        if self.use_bias:
            return weights, bias
        else:
            return weights, 0
