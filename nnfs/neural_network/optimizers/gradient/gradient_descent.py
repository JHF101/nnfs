from nnfs.neural_network.optimizers.gradient.gradient_optimizer import GradientOptimizer

from nnfs.utils.logs import create_logger
log = create_logger(__name__)

class GradientDescent(GradientOptimizer):
    def __init__(self, learning_rate, weights_initialization=None):
        super().__init__()
        self.learning_rate=learning_rate
        self.initialization_method = weights_initialization
        self.optimizer_name = 'gradient descent'
        log.info(f"Optimizer Name: {self.optimizer_name}")

    # Maybe seperate here
    def optimize(self, **kwargs):

        dE_dwij_t=kwargs["dE_dwij_t"]
        dE_dbij_t=kwargs["dE_dbij_t"]
        weights=kwargs["weights"]
        bias=kwargs["bias"]

        for w in range(0,len(weights)):
            # normal
            weights[w] -= self.learning_rate * dE_dwij_t[w]

            if self.use_bias:
                bias[w] -= self.learning_rate * dE_dbij_t[w]

        if self.use_bias:
            return weights, bias
        else:
            return weights, 0
