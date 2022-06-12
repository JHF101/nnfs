import numpy as np
from .initializers import Initializers

from utils.logs import create_logger
log = create_logger(__name__)

# Create a base Optimizer class
class Optimizer(Initializers):
    def __init__(self):
        self.activation_functions = []

    def init_bias_usage(self, use_bias):
        """Initializes the bias usage for the optimizer.

        Parameters
        ----------
        use_bias : boolean
            Boolean which determines if optimizer will use bias
        """
        self.use_bias=use_bias

    def init_error_func(self, error_function):
        """Initializes the type of error function being used.

        Parameters
        ----------
        error_function : function
            Error function
        """
        self.error_function = error_function

    def init_network_structures(self):
        """Base function of optimizer class. Initializes
        the network structures for the network.

        Returns
        -------
        NotImplementedError
        """
        log.exception("init_network_structures() not implemented")
        return NotImplementedError("Please make sure the optimizer is initializing network")

    def init_activation_functions(self, **kwargs):
        """Initializes the optimizers activation functions.

        Returns
        -------
        list[functions]
            Activation functions used by the neural network

        Raises
        ------
        Exception
            Activation functions length != Length of layers
        """
        # Get the activation functions of the layers
        if ('activation_functions' in kwargs.keys()):
            activation_func_len = len(kwargs['activation_functions'])
            layer_len = len(self.layers)

            if (layer_len == activation_func_len):
                # Can change the activation functions
                self.activation_functions = kwargs['activation_functions']

                # Logging the layer length
                for i in range(0, layer_len):
                    log.info(f"Layer {i}'s activation function is: {self.activation_functions[i]}")
            else:
                raise Exception(f'Lengths of activation functions are not the same ({activation_func_len}!={layer_len}).')

        else:
            # Using the activation functions provided by the layers
            # whether it is from the original model or loaded model
            for i in range(0, len(self.layers)):
                self.activation_functions.append(self.layers[i][1])
                log.info(f"Layer {i}'s activation function is: {self.activation_functions[i]}")

        return self.activation_functions

    def init_propagations(self):
        """Base function of optimizer class.
        Initializes propagation methods.
        i.e. FeedForward and Back Propagation.

        Returns
        -------
        NotImplementedError
        """
        log.exception("init_propagations() not implemented")
        return NotImplementedError("Please make sure the propagation methods are being used.")

    def forward_prop_fit(self):
        """Base function for optimizer class.
        Initializes part of fitting function.

        Returns
        -------
        NotImplementedError.
        """
        log.exception("forward_prop_fit() not implemented")
        return NotImplementedError("Please make sure the forward propagation is being used.")

    def optimize(self):
        """Base function for optimizer class.
        Initializes optimization function.

        Returns
        -------
        NotImplementedError
        """
        log.exception("optimize() not implemented")
        return NotImplementedError("Please make sure the optimization is being used.")


