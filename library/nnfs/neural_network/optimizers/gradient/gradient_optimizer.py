from nnfs.propagation.backprop import BackProp
from nnfs.propagation.feedforward import FeedForward
from nnfs.neural_network.optimizers.optimizer import Optimizer

import numpy as np

from nnfs.utils.logs import create_logger

log = create_logger(__name__)


class GradientOptimizer(FeedForward, BackProp, Optimizer):
    """Basis of all gradient based optimizations. This class
    contains initialization and propagation methods that are
    common amongst the different gradient based optimization
    methods.
    """

    def __init__(self):
        self.optimizer_type = "gradient"
        log.info(f"Type of optimizer: {self.optimizer_type}")
        Optimizer.__init__(self)

    def init_network_structures(self, layers):
        """Overrides the init_network_structures function of the Optimizer
        base class. Since it is gradient based only one set of weights
        and/or biases will be initialized and used by the Network and
        optimizer.

        Parameters
        ----------
        layers : list(tuple)
            Network structure containing activation functions.

        Returns
        -------
        weights, biases
            The initialized weights and biases of the network.
        """
        log.info("Initializing Gradient Descent Algorithm Structure")
        self.layers = layers

        num_layers = len(layers)
        weights = []
        if self.use_bias:
            bias = []

        if self.initialization_method is None:
            optimizer_params = dict(name="standard")
        else:
            # Weights initializer
            optimizer_params = self.initialization_method
            log.info(f"The initialization paramters are {optimizer_params}")
            intializer = getattr(Optimizer(), optimizer_params["name"])

        # Initialize random weights and biases (Find different way to init)
        for i in range(1, num_layers):
            optimizer_params["dim0"] = self.layers[i - 1][0]
            optimizer_params["dim1"] = self.layers[i][0]

            if optimizer_params["name"] == "standard":
                # Default
                resulting_init = (
                    np.random.rand(optimizer_params["dim0"], optimizer_params["dim1"])
                    - 0.5
                )
            else:
                resulting_init = intializer(**optimizer_params)

            weights.append(resulting_init)
            if self.use_bias:
                bias.append(
                    np.random.rand(1, self.layers[i][0]) - 0.5,
                )

            log.info(f"Weights Shape: {weights[i-1].shape}")
            if self.use_bias:
                log.info(f"Bias Shape: {bias[i-1].shape}")

        if self.use_bias:
            return weights, bias
        else:
            return weights, 0

    def init_propagations(self):
        # Initializing FeedForward
        FeedForward.__init__(
            self, use_bias=self.use_bias, activation_functions=self.activation_functions
        )

        # Initializing Back Propagation
        BackProp.__init__(
            self,
            use_bias=self.use_bias,
            activation_functions=self.activation_functions,
            error_function=self.error_function,
        )

    def init_measures(self, weights):
        """
        Initialize the measures for each epoch.
        Training Loss/Acc, Validations Loss/Acc, Test Loss/Acc
        """

        train_accuracy_results = 0
        total_training_error = 0

        validation_accuarcy_results = 0
        total_validation_error = 0

        test_accuarcy_results = 0
        total_test_error = 0

        return (
            train_accuracy_results,
            total_training_error,
            validation_accuarcy_results,
            total_validation_error,
            test_accuarcy_results,
            total_test_error,
        )

    def forward_prop_fit(
        self,
        X,
        Y,
        accuracy_results,  # TODO: Make into kwargs
        total_training_error,  # TODO
        weights,
        bias,
    ):
        """Function used for training the gradient based network.

        Parameters
        ----------
        X : numpy.ndarray
            Features
        Y : numpy.ndarray
            Labels
        accuracy_results : int
            State variable to keep track of model accuracy during training
        total_training_error : float
            State variable to keep track of loss during training
        weights : list
            Weights of the network
        bias : list
            Biases of the network

        Returns
        -------
        total_training_error
            State variable to keep track of model accuracy during training
        accuracy_results
            State variable to keep track of loss during training
        ff_results
            Output of the final layer of the network
        """
        weights_val = weights  # Make a copy of the array
        # y_predicted = data_layer[-1]
        if self.use_bias:
            ff_results = self.feedforward(X, weights=weights_val, bias=bias)
        else:
            ff_results = self.feedforward(X, weights=weights_val)

        # Getting output on final layer
        final_layer = ff_results[-1]
        final_layer_output = self.error_function(Y, final_layer)
        total_training_error += final_layer_output

        # --- Training Accuracy
        if np.argmax(final_layer) == np.argmax(Y):
            # Keeping the total correct predictions of classification task
            accuracy_results += 1

        return total_training_error, accuracy_results, ff_results

    def predict(self, X, weights, bias):
        """Predict function uses the feedforward function and only
        returns the final layer output of the network.

        Parameters
        ----------
        X : numpy.ndarray
            Features
        weights : list[numpy.ndarray]
            Weights of the network
        bias : list[numpy.ndarray]
            Bias of the network

        Returns
        -------
        output_layer : numpy.ndarray
            The final layer activations of the network
        """
        if self.use_bias:
            ff_results = self.feedforward(X, weights=weights, bias=bias)
        else:
            ff_results = self.feedforward(X, weights=weights)

        output_layer = ff_results[-1]

        return output_layer

    def backpropagation(self, Y, ff_results, weights, data_layer):
        # Getting loss according to error function
        # Error of all parents?
        delta_weight_arr, delta_bias_arr = self.backprop(
            Y, ff_results, weights, data_layer
        )

        return delta_weight_arr, delta_bias_arr

    def flip_gradient_arr(self, weights_arr, bias_arr):
        """Reverses the order of the weights and bias arrays
        to make backpropagation indexing easier.

        Parameters
        ----------
        weights_arr : list[numpy.ndarray]
            Non-reversed weight array
        bias_arr : list[numpy.ndarray]
            Non-reversed bias array

        Returns
        -------
        dE_dwij_t : list[numpy.ndarray]
            Reversed weight array
        dE_dbij_t : list[numpy.ndarray]
            Reversed bias array

        """
        # Weight Gradient
        dE_dwij_t = weights_arr[::-1]

        # Bias Gradient
        if self.use_bias:
            dE_dbij_t = bias_arr[::-1]
        else:
            dE_dbij_t = 0

        return dE_dwij_t, dE_dbij_t
