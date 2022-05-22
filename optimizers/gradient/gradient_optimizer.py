from propagation.backprop import BackProp
from propagation.feedforward import FeedForward
from ..optimizer import Optimizer
import numpy as np
import logging


log = logging.getLogger('optimizer')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('program_logs/optimizer.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

class GradientOptimizer(FeedForward, BackProp, Optimizer):
    def __init__(self):
        self.optimizer_type = 'gradient'
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
            The initialized weights of the network.
        """
        log.info("Initializing Gradient Descent Algorithm Structure")
        self.layers = layers

        num_layers = len(layers)
        weights = []
        if self.use_bias:
            bias = []

        if self.initialization_method is None:
            optimizer_params = dict(name='standard')
        else:
            # Weights initializer
            optimizer_params = self.initialization_method
            log.info(f"The initialization paramters are {optimizer_params}")
            # intializer=getattr(Initializers(), optimizer_params['name'])
            intializer=getattr(Optimizer(), optimizer_params['name'])

        # Initialize random weights and biases (Find different way to init)
        for i in range(1, num_layers):
            optimizer_params['dim0'] = self.layers[i-1][0]
            optimizer_params['dim1'] =self.layers[i][0]

            if (optimizer_params['name']=="standard"):
                # Default
                resulting_init = np.random.rand(
                    optimizer_params['dim0'],
                    optimizer_params['dim1'])-0.5
            else:
                resulting_init = intializer(**optimizer_params)

            weights.append(
                resulting_init
                #np.random.rand(self.layers[i-1][0], self.layers[i][0]) - 0.5,
            )
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
        FeedForward.__init__(self,
                                use_bias=self.use_bias, 
                                activation_functions=self.activation_functions
                            )

        # Initializing Back Propagation
        BackProp.__init__(self,
                            use_bias=self.use_bias, 
                            activation_functions=self.activation_functions,
                            error_function=self.error_function
                            )

    def init_measures(self, weights):
        """
        Initialize the measures for epoch
        """

        train_accuracy_results = 0
        total_training_error = 0

        verification_accuarcy_results = 0
        total_verification_error = 0

        test_accuarcy_results = 0
        total_test_error = 0

        return train_accuracy_results, total_training_error, \
                verification_accuarcy_results, total_verification_error, \
                    test_accuarcy_results, total_test_error

    def forward_prop_fit(self, 
                        X, 
                        Y, 
                        accuracy_results, # TODO: Make into kwargs
                        total_training_error, # TODO
                        weights, bias):

        weights_val = weights # Make a copy of the array
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
        if (np.argmax(final_layer) == np.argmax(Y)):
            accuracy_results += 1 # Used for genetic algorithm

        return total_training_error, accuracy_results, ff_results

    def predict(self, X, weights, bias):

        if self.use_bias:
            ff_results = self.feedforward(X, weights=weights, bias=bias)
        else:
            ff_results = self.feedforward(X, weights=weights)

        output_layer = ff_results[-1]

        return output_layer

    def backpropagation(self, Y, ff_results, weights, data_layer):
        # Getting loss according to error function
        # Error of all parents?
        delta_weight_arr, delta_bias_arr = self.backprop(Y, ff_results, weights, data_layer=data_layer)

        return delta_weight_arr, delta_bias_arr

    def flip_weights(self, average_gradients, average_bias):
        # --------------------------------------------------------------------------- #
        #                   Saving weights and bias gradients                         #
        # --------------------------------------------------------------------------- #
        # --- Reversing the gradients
        dE_dwij_t = average_gradients[::-1] # Weight Gradient
        
        if self.use_bias:
            dE_dbij_t = average_bias[::-1] # Bias Gradient
        else:
            dE_dbij_t = 0

        return dE_dwij_t, dE_dbij_t