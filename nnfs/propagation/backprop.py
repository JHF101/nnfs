import numpy as np
from nnfs.utils.logs import create_logger

log = create_logger(__name__)
class BackProp:
    def __init__(self, use_bias, activation_functions, error_function):
        self.activation_functions = activation_functions
        self.error_function = error_function
        self.use_bias = use_bias

    def back_prop_weights(self, output_error, forward_prop_input_data, weights=None, activation_func=None):
        """
        Calculating the gradient of the layer
        """

        weights = np.array(weights)

        delta_err = np.matmul(output_error, weights.T)

        #The output of the node before activation wrt the weights: d_E_total/d_w
        delta_weight = np.matmul(forward_prop_input_data.T, output_error)

        return delta_err, delta_weight

    def back_prop_activations(self, output_error, forward_prop_input_data, activation_func=None):
        """
        Getting the gradient of layer by taking the derivative of the activation
        function.
        """
        """
        forward_prop_input_data:
            Expects the same data entering from the left to the right propagating
            through the network as in the forward pass.
        """
        # Essentially performing the chain rule
        delta_err = activation_func(forward_prop_input_data, derivative=1)
        #The output of the node wrt the output of the node before activation: d_out/d_net
        delta_err *= output_error
        return delta_err

    def backprop(self, y_true, y_predicted, weights, data_layer):
        """Back propagates through the entire network.

        Parameters
        ----------
        y_true : numpy.ndarray
            The labelled output vector
        y_predicted : numpy.ndarray
            The predicted output vector
        weights : list[numpy.ndarray]
            The weights of the entire network
        data_layer : list[numpy.ndarray]
            Memory component keeping track of the neuron activations

        Returns
        -------
        delta_weight_arr : list[numpy.ndarray]
            The weight gradients resulting from back propagation
        delta_bias_arr : list[numpy.ndarray]
            The bias gradients resulting from back propagation.
            (If the use of bias is specified)
        """

        # Saved in reverse order
        delta_weight_arr = []

        if self.use_bias:
            delta_bias_arr = []

        # dE/dy
        delta_err = self.error_function(y_true, y_predicted, derivative=1)

        # Variable used to keep track of the state of NN
        count_layer = -1
        data_layer_count = -1

        # Full back propagation
        for i in range(0, int(2*len(weights))):
            # Even layer
            if (i%2 == 0):
                log.info(f"Iteration number {i}")
                # Derivative of the activation function: dout_y/dnet_y
                delta_err = self.back_prop_activations(output_error=delta_err,
                                                       forward_prop_input_data=data_layer[data_layer_count],
                                                       activation_func=self.activation_functions[data_layer_count])

                if self.use_bias:
                    delta_bias_arr.append(delta_err)

                data_layer_count -= 1 # Has a different indexing schema

            # Odd layer
            else:
                # Derivatives of dE/dnet_y and dnet_y/dw_ij
                delta_err, delta_weight = self.back_prop_weights(output_error=delta_err,
                                                                forward_prop_input_data=data_layer[data_layer_count],
                                                                weights=weights[count_layer])

                delta_weight_arr.append(delta_weight)
                count_layer -= 1

        if self.use_bias:
            return delta_weight_arr, delta_bias_arr
        else:
            return delta_weight_arr, 0
