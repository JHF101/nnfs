import numpy as np
from nnfs.utils.logs import create_logger
log = create_logger(__name__)
class BackProp:
    def __init__(self, use_bias, activation_functions, error_function):
        self.activation_functions = activation_functions
        self.error_function = error_function
        self.use_bias = use_bias

    def back_prop_per_layer(self,
                            output_error,
                            input_data,
                            weights=None,
                            activation_func=None):
        """
        input_data:
            This must be the same as the data entering from the left to the right
        """
        output_error = np.array(output_error)
        input_data = np.array(input_data)

        if (weights is not None):
            """
            Calculating the gradient
            Calculating the deltas of the layer
            """
            weights = np.array(weights)

            input_err = np.matmul(output_error, weights.T)

            #The output of the node before activation wrt the weights: d_E_total/d_w
            weights_delta = np.matmul(input_data.T, output_error)

            return input_err, weights_delta
        else:
            """
            Getting the gradient of the activated layer units by taking the derivative of the activation
            function.
            """
            input_err = activation_func(input_data, derivative=1)
            #The output of the node wrt the output of the node before activation: d_out/d_net
            input_err *= output_error
            return input_err

    def backprop(self, y_true, y_predicted, weights, data_layer):
        # --- Calculating the total error --- #

        # Saved in reverse order
        delta_weight_arr = []

        if self.use_bias:
            delta_bias_arr = []

        # dE/dy
        delta_err = self.error_function(y_true, y_predicted, derivative=1)

        # Variable used to keep track of the state of NN
        count_layer = -1
        data_layer_count = -1

        # --- Full back propagation
        for i in range(0, int(2*len(weights))): # TODO: Check this
            # Even layer
            if (i%2 == 0):
                log.info(f"Iteration number {i}")
                # Derivative of the activation function: dout_y/dnet_y
                delta_err = self.back_prop_per_layer(output_error=delta_err,
                                                    input_data=data_layer[data_layer_count],
                                                    activation_func=self.activation_functions[data_layer_count])
                log.info(f"Data_layer_count : {data_layer_count}")
                log.info(f"Data layer values {i}: {data_layer[data_layer_count]}")
                log.info(f"Activation functions : {self.activation_functions[data_layer_count].__name__}")

                if self.use_bias:
                    delta_bias_arr.append(delta_err)

                data_layer_count -= 1 # Has a different indexing schema

            # Odd layer
            else:
                # Derivatives of dE/dnet_y and dnet_y/dw_ij
                delta_err, delta_weight = self.back_prop_per_layer( output_error=delta_err,
                                                                    input_data=data_layer[data_layer_count],
                                                                    weights=weights[count_layer])

                log.critical(f"Data layer values {i}: {data_layer[data_layer_count]}")
                log.critical(f"Count layer {i}: {count_layer}")
                delta_weight_arr.append(delta_weight)
                count_layer -= 1

        log.info("Done!")
        if self.use_bias:
            return delta_weight_arr, delta_bias_arr
        else:
            return delta_weight_arr, 0
