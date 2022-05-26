import numpy as np

# TODO: Consider fully connected and then dropout as well
# I think dropout might just be a multiplication of a mask
class FeedForward:
    def __init__(self, use_bias, activation_functions):
        self.use_bias = use_bias
        self.activation_functions = activation_functions

    # Forward propagation
    def forward_prop_per_layer(self, weight, input_data, bias=0):
        if self.use_bias:
            return np.matmul(input_data, weight) + bias
        else:
            return np.matmul(input_data, weight)
    
    def feedforward(self, x, **kwargs):
        """
        Propagates the result through the entire network, kind of like the predict function
        Kwargs allow us to overwrite the weights 
        """
        # Data layer holds all of the intermediate data between that is 
        # calculated during forward passes and can be used by backward pass
        data_layer = []
        data_layer.append(x)
            
        if self.use_bias:
            if ('weights' not in kwargs.keys()) and ('bias' not in kwargs.keys()):
                print('kwarg items', kwargs.keys())
                raise NotImplementedError("The weights or biases are not being used")

        else: 
            if 'weights' not in kwargs.keys():
                print('kwarg items', kwargs.keys())
                raise NotImplementedError("The weights have not been used")

        # Do a full propagation left to right
        weight_len = len(kwargs['weights'])
        for i in range(weight_len):
            result = self.forward_prop_per_layer(
                weight=kwargs['weights'][i],
                input_data=data_layer[i],
                bias=kwargs['bias'][i] if self.use_bias else 0
                )

            # Apply the activation function and save to data layer
            # We are only interested in the outputs after the activation for back prop
            data_layer.append(
                self.activation_functions[i+1](result)
                )
        
        return data_layer # Returning the data layer output (the output is at [-1])