import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from activation_functions import tanh, sigmoid, relu, softmax
from error_functions import rms
import pickle

from utils import shuffle_arrays
import keyboard

import seaborn as sn
import pandas as pd

from matplotlib.pyplot import figure

figure(figsize=(16, 9), dpi=80)
plt.style.use('fivethirtyeight')

class Network:
    """
    Neural Network class that uses fully connected layers
    """
    def __init__(self, 
                layers=[], 
                error_function=None, 
                learning_rate=0.5, 
                bias=True, 
                optimizer='backprop',
                additional_optimizer_params={},
                training_params={}):
        """_summary_

        Parameters
        ----------
        layers : tuple
            (Number of nodes in a layer, activation function)
            >>> (6, activation_function)
        learning_rate : float
            Learning rate used in the model
        bias : bool, optional
            Determines whether to add bias nodes on each layer, except the 
            output layer, by default True
        plotting_func_bool : bool, optional
            Determines whether to plot, by default False

        """

        self.learning_rate = learning_rate # TODO: Make Learning Rate Scheduling
        self.optimizer = optimizer
        # Take a tuple of number of neurons in a layer and the activation
        # Get the number of layers
        # Number of weights is number of layers - 1
        self.num_layers = len(layers)
        self.weights = []
        self.bias = []
        # Initialize random weights and biases
        for i in range(1,self.num_layers):
            self.weights.append(
                np.random.rand(layers[i-1][0], layers[i][0]) - 0.5,
                # np.random.uniform(low=-0.5, high=0.5, size=(layers[i-1][0], layers[i][0])), # Has a big effect on rprop

            )
            if bias:
                self.bias.append(
                    np.random.rand(1, layers[i][0]) - 0.5,
                    # np.random.uniform(low=-0.5, high=0.5, size=(1, layers[i][0]))

                )
            else:
                self.bias.append(
                    np.zeros((1,layers[i][0]))
                )
            print("Weights Shape", self.weights[i-1].shape)
            print("Bias Shape", self.bias[i-1].shape)

        # Get the activation functions of the layers
        self.activation_functions = []
        for i in range(0, self.num_layers):
            self.activation_functions.append(layers[i][1])
            print(f"Layer {i}'s activation function is:", self.activation_functions[i])

        # Get the loss function
        self.error_function = error_function

        # Save these plots
        self.epoch_error_training_plot = [] 
        self.epoch_error_validation_plot = [] 
        self.epoch_error_testing_plot = []
        self.epoch_training_accuracy_plot = []
        self.epoch_validation_accuracy_plot = []
        self.epoch_testing_accuracy_plot = []

        # Save all weights into a super mega array
        self.mega_weights_array=[]
        self.mega_bias_array = []

        self.mega_delta_weights_array = []
        self.mega_delta_bias_array = []

        if self.optimizer=="rprop":
            if additional_optimizer_params == {}:
                raise Exception("RPROP Requires you to have additional parameters specified")
            self.additional_parameters = additional_optimizer_params

        # Used to stop training
        self.training_params = training_params
        
        # Overfit parameter
        self.GL = 0
        if self.training_params != {}: # TODO: Maybe make this into a class that can be passed in that you can access via class.name etc
            # K Epochs
            self.k_epochs = self.training_params['k_epochs']
            self.pkt_threshold = self.training_params['pkt_threshold']
            # Early stopping
            self.alpha = self.training_params['alpha'] 
            # The point at which training was stopped 
            self.optimal_error_position = 0

    def get_model_name(self):
        dimensions = "-" + str(self.weights[0].shape[0]) + "-" 
        for i in range(0,len(self.weights)):
            dimensions += str(self.weights[i].shape[1]) + "-" + str(self.activation_functions[i+1].__name__) + "-" 
        name_str = self.optimizer + dimensions + str(self.learning_rate)
        return name_str 
    
    # Forward propagation
    def forward_prop_per_layer(self, weight, input_data, bias):
        # print("In forward propagation")
        return np.matmul(input_data, weight) + bias
    
    def feedforward(self, x):
        """
        Propagates the result through the entire network, kind of like the predict function
        """
        # Data layer holds all of the intermediate data between that is 
        # calculated during forward passes and can be used by backward pass
        self.data_layer = []
        self.data_layer.append(x)

        # Do a full propagation left to right
        for i in range(len(self.weights)):
            result = self.forward_prop_per_layer(
                weight=self.weights[i],
                input_data=self.data_layer[i],
                bias=self.bias[i] 
                )

            # Apply the activation function and save to data layer
            # We are only interested in the outputs after the activation for back prop
            self.data_layer.append(
                self.activation_functions[i+1](result)
                )

        return self.data_layer[-1] # Returning the output

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

    def backprop(self, y_true, y_predicted):
        # --- Calculating the total error --- #
        final_layer_output = self.error_function(y_true, y_predicted)
        
        self.delta_weight = [] # Saved in reverse order
        self.delta_bias = []
        # dE/dy
        delta_err = self.error_function(y_true, y_predicted, derivative=1)

        # Variable used to keep track of the state of NN
        count_layer = -1
        data_layer_count = -1

        delta_err_arr = []
        # --- Full back propagation 
        for i in range(0, int(2*len(self.weights))): # TODO: Check this
            # Even layer
            if i%2 == 0:
                # Derivative of the activation function: dout_y/dnet_y
                delta_err = self.back_prop_per_layer(output_error=delta_err, 
                                                    input_data=self.data_layer[data_layer_count], 
                                                    activation_func=self.activation_functions[data_layer_count])

                self.delta_bias.append(delta_err) # Bias lies on a layer not with a weight?
                data_layer_count -= 1 # Has a different indexing schema

            # Odd layer
            else:
                # Derivatives of dE/dnet_y and dnet_y/dw_ij
                delta_err, delta_weight = self.back_prop_per_layer( output_error=delta_err, 
                                                                    input_data=self.data_layer[data_layer_count], 
                                                                    weights=self.weights[count_layer])
                self.delta_weight.append(delta_weight)        
                count_layer -= 1

        return final_layer_output

    def fit(self, 
            x_train, y_train, 
            x_test, y_test, 
            x_validate=None, y_validate=None, 
            epochs=1, 
            validations_set_percent=0.0, 
            batch_size=1, 
            shuffle_training_data=True):
        
        self.optimal_error_position = epochs-1
        self.batch_size = batch_size
        self.data_shuffle = shuffle_training_data
        self.total_epochs = epochs
        
        if (self.optimizer == 'delta_bar_delta'):
            dbd_init_process = True

        if (self.optimizer == 'rprop'):
            rprop_init_process = True
            delta_max = self.additional_parameters['delta_max']
            delta_min = self.additional_parameters['delta_min']
            eta_plus = self.additional_parameters['eta_plus']
            eta_minus = self.additional_parameters['eta_minus']

            del_ij_t_w = []
            del_ij_t_b = []
            # Initialize to zero
            dE_dwij_t_1 = []
            dE_dbij_t_1 = []
            train_err_arr = [] # RProp+
                
        x_train_size = len(x_train)

        training_set_size = 0

        if (x_validate is not None) and (y_validate is not None):
            # Determing the sets were actually given
            x_validation_set = x_validate
            y_validation_set = y_validate
            training_set_size = x_train_size 
            validations_set_size = len(x_validate) 
        else:
            # If the validation sets are not given, we can make our own
            if (1.0>validations_set_percent>0.0):
                training_set_size = int(x_train_size*(1-validations_set_percent))
                validations_set_size = x_train_size - training_set_size
                x_validation_set = x_train[training_set_size:training_set_size+validations_set_size]
                y_validation_set = y_train[training_set_size:training_set_size+validations_set_size]  
            elif (validations_set_percent==0.0):
                training_set_size=x_train_size
                print("No validation set will be used")
            else:
                raise Exception("Validation set size cannot be greater or less than 1")
        
        x_train = x_train[0:training_set_size]
        y_train = y_train[0:training_set_size]

        # Note: Randomize the order possibly
        example_size = training_set_size
        # example_size = training_set_size-1 # N-1
        total_training_error = 0
        total_validation_error = 0
        total_test_error = 0 

        # Train
        for i in range(epochs):
            
            # ------------------------------------------------------------------ #
            #                              Training                              #
            # ------------------------------------------------------------------ #
            # Randomizing the data order
            if (shuffle_training_data==True):
                x_train, y_train = shuffle_arrays(x_train, y_train)

            count_correct_training = 0
            
            # Batch size is the number of samples processed before the model is updated
                # if batch size = 1 then we have online learning
            if batch_size > 0:
                batches = int(np.ceil(example_size/batch_size)) # Total number of iterations
                print("Number of batches", batches)
                for outer in range(0, batches):
                    # Stop Training immediately
                    if keyboard.is_pressed('q'):
                        raise Exception("Run stopped")

                    delta_weight_counter = 0 # Used as a state control variable
                    num_of_summations_per_batch = 0 # Keeps track of the number of gradients

                    lower_batch = int(outer*batch_size)
                    upper_batch = int((outer+1)*batch_size)
                    for j in range(lower_batch, upper_batch):
                        # Stop Training immediately
                        if keyboard.is_pressed('q'):
                            raise Exception("Run stopped")
                        # Condition to break out of the training loop
                        if j >= example_size:
                            break
                        # --- Forward Propagation
                        ff_result = self.feedforward(x_train[j]) 
                        # --- Training Accuracy 
                        if np.argmax(ff_result) == np.argmax(y_train[j]):
                            count_correct_training += 1

                        if (self.optimizer == 'backprop') or (self.optimizer == "backprop+momentum") or \
                            (self.optimizer == 'rprop') or (self.optimizer=="delta_bar_delta"):
                            # --- Backward Propagation
                            total_training_error += self.backprop(y_train[j], ff_result)

                            # --- Average Weights
                            if (delta_weight_counter==0):
                                average_gradients = self.delta_weight
                                average_bias = self.delta_bias
                            if (delta_weight_counter>0):
                                for b in range(len(self.delta_weight)):
                                    average_gradients[b] += self.delta_weight[b]
                                    average_bias[b] += self.delta_bias[b]
                                
                        delta_weight_counter += 1
                        num_of_summations_per_batch +=1
                    if len(average_gradients) != len(self.delta_weight):
                        raise Exception("The length of the gradient summation is not the same")
                    # Update weights after batch 
                    # This average term has a different effect on the output    
                    average_gradients = [average_gradients[i]/num_of_summations_per_batch for i in range(len(average_gradients))]
                    average_bias = [average_bias[i]/num_of_summations_per_batch for i in range(len(average_bias))]

                    # --------------------------------------------------------------------------- #
                    #                   Saving weights and bias gradients                         #
                    # --------------------------------------------------------------------------- #
                    # --- Reversing the gradients
                    dE_dwij_t = average_gradients[::-1] # Weight Gradient
                    dE_dbij_t = average_bias[::-1] # Bias Gradient

                    # NOTE: You have reversed the order in which the gradients are saved !!!
                    self.mega_delta_weights_array.append(dE_dwij_t)
                    self.mega_delta_bias_array.append(dE_dbij_t)
                    # To improve efficiency
                    if len(self.mega_delta_weights_array)>5:
                        self.mega_delta_weights_array.pop(0)
                    if len(self.mega_delta_bias_array)>5:
                        self.mega_delta_bias_array.pop(0)

                    if (self.optimizer == 'rprop'):
                        # --------------------------------------------------------------------------------- #
                        #                                       RPROP                                       #
                        # --------------------------------------------------------------------------------- #
                        # TODO: Optimization can be done with numpy.where

                        if rprop_init_process == True:
                            # ---- Initialization ---- #
                            for w in range(0,len(self.weights)):
                                gradient_multiplication = dE_dwij_t[w]
                                bias_multiplication = dE_dbij_t[w]

                                del_ij_t_w.append(np.random.uniform(low=0.05, high=0.2, size=gradient_multiplication.shape))
                                del_ij_t_b.append(np.random.uniform(low=0.05, high=0.2, size=bias_multiplication.shape))

                                dE_dwij_t_1.append(np.random.uniform(low=0.05, high=0.2, size=gradient_multiplication.shape))   
                                dE_dbij_t_1.append(np.random.uniform(low=0.05, high=0.2, size=bias_multiplication.shape))  

                        rprop_init_process = False

                        if len(dE_dwij_t) != len(dE_dwij_t_1):
                            raise Exception("The size of the derivatives in RProp do not match")
                        if len(dE_dwij_t) != len(dE_dbij_t):
                            raise Exception("Length of bias graidents does not equal the length of the weights gradients")

                        for outer in range(0, len(dE_dwij_t)): # Looping through the first set of weights
                            
                            if (dE_dwij_t[outer].shape[0] != dE_dwij_t_1[outer].shape[0]):
                                raise Exception("The element size of derivatives in RProp do not match")

                            for p in range(0, dE_dwij_t[outer].shape[0]):

                                if (dE_dwij_t[outer].shape[1] != dE_dwij_t_1[outer].shape[1]):
                                    raise Exception("The row size of derivatives in RProp do not match")
                                    
                                for q in range(0, dE_dwij_t[outer].shape[1]):

                                    # (dE(t)/dw_i(t))*(dE(t-1)/dw_i(t-1)) 
                                    gradient_mult = dE_dwij_t_1[outer][p][q] * dE_dwij_t[outer][p][q]
                                    # print("Index", outer, p, q)
                                    if (gradient_mult > 0.0): # This loop only runs once
                                        del_ij_t_w[outer][p][q] = np.minimum(del_ij_t_w[outer][p][q] * eta_plus, delta_max)
                                        del_w_ij_t = -1.0 * np.sign(dE_dwij_t[outer][p][q]) * del_ij_t_w[outer][p][q]
                                        self.weights[outer][p][q] = self.weights[outer][p][q] + del_w_ij_t
                                        dE_dwij_t_1[outer][p][q] = dE_dwij_t[outer][p][q]

                                    elif (gradient_mult < 0.0):
                                        del_ij_t_w[outer][p][q] = np.maximum(del_ij_t_w[outer][p][q] * eta_minus, delta_min) # Error was delta min was huge
                                        dE_dwij_t_1[outer][p][q] = 0.0
                                        # if train_err_arr[-1]>train_err_arr[-2]: #Modification for RPROP+
                                        #     self.weights[outer][p][q] -= self.learning_rate*dE_dwij_t[outer][p][q]

                                    elif (gradient_mult == 0.0):
                                        del_w_ij_t = -1.0 * np.sign(dE_dwij_t[outer][p][q]) * del_ij_t_w[outer][p][q]
                                        self.weights[outer][p][q] = self.weights[outer][p][q] + del_w_ij_t
                                        dE_dwij_t_1[outer][p][q] = dE_dwij_t[outer][p][q]
                                    else:
                                        raise Exception("There exists no value you are looking for")

                            # Assuming bias and weights arrats are the same length
                            for s in range(0, dE_dbij_t[outer].shape[0]):
                                for r in range(0, dE_dbij_t[outer].shape[1]):

                                    bias_grad_mult = dE_dbij_t_1[outer][s][r] * dE_dbij_t[outer][s][r]

                                    if ( bias_grad_mult > 0.0): # This loop only runs once

                                        del_ij_t_b[outer][s][r] = np.minimum(del_ij_t_b[outer][s][r] * eta_plus, delta_max)
                                        del_b_ij_t = -1.0 * np.sign(dE_dbij_t[outer][s][r]) * del_ij_t_b[outer][s][r]
                                        self.bias[outer][s][r] = self.bias[outer][s][r] + del_b_ij_t
                                        dE_dbij_t_1[outer][s][r] = dE_dbij_t[outer][s][r]

                                    elif (bias_grad_mult < 0.0):
                                        del_ij_t_b[outer][s][r] = np.maximum(del_ij_t_b[outer][s][r] * eta_minus, delta_min) # Error was delta min was huge
                                        dE_dbij_t_1[outer][s][r] = 0.0
                                        # if train_err_arr[-1]>train_err_arr[-2]: #Modification for RPROP+
                                        #     self.bias[outer][s][r] -=  self.learning_rate*dE_dbij_t[outer][s][r]

                                    elif (bias_grad_mult == 0.0):
                                        del_b_ij_t = -1.0 * np.sign(dE_dbij_t[outer][s][r]) * del_ij_t_b[outer][s][r]
                                        self.bias[outer][s][r] = self.bias[outer][s][r] + del_b_ij_t
                                        dE_dbij_t_1[outer][s][r] = dE_dbij_t[outer][s][r]
                                        # print(self.weights[outer][s][q])
                                    else:
                                        raise Exception("There exists no value you are looking for")

                        # --------------------------------------------------------------------------------- #
                        #                                   END RPROP                                       #
                        # --------------------------------------------------------------------------------- #
                        
                    if (self.optimizer == 'delta_bar_delta'):
                        # --------------------------------------------------------------------------------- #
                        #                                 DELTA-BAR-DELTA                                   #
                        # --------------------------------------------------------------------------------- #
                        
                        if dbd_init_process == True:
                            theta = 0.1 #[0,1]
                            mini_k = 0.01 # Constant coefficient increment factor
                            phi = 0.1 # [0,1]- Constant learning coefficient decrement factor

                            learning_rates_weights = [np.random.uniform(low=0.05, high=0.2, size=dE_dwij_t[i].shape) for i in range(len(dE_dwij_t))]
                            learning_rates_weights_deltas = [np.random.uniform(low=0.05, high=0.2, size=dE_dwij_t[i].shape) for i in range(len(dE_dwij_t))]
                            average_bar_delta_weights = [np.zeros(dE_dwij_t[i].shape) for i in range(len(dE_dbij_t))]

                            learning_rates_bias = [np.random.uniform(low=0.05, high=0.2, size=dE_dbij_t[i].shape) for i in range(len(dE_dbij_t))]
                            learning_rates_bias_deltas = [np.random.uniform(low=0.05, high=0.2, size=dE_dbij_t[i].shape) for i in range(len(dE_dbij_t))]
                            average_bar_delta_bias =[np.zeros(dE_dbij_t[i].shape) for i in range(len(dE_dbij_t))]
                            
                            dbd_init_process = False

                        for dw in range(len(dE_dwij_t)): # TODO: Check if this can be expanded to RPROP
                            if len(self.mega_delta_weights_array)>1:
                                # delta_bar(t) =  (1-theta)*gradient_of_weights + theta*previous_gradent of weights  
                                average_bar_delta_weights[dw]=(1-theta)*dE_dwij_t[dw] + theta * self.mega_delta_weights_array[-2][dw]#theta*average_bar_delta_weights[dw]# theta * self.mega_delta_weights_array[-2][dw] #theta*average_bar_delta_weights[dw] ##
                            else:
                                average_bar_delta_weights[dw]=(1-theta)*dE_dwij_t[dw]
                            

                            # The muliplication of delta(t) * delta(t-1)
                            condition_param = average_bar_delta_weights[dw] * dE_dwij_t[dw]

                            # Looping per layer weight
                            # Else condition
                            learning_rates_weights_deltas[dw] = np.where(condition_param==0, 0.0, learning_rates_weights_deltas[dw])
                            # Greater than condition
                            learning_rates_weights_deltas[dw] = np.where(condition_param>0, mini_k, learning_rates_weights_deltas[dw])
                            # Less that condition
                            learning_rates_weights_deltas[dw] = np.where(condition_param<0, -phi*learning_rates_weights[dw], learning_rates_weights_deltas[dw])

                            # Adding the learning deltas to the learning rate arrays
                            learning_rates_weights[dw] += learning_rates_weights_deltas[dw]
                            
                            # Adjusting the weights
                            self.weights[dw] = self.weights[dw] - learning_rates_weights[dw] *dE_dwij_t[dw]


                        for db in range(len(dE_dbij_t)):
                            # Oscillates a bit more when here, when at the end it is smoother just does now work as well
                            if len(self.mega_delta_bias_array)>1: 
                                average_bar_delta_bias[db]=(1-theta)*dE_dbij_t[db] + theta*self.mega_delta_bias_array[-2][db]#theta*average_bar_delta_bias[db] #theta*self.mega_delta_bias_array[-1][db]
                            else:
                                average_bar_delta_bias[db]=(1-theta)*dE_dbij_t[db]

                            # print(average_bar_delta_bias)
                            bias_condition_param = average_bar_delta_bias[db] * dE_dbij_t[db]

                            # First condition 
                            learning_rates_bias_deltas[db] = np.where(bias_condition_param==0, 0.0, learning_rates_bias_deltas[db])
                            learning_rates_bias_deltas[db] = np.where(bias_condition_param>0, mini_k, learning_rates_bias_deltas[db])
                            learning_rates_bias_deltas[db] = np.where(bias_condition_param<0, -phi*learning_rates_bias[db], learning_rates_bias_deltas[db])
                            
                            learning_rates_bias[db] += learning_rates_bias_deltas[db]
                            
                            # This is correct
                            self.bias[db] = self.bias[db] - learning_rates_bias[db] * dE_dbij_t[db]

                        # --------------------------------------------------------------------------------- #
                        #                                   END DELTA-BAR-DELTA                             #
                        # --------------------------------------------------------------------------------- #

                    if (self.optimizer == 'backprop+momentum'):

                        # --- Adjust Weights -- TODO: Consider reversing the delta arrays
                        for w in range(0,len(self.weights)):
                            # --------------------------------------------------------------------------------- #
                            #                       Backpropagation + Momentum                                  #
                            # --------------------------------------------------------------------------------- #
                            # momentum
                            if (len(self.mega_delta_bias_array)>1):
                                # print("In bias")
                                v_dW = self.mega_delta_weights_array[-1][w] - self.mega_delta_weights_array[-2][w]
                                v_dB = self.mega_delta_bias_array[-1][w] - self.mega_delta_bias_array[-2][w]

                                beta = 0.9

                                v_dW = beta* v_dW + (1-beta)*dE_dwij_t[w]
                                v_dB = beta* v_dB + (1-beta)*dE_dbij_t[w]

                                self.weights[w] -= self.learning_rate * v_dW
                                self.bias[w] -= self.learning_rate * v_dB
                            else:
                                # normal
                                self.weights[w] -= self.learning_rate * dE_dwij_t[w]
                                self.bias[w] -= self.learning_rate * dE_dbij_t[w]

                            # --------------------------------------------------------------------------------- #
                            #                                   Backpropagation                                 #
                            # --------------------------------------------------------------------------------- #
                    if (self.optimizer == 'backprop'):
                        for w in range(0,len(self.weights)):
                            # normal
                            self.weights[w] -= self.learning_rate * dE_dwij_t[w]
                            self.bias[w] -= self.learning_rate * dE_dbij_t[w]

                # Averaging over all samples 
                total_training_error /= example_size
                training_accuracy = count_correct_training / example_size

            # ------------------------------------------------------------------ #
            #                              Validating                            #
            # ------------------------------------------------------------------ #
            count_correct_validation = 0
            try:
                # Evaluate the performance set
                if (validations_set_percent>0.0) or (len(x_validate)>0):
                    for v in range(0,validations_set_size):
                        ff_validation_result = self.feedforward(x_validation_set[v])
                        # Validation accuracy
                        if np.argmax(ff_validation_result) == np.argmax(y_validation_set[v]):
                            count_correct_validation += 1
                        # Validations Loss
                        final_layer_output_error_validation = self.error_function(y_validation_set[v], ff_validation_result)
                        total_validation_error += final_layer_output_error_validation
                    # Averaging over all samples 
                    total_validation_error /= validations_set_size
                    validation_accuracy = count_correct_validation / validations_set_size
            except:
                print("No validation set")
            # ------------------------------------------------------------------ #
            #                              Testing                               #
            # ------------------------------------------------------------------ #
            y_test_len = len(y_test)

            count_correct_test = 0
            total_test_error = 0
            all_predictions = []
            actual_values=[]
            for t in range(y_test_len):
                ff_test_result = self.predict(x_test[t])
                # Testing accuracy
                categorical_prediction = np.argmax(ff_test_result)
                all_predictions.append(categorical_prediction)
                actual_output= np.argmax(y_test[t])
                actual_values.append(actual_output)

                if (categorical_prediction == actual_output):
                    count_correct_test += 1
                    
                # Testing Loss
                total_test_error += self.error_function(y_test[t], ff_test_result)
            total_test_error /= y_test_len
            test_accuracy = count_correct_test / y_test_len

            # ---- Confusion Matrix
            actual_values = np.array(actual_values)
            all_predictions = np.array(all_predictions)
            self.confusion_matrix = confusion_matrix(actual_values, all_predictions)

            if count_correct_validation > 0: # Means that we have more 
                print(f"""epoch {i+1}/{epochs} train_loss = {total_training_error} - train_accuracy = {training_accuracy} | val_loss = {total_validation_error} - val_accuracy = {validation_accuracy} | test_loss={total_test_error} - test_accuracy = {test_accuracy} \r""")
            else:
                print(f"""epoch {i+1}/{epochs} train_loss = {total_training_error} - train_accuracy = {training_accuracy} | test_loss={total_test_error} - test_accuracy = {test_accuracy} \r""")

            # ------------------------------------------------------------------ #
            #                         Saving Model                               #
            # ------------------------------------------------------------------ #
            self.epoch_error_training_plot.append(total_training_error)
            self.epoch_training_accuracy_plot.append(training_accuracy)

            try:
                if (validations_set_percent>0.0) or (len(x_validate)>0):    
                    self.epoch_error_validation_plot.append(total_validation_error)
                    self.epoch_validation_accuracy_plot.append(validation_accuracy)
            except:
                print("No validation set")

            self.epoch_error_testing_plot.append(total_test_error)
            self.epoch_testing_accuracy_plot.append(test_accuracy)
            # Contains all of the weights for training
            self.mega_weights_array.append(self.weights)
            self.mega_bias_array.append(self.bias)

            try:
                if (self.early_stopping() == True):
                    print("Early stopping stopped training")
                    # Going back and getting where validation is a minimum
                    # and these are the values used in the table

                    self.total_epochs = i + 1 # NOTE: actually has an offset of 1
                    self.optimal_error_position = i-np.argmin(self.epoch_error_validation_plot[::-1][0:self.k_epochs]) + 1
                    # print(self.epoch_error_validation_plot)
                    # print("Epoch",i)
                    # print("Argmin",np.argmin(self.epoch_error_validation_plot[::-1][0:self.k_epochs]))
                    # print("Optimal Error Position", self.optimal_error_position)
                    break
                if (self.training_progress()==True):
                    print("P_k(t) stopped training")
                    self.total_epochs = i + 1 # NOTE: actually has an offset of 1
                    # Going back and getting where validation is a minimum
                    # and these are the values used in the table
                    self.optimal_error_position = i-np.argmin(self.epoch_error_validation_plot[::-1][0:self.k_epochs]) + 1
                    # print(self.epoch_error_validation_plot)
                    # print("Epoch",i)
                    # print("Argmin",np.argmin(self.epoch_error_validation_plot[::-1][0:self.k_epochs]))
                    # print("Optimal Error Position", self.optimal_error_position)
                    break
            except:
                pass
        print("Done Training Model")        

    # Alias for feedforward
    def predict(self, X):
        return self.feedforward(X)

    def training_progress(self):
        # TODO: Make sure that the optimal weight is saved at the minimum
        k_epochs = self.k_epochs
        try:            
            numerator = 0
            denominator = 0
            denom_check_arr= []
            for i in range(1,k_epochs+1):
                store = self.epoch_error_training_plot[-i]
                numerator += store
                denom_check_arr.append(store)

            denominator = k_epochs*np.min(denom_check_arr)

            P_k_t = 1000*((numerator/denominator)-1)
            print("P_k_t", P_k_t)
            if (P_k_t < self.pkt_threshold):
                return True
            else:
                return False
        except:
            print("Training has not been for long enough")
            return False

    def early_stopping(self):
        if self.training_params !={}:
            # page 24
            # Call this function in the loop you want it to stop

            E_opt = np.min(self.epoch_error_validation_plot)

            # Generalization Loss
            self.GL = 100 * ((self.epoch_error_validation_plot[-1]/E_opt) - 1)
            if (self.alpha < self.GL):
                return True
            else:
                return False
        else: 
            return False

    def save_model(self, file_dir):
        """
        Save the weights of the model to a npy which can then be used to by the model
        again
        """
        try:
            print("Saved weights successfully")
            
            with open(f'{file_dir}', 'wb') as f:
                pickle.dump([self.mega_weights_array, 
                            self.mega_bias_array, 
                            self.activation_functions,
                            self.learning_rate,
                            self.num_layers,
                            self.epoch_error_testing_plot,
                            self.epoch_error_training_plot,
                            self.epoch_error_validation_plot,
                            self.epoch_testing_accuracy_plot,
                            self.epoch_training_accuracy_plot,
                            self.epoch_validation_accuracy_plot,
                            self.optimal_error_position,
                            self.GL,
                            self.batch_size,
                            self.data_shuffle,
                            self.total_epochs,
                            ], f)

        except:
            raise Exception("Weights were not saved correctly")

    def load_model(self, file_dir, epoch_selection=0):
        """
        Load the weights to be used by the model.
        Epoch selection allows the user to choose the set of weights
        that they would like to use for prediction 
        """
        # TODO: Add GL
        if (epoch_selection==0):
            print("Please select an epoch")
        # TODO: Add try to loads for these as older models might be missing them
        self.mega_weights_array, 
        self.mega_bias_array, 
        with open(f'{file_dir}', 'rb') as f:
            mega_weights_array, mega_bias_array, activations, learning_rate, \
            num_layers, epoch_error_testing_plot, epoch_error_training_plot, \
            epoch_error_validation_plot, epoch_testing_accuracy_plot, \
            epoch_training_accuracy_plot, epoch_validation_accuracy_plot, \
            optimal_error_position, overfit, batch_size, data_shuffle, total_epochs = pickle.load(f)
        
        self.mega_weights_array = mega_weights_array
        self.mega_bias_array = mega_bias_array
        self.weights = mega_weights_array[epoch_selection]
        self.bias = mega_bias_array[epoch_selection]
        self.activation_functions = activations
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.epoch_error_testing_plot = epoch_error_testing_plot
        self.epoch_error_training_plot = epoch_error_training_plot
        self.epoch_error_validation_plot = epoch_error_validation_plot
        self.epoch_testing_accuracy_plot = epoch_testing_accuracy_plot
        self.epoch_training_accuracy_plot = epoch_training_accuracy_plot
        self.epoch_validation_accuracy_plot = epoch_validation_accuracy_plot
        self.optimal_error_position = optimal_error_position
        self.GL = overfit
        self.batch_size = batch_size
        self.data_shuffle = data_shuffle
        self.total_epochs = total_epochs

    def plot_epoch_error(self, ds_name, save_dir):
        # Naming
        figure(figsize=(16, 9), dpi=80)
        architecture = ''
        for i in range(0,len(self.weights)):                    
                                                                # Activation function 
            architecture += str(self.weights[i].shape[1]) + "-" + str(self.activation_functions[i+1].__name__)[0] + "+" 
        architecture = architecture[:-1] # Remove to the last plus
        architecture += " " + self.error_function.__name__  
        if (self.learning_rate>0):
            architecture += " " + "lr: " + str(self.learning_rate)
        # Creating indeces
        epochs_idx = [i+1 for i in range(len(self.epoch_error_testing_plot))]

        plt.plot(epochs_idx, self.epoch_error_testing_plot, 'b', label="Testing")

        plt.plot(epochs_idx, self.epoch_error_training_plot, 'r', label="Training")
        
        if (len(self.epoch_error_validation_plot) > 0):
            plt.plot(epochs_idx, self.epoch_error_validation_plot, 'g', label="Validation")
            plt.plot(self.optimal_error_position, self.epoch_error_validation_plot[self.optimal_error_position-1], 'ko', label="Min validation error", markersize=12,linewidth=2)
        
        plt.title(ds_name+' '+str.capitalize(self.optimizer)+' '+architecture)
        plt.ylabel("Error")
        plt.ylim(0,1)
        plt.xlabel("Epochs")
        plt.legend()
        plt.grid()
        if (save_dir is not None):
            plt.style.use('fivethirtyeight')
            plt.savefig(save_dir)
        plt.clf()
        # plt.close()

    def plot_epoch_accuracy(self, ds_name, save_dir): 
        # Naming
        figure(figsize=(16, 9), dpi=80)
        architecture = ''
        for i in range(0,len(self.weights)):                    
                                                                # Activation function 
            architecture += str(self.weights[i].shape[1]) + "-" + str(self.activation_functions[i+1].__name__)[0] + "+" 
        architecture = architecture[:-1] # Remove to the last plus    
        architecture += " "+self.error_function.__name__   
        if (self.learning_rate>0):
            architecture += " " + "lr: " + str(self.learning_rate)
        epochs_idx = [i+1 for i in range(len(self.epoch_error_testing_plot))]

        plt.plot(epochs_idx, self.epoch_testing_accuracy_plot,'b', label="Testing")
        
        plt.plot(epochs_idx, self.epoch_training_accuracy_plot, 'r', label="Training")
        
        if (len(self.epoch_validation_accuracy_plot)>0):
            plt.plot(epochs_idx, self.epoch_validation_accuracy_plot, 'g', label="Validation")
            plt.plot(self.optimal_error_position, self.epoch_testing_accuracy_plot[self.optimal_error_position-1], 'ko', label="Min validation error", markersize=12,linewidth=2)
        
        plt.title(ds_name+' '+str.capitalize(self.optimizer)+' '+architecture)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.ylim(0,1)
        plt.legend()
        plt.grid()
        if (save_dir is not None):
            plt.style.use('fivethirtyeight')
            plt.savefig(save_dir)
        plt.clf()
        # plt.close()

    def plot_confusion_matrix(self, save_dir):
        figure(figsize=(16, 9), dpi=80)
        df = pd.DataFrame(self.confusion_matrix)
        # sn.set(font_scale=1.4)
        sn.heatmap(df, annot=True, annot_kws={"size": 16}, fmt='g') 
        plt.xlabel("Predictions")
        plt.ylabel("Actual Values")
        # plt.savefig('B7_confusion_matrix.png')
        if (save_dir is not None):
            # plt.style.use('seaborn')
            plt.savefig(save_dir)
        plt.cla()
        plt.clf()
        plt.close()


    def model_logging(self):
        """
        For a single model, it logs these specific parameters
        """
        architecture = ''
        for i in range(0,len(self.weights)):                    
                                                                # Activation function 
            architecture += str(self.weights[i].shape[1]) + "-" + str(self.activation_functions[i+1].__name__)[0] + "+" 
        architecture = architecture[:-1] # Remove to the last plus

        error_func = self.error_function.__name__

        optimizer = self.optimizer 
        
        if optimizer == 'backprop':
            learning_rate= str(self.learning_rate)
        else:
            learning_rate=str(0)
        
        # Error
        best_validation_set_err = self.epoch_error_validation_plot[self.optimal_error_position-2] * 100
        best_validation_set_classif_err = (1-self.epoch_validation_accuracy_plot[self.optimal_error_position-2]) * 100

        best_test_set_err = self.epoch_error_testing_plot[self.optimal_error_position-1] * 100
        best_test_set_classif_err = (1-self.epoch_testing_accuracy_plot[self.optimal_error_position-1]) * 100
        
        total_epochs = self.total_epochs
        # Also called relevant epoch
        best_epoch = self.optimal_error_position  # NOTE: Epoch is saved on the best size

        data_shuffle = self.data_shuffle

        batch_size = self.batch_size 

        overfit = self.GL

        return [optimizer, learning_rate, batch_size, architecture, error_func, best_validation_set_err, best_validation_set_classif_err, \
            best_test_set_err, best_test_set_classif_err, best_epoch, total_epochs, data_shuffle, overfit]
    
    def __del__(self):
        # Saving memory
        print('Destructor called, Network Deleted.')

