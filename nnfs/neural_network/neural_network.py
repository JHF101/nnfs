import pickle

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np

from sklearn.metrics import confusion_matrix
from nnfs.common.plotting_functions import Plots
from nnfs.utils.utils import shuffle_arrays

from nnfs.utils.logs import create_logger
log = create_logger(__name__)
class Network(Plots):
    """
    Neural Network class that uses fully connected layers
    """
    def __init__(self,
                optimizer=None,
                layers=[],
                use_bias=False,
                error_function=None,
                training_params=None,
                load_model=False):
        """Initialization of Neural Network

        Parameters
        ----------
        layers : tuple
            (Number of nodes in a layer, activation function)
            >>> (6, activation_function)
        learning_rate : float
            Learning rate used in the model
        use_bias : bool, optional
            Determines whether to add bias nodes on each layer, except the
            output layer, by default True
        plotting_func_bool : bool, optional
            Determines whether to plot, by default False
        """

        if (load_model==False):
            self.use_bias = use_bias
            log.info(f"Is bias being used? {self.use_bias}")

            # Number of weights is number of layers - 1
            self.optimizer = optimizer
            log.info(f"Using the {self.optimizer.optimizer_name} is used")

            # Just init for now
            self.bias = []

            # Intialize layers
            self.layers = layers

            # Initialize error function
            self.error_function = error_function

            # Initialize error function
            self.optimizer.init_error_func(self.error_function)
            self.optimizer.init_bias_usage(use_bias)

            # Initialize weight structures
            self.weights, self.bias = self.optimizer.init_network_structures(
                                                                layers=self.layers,
                                                                )

            # Initialize activation functions
            self.activation_functions = self.optimizer.init_activation_functions()

            # Initialize propagation methods
            self.optimizer.init_propagations()

            # TODO: Save these plots
            self.epoch_error_training_plot = []
            self.epoch_error_validation_plot = []
            self.epoch_error_testing_plot = []
            self.epoch_training_accuracy_plot = []
            self.epoch_validation_accuracy_plot = []
            self.epoch_testing_accuracy_plot = []

            self.confusion_matrix = []

            # TODO: Save all weights into a super mega array
            self.mega_weights_array=[]
            self.mega_bias_array = []

            self.mega_delta_weights_array = []
            self.mega_delta_bias_array = []

            # Used to stop training
            if training_params is not None:
                self.early_stopping = training_params
                log.info(self.early_stopping.early_stopper)

                # Overfit parameter
                self.GL = 0
                self.k_epochs = self.early_stopping.k_epochs
                self.pkt_threshold = self.early_stopping.pkt_threshold
                self.alpha = self.early_stopping.alpha
            else :
                self.early_stopping = None
                log.info("No early stopping is used")

            # Total number of epochs
            self.total_epochs = 0

            # The point at which training was stopped
            self.optimal_error_position = 0
        else:
            # Expecting model
            self.optimizer = optimizer

    def fit(self,
            x_train, y_train,
            x_test, y_test,
            x_validate=None, y_validate=None,
            epochs=1,
            validations_set_percent=0.0,
            batch_size=1,
            shuffle_training_data=True,
            generate_plots=False):
        """Fit the model to the dataset.

        x_validate and y_validate can be given by the user, however
        if it is not provided, then no validation set has to be used
        or the user can specify what percentage of the training dataset
        should be used for validation.

        Parameters
        ----------
        x_train : np.array
            The training data features
        y_train : np.array
            The training data labels
        x_test : np.array
            The test data features
        y_test : np.array
            The test data labels
        x_validate : np.array, optional
            The validation data features, by default None
        y_validate : np.array, optional
            The validation data labels, by default None
        epochs : int, optional
            Number of passes of the entire training set, by default 1
        validations_set_percent : float, optional
            Percentage of training data used for validation data, by default 0.0
        batch_size : int, optional
            Number of training samples used before updating the weights, by default 1
        shuffle_training_data : bool, optional
            Shuffles the data, by default True

        Raises
        ------
        Exception
            Validation set percentage must be [0,1]
        """
        # Whether to display graphs or not
        self.generate_plots = generate_plots

        if self.generate_plots:
            Plots.__init__(self)

        x_train_size = len(x_train)

        # Init
        average_bias = 0

        training_set_size = 0

        if (x_validate is not None) and (y_validate is not None):
            # Determine the validation sets were actually given
            x_validation_set = x_validate
            y_validation_set = y_validate
            training_set_size = x_train_size
            validations_set_size = len(x_validate)

        else:
            if (1.0 > validations_set_percent > 0.0):
                # If the validation sets are not given, we can make our own
                training_set_size = int(x_train_size*(1-validations_set_percent))
                validations_set_size = x_train_size - training_set_size
                x_validation_set = x_train[training_set_size:training_set_size+validations_set_size]
                y_validation_set = y_train[training_set_size:training_set_size+validations_set_size]

            elif (validations_set_percent == 0.0):
                # No validation set will be used
                training_set_size = x_train_size
                validations_set_size = 0
                log.info("No validation set will be used")

            else:
                raise Exception("Validation percentage size must be between 0 and 1")

        # Setting the training size
        x_train = x_train[0:training_set_size]
        y_train = y_train[0:training_set_size]

        # Note: Randomize the order possibly
        example_size = training_set_size

        # Test set length
        y_test_len = len(y_test)

        # example_size = training_set_size-1 # N-1
        log.info("Begin training model")

        # Train
        for i in range(epochs):

            # Initializing accuracy measures
            train_accuracy, total_training_error, \
                validation_accuracy, total_validation_error, \
                    test_accuracy, total_test_error = \
                        self.optimizer.init_measures(self.weights)

            # -------------------------------------------------------------- #
            #                          Training                              #
            # -------------------------------------------------------------- #
            # Randomizing the data order
            if (shuffle_training_data==True):
                x_train, y_train = shuffle_arrays(x_train, y_train)

            # Total number of iterations
            batches = int(np.ceil(example_size/batch_size))
            log.debug(f"Number of batches {batches}")

            for outer in range(0, batches):

                if (self.optimizer.optimizer_type == 'gradient'):
                    # Used as a state control variable
                    delta_weight_counter = 0
                    # Keeps track of the number of gradients
                    num_of_summations_per_batch = 0

                # Setting lower and upper bounds for batch
                lower_batch = int(outer*batch_size)
                upper_batch = int((outer+1)*batch_size)

                for j in range(lower_batch, upper_batch):

                    # Condition to break out of the training loop
                    if (j >= example_size):
                        break

                    # Average Weights
                    if (self.optimizer.optimizer_type == 'gradient'):

                        # Forward Propagation
                        total_training_error, train_accuracy, ff_results = \
                            self.optimizer.forward_prop_fit(
                                x_train[j],
                                y_train[j],
                                train_accuracy,
                                total_training_error,
                                self.weights,
                                self.bias)

                        # Backward Propagation
                        self.delta_weight, self.delta_bias = \
                            self.optimizer.backpropagation(
                                y_train[j],
                                ff_results[-1],
                                self.weights,
                                ff_results)

                        # Keeping ttrack of gradients
                        if (delta_weight_counter==0):
                            # Initializing for summations
                            average_gradients = self.delta_weight
                            if self.use_bias:
                                average_bias = self.delta_bias

                        if (delta_weight_counter>0):
                            # Summing the gradients for batch
                            for b in range(len(self.delta_weight)):
                                average_gradients[b] += self.delta_weight[b]
                                if self.use_bias:
                                    average_bias[b] += self.delta_bias[b]

                        # Incrementing the counters
                        delta_weight_counter += 1
                        num_of_summations_per_batch +=1

                    elif (self.optimizer.optimizer_type == 'non-gradient'):

                        # --- Forward Propagation
                        total_training_error, train_accuracy, _ = \
                            self.optimizer.forward_prop_fit(
                                x_train[j],
                                y_train[j],
                                train_accuracy,
                                total_training_error,
                                self.weights,
                                self.bias)

                    else:
                        log.exception("Something went wrong!")


            # Averaging over all samples
            total_training_error /= example_size # TODO: Consider PROBEN1 error
            train_accuracy /= example_size

            if (self.optimizer.optimizer_type == 'gradient'):

                # Averaging gradients
                average_gradients = [average_gradients[i]/num_of_summations_per_batch for i in range(len(average_gradients))]
                if self.use_bias:
                    average_bias = [average_bias[i]/num_of_summations_per_batch for i in range(len(average_bias))]

                # Reversing the order of gradients to make optimizations easier
                dE_dwij_t, dE_dbij_t = self.optimizer.flip_weights(
                    average_gradients=average_gradients,
                    average_bias=average_bias)

                # Mega Delta Array
                # NOTE: The order in which the gradients are saved are reversed
                self.mega_delta_weights_array.append(dE_dwij_t)
                if self.use_bias:
                    self.mega_delta_bias_array.append(dE_dbij_t)

                # Removing old entries to improve efficiency
                if len(self.mega_delta_weights_array)>5:
                    self.mega_delta_weights_array.pop(0)
                if self.use_bias:
                    if len(self.mega_delta_bias_array)>5:
                        self.mega_delta_bias_array.pop(0)

                # Optimizing the weights
                self.weights, self.bias = self.optimizer.optimize(
                    dE_dwij_t=dE_dwij_t,
                    dE_dbij_t=dE_dbij_t,
                    weights=self.weights,
                    bias=self.bias,
                    mega_delta_weights_array=self.mega_delta_weights_array,
                    mega_delta_bias_array=self.mega_delta_bias_array if self.use_bias else 0)

            elif ((self.optimizer.optimizer_type == 'non-gradient') \
                    and (self.optimizer.optimizer_name == 'genetic')):
                # Only genetic algorithm for now is being used
                self.weights, self.bias = self.optimizer.optimize(
                    error=total_training_error,
                    accuracy=train_accuracy,
                    weights=self.weights,
                    bias=self.bias)

            else:
                log.exception("Something went wrong! Optimizer type is not being used correctly.")

            # -------------------------------------------------------------- #
            #                          Validating                            #
            # -------------------------------------------------------------- #

            if ((validations_set_percent > 0.0) or (validations_set_size > 0)):
                for v in range(0, validations_set_size):
                    # Forward Propagation
                    total_validation_error, validation_accuracy, _ = \
                        self.optimizer.forward_prop_fit(
                            X=x_validation_set[v],
                            Y=y_validation_set[v],
                            accuracy_results=validation_accuracy,
                            total_training_error=total_validation_error,
                            weights=self.weights,
                            bias=self.bias)

                # Averaging
                total_validation_error /= validations_set_size
                validation_accuracy /= validations_set_size

            # -------------------------------------------------------------- #
            #                              Testing                           #
            # -------------------------------------------------------------- #

            # Saving for confusion matrix
            all_predictions = []
            actual_values=[]

            for t in range(y_test_len):
                total_test_error, test_accuracy, ff_test_result = \
                    self.optimizer.forward_prop_fit(
                        X=x_test[t],
                        Y=y_test[t],
                        accuracy_results=test_accuracy,
                        total_training_error=total_test_error,
                        weights=self.weights,
                        bias=self.bias)

                temp_all_predictions = []
                temp_actual_values=[]

                # for cp in range(len(ff_test_result)):
                # Testing accuracy
                categorical_prediction = np.argmax(ff_test_result[-1])
                log.warning(f"Categorical prediction of : {categorical_prediction}")
                temp_all_predictions.append(categorical_prediction)
                actual_output= np.argmax(y_test[t])
                log.warning(f"Actual output prediction of : {actual_output}")
                temp_actual_values.append(actual_output)

                # Arrays used for confusion matrix
                all_predictions.append(temp_all_predictions)
                actual_values.append(temp_actual_values)

            # Averaging
            total_test_error /= y_test_len
            test_accuracy /= y_test_len

            # -------------------------------------------------------------- #
            #                    Confusion Matrix                            #
            # -------------------------------------------------------------- #

            actual_values = np.array(actual_values)
            all_predictions = np.array(all_predictions)

            # Get the columns of the predictions
            # TODO: This is causing big errors
            # for cm in range(all_predictions.shape[0]):
            #     self.confusion_matrix.append(
            #         confusion_matrix(actual_values[cm,:], all_predictions[cm,:]) # This is wrong
            #     )
            self.confusion_matrix.append(confusion_matrix(actual_values, all_predictions))

            # Non-Gradient (Genetic)
            if (type(validation_accuracy)==list): # Means that we have more
                log.info("="*30)
                log.info(f"epoch {i+1}/{epochs}")
                log.info("*"*5, "Training", "*"*5)
                log.info(f"train_loss = \n{total_training_error}")
                log.info(f"train_accuracy = \n{train_accuracy}")
                log.info("-"*5, "Validation", "-"*5)
                log.info(f"val_loss = \n{total_validation_error}")
                log.info(f"val_accuracy = \n{validation_accuracy}")
                log.info("-"*5, "Validation", "-"*5)
                log.info(f"total_loss = \n{total_test_error}")
                log.info(f"test_accuracy = \n{test_accuracy}")
                log.info("-"*30)
                log.info("\n")

            elif (type(validation_accuracy)==float):
                # Gradient Based
                log.info(f"""epoch {i+1}/{epochs} train_loss = {total_training_error} - train_accuracy = {train_accuracy} | val_loss = {total_validation_error} - val_accuracy = {validation_accuracy} | test_loss={total_test_error} - test_accuracy = {test_accuracy} \r""")

            else:
                # No validation set used
                log.info(f"""epoch {i+1}/{epochs} train_loss = {total_training_error} - train_accuracy = {train_accuracy} | test_loss={total_test_error} - test_accuracy = {test_accuracy} \r""")

            # -------------------------------------------------------------- #
            #                         Saving Model                           #
            # -------------------------------------------------------------- #
            # Saving error and accuracy plots
            self.epoch_error_training_plot.append(total_training_error)
            self.epoch_training_accuracy_plot.append(train_accuracy)

            try:
                if (validations_set_percent>0.0) or (len(x_validate)>0):
                    self.epoch_error_validation_plot.append(total_validation_error)
                    self.epoch_validation_accuracy_plot.append(validation_accuracy)
            except:
                log.info("No validation set")

            self.epoch_error_testing_plot.append(total_test_error)
            self.epoch_testing_accuracy_plot.append(test_accuracy)

            # Contains all of the weights for training
            self.mega_weights_array.append(self.weights)
            if self.use_bias:
                self.mega_bias_array.append(self.bias)

            # -------------------------------------------------------------- #
            #                      Stopping Condition                        #
            # -------------------------------------------------------------- #

            if (self.early_stopping is not None):
                # Number of epochs actually run
                self.total_epochs += 1 # NOTE: actually has an offset of 1
                try:
                    """Only take the top performing for genetic and only one element for
                    the gradient based methods.
                    """
                    for es in range(1): #len(validation_accuracy)):
                        if (self.optimizer == 'non-gradient'):
                            epoch_error_validation_plot_temp = np.array(self.epoch_error_validation_plot, dtype=object)
                            epoch_error_validation_plot_per_parent = epoch_error_validation_plot_temp[:,es].tolist()
                        else:
                            epoch_error_validation_plot_per_parent = self.epoch_error_validation_plot

                        # ----- Early stopping ----- #
                        early_stop, self.GL = self.early_stopping.early_stopping(
                            epoch_error_validation_plot_per_parent)

                        if (early_stop == True):
                            log.info("Early stopping stopped training.")
                            self.optimal_error_position = self.total_epochs - \
                                np.argmin(epoch_error_validation_plot_per_parent[::-1][0:self.k_epochs])
                            break

                        # ----- Training Progress ----- #
                        pkt_stop, self.pkt_threshold = self.early_stopping.training_progress(
                            epoch_error_validation_plot_per_parent, self.pkt_threshold)

                        if (pkt_stop == True):
                            log.info("P_k(t) stopped training.")
                            self.optimal_error_position = self.total_epochs - \
                                np.argmin(epoch_error_validation_plot_per_parent[::-1][0:self.k_epochs])
                            break

                    if ((pkt_stop == True) or (early_stop == True)):
                        log.info("We are done training because stopping criterion is met.")
                        break

                except:
                    log.debug(f"Skipping the stopping criterion, epoch = {i}.")
                    pass

            # -------------------------------------------------------------- #
            #                        Live Plotting                           #
            # -------------------------------------------------------------- #
            if self.generate_plots:
                self.update_data(self_data=self)
                self.plot_epoch_error(ds_name="Stream", save_dir="Stream")
                self.plot_epoch_accuracy(ds_name="Stream1", save_dir="Stream1")
                self.plot_confusion_matrix(ds_name="Stream2", save_dir="Stream2")

        log.info("Done Training Model.")

    def predict(self, X):
        """Used to be able to predict output of feature vectors X.

        Parameters
        ----------
        X : np.array
            Feature vector

        Returns
        -------
        prediction
            Output vector
        """

        # X is a single element
        return self.optimizer.predict(X=X,
                                    weights=self.weights,
                                    bias=self.bias)

    def save_model(self, file_dir):
        """Saves model to a pickle file.

        Parameters
        ----------
        file_dir : str
            The directory to which the model will be saved
        """
        model_params={ key:value for key, value in vars(self).items() }
        with open(f'{file_dir}', 'wb') as f:
            pickle.dump(model_params, f)

        log.info(f'The following class variables were saved: {[key for key, value in vars(self).items()]}')
        log.info(f"Saved model successfully to {file_dir}")

    def load_model(self, file_dir, **kwargs):
        """Loads the saved model to the class.

        Parameters
        ----------
        file_dir : str
            Directory for the saved model to be loaded from
        kwargs:
            activation_functions
        """
        with open(f'{file_dir}', 'rb') as f:
            pickle_dict = dict(pickle.load(f))

        # Loading the pickle elements to the class
        for key_pic, value in pickle_dict.items():
            if (key_pic != "optimizer"):
                # Allows user to select their optimizer
                setattr(self, key_pic, value)

        log.info(f"Elements contained in class: {[key for key, value in vars(self).items()]}")
        log.info(f"Optimal error position: {self.optimal_error_position}")
        log.info(f"Validation accuracy of previous model: {self.epoch_validation_accuracy_plot[self.optimal_error_position-1]}")

        # Load the optimal weights (The one from the best epoch)
        self.weights = self.mega_weights_array[self.optimal_error_position-1]

        # Determine if bias is being used
        if ('use_bias' in kwargs.keys()):
            self.use_bias = kwargs['use_bias']

        # Initialize Bias Usage
        self.optimizer.init_bias_usage(self.use_bias)

        if self.use_bias:
            log.info("Bias is being used")

            # Checking if we do have a bias array
            if (type(self.bias)==list):
                self.bias = self.mega_bias_array[self.optimal_error_position-1]
            else:
                self.bias = []
                for l in range(1, len(self.layers)):
                    self.bias.append(
                        np.random.rand(1, self.layers[l][0]) - 0.5,
                    )

                log.info("Bias was not used in the previous model, so creating structure")

        else:
            # If bias is not being used make equal to zero
            self.bias = 0

        # Loading layers into the optimizer
        self.optimizer.layers=self.layers

        # Ability to change activation functions
        if ('activation_functions' in kwargs.keys()):
            self.optimizer.init_activation_functions(activation_functions=kwargs['activation_functions'])
            log.info('Using new activation functions.')
        else:
            self.optimizer.init_activation_functions()
            log.info("Using the same activation functions.")

        # Ability to change error function
        if ('error_function' in kwargs.keys()):
            self.optimizer.init_error_func(kwargs['error_function'])
            log.info('Using new error_function functions.')
        else:
            self.optimizer.init_error_func(self.error_function)
            log.info("Using the same error function.")

        # Initializing propagations of optimizer
        self.optimizer.init_propagations()

        # TODO: Account for non-gradient to gradient methods
        # TODO: Account for gradient to non-gradient methods