import numpy as np
from propagation.feedforward import FeedForward
from optimizer import Optimizer
import logging

from utils.logs import create_logger
log = create_logger(__name__)

class GeneticOptimizer(FeedForward, Optimizer):
    """
    - An Empirical Study of Non-binary Genetic Algorithm based
        Neural Approaches for Classification.
    """
    def __init__(self, number_of_parents, fitness_eval, weights_initialization=None):
        """_summary_

        Parameters
        ----------
        number_of_parents : int
            The number of parents (should be an even number)
        fitness_eval : 
            'accuracy' or 'error' which is used as fitness evaluation
            metric.
        """

        self.initialization_method = weights_initialization

        if number_of_parents%2==0:
            self.number_of_parents = number_of_parents
        else:
            raise Exception("Number of parents should be even.")

        if fitness_eval=="accuracy" or fitness_eval=="error":
            self.fitness_eval = fitness_eval
        else:
            raise Exception("'accuracy' or 'error' should be used the fitness metric.")

        self.optimizer_type = 'non-gradient'
        self.optimizer_name = 'genetic'

        log.info(f"Optimizer Name: {self.optimizer_name}")
        log.info(f"Type of optimizer: {self.optimizer_type}")

        Optimizer.__init__(self)

    def init_network_structures(self, layers):
        """Initialize weights, biases and activation functions
        used in the network, which are saved in the layers 
        passed in. Since this is a genetic optimizer, the number
        of parents creates the number of random weights and/or bias
        instances used in feedforward. 

        Parameters
        ----------
        layers : tuple (int, function)
            (<Number of Neurons in the layer>, <Activation function>)

        Returns
        -------
        weights and biases used by the optimizer
        """
        # Assigning the layers to be used by the optimizers
        self.layers = layers
        
        log.info("Initiliazing Genetic Algorithm Structure")

        # Genetic Algorithm has multiple parents(so multiple instantiations of weights)
        weights = []
        if self.use_bias:
            bias = []

        for p in range(self.number_of_parents):

            single_parent_weight = []
            if self.use_bias:
                single_parent_bias = []

            if self.initialization_method is None:
                optimizer_params = dict(name='standard')
            else:
                # Weights initializer
                optimizer_params = self.initialization_method
                log.info(f"The initialization paramters are {optimizer_params}")
                # intializer=getattr(Initializers(), optimizer_params['name'])
                # Get access to the initializer class
                intializer=getattr(Optimizer(), optimizer_params['name'])

            for i in range(1, len(self.layers)):
                optimizer_params['dim0'] = self.layers[i-1][0]
                optimizer_params['dim1'] =self.layers[i][0]

                if (optimizer_params['name']=="standard"):
                    # Default
                    resulting_init = np.random.rand(
                        optimizer_params['dim0'],
                        optimizer_params['dim1'])-0.5
                else:
                    resulting_init = intializer(**optimizer_params)

                single_parent_weight.append(
                    resulting_init
                    # np.random.rand(self.layers[i-1][0], self.layers[i][0]),
                )

                log.info(f"Weights Shape {p} {single_parent_weight[i-1].shape}")

                if self.use_bias:
                    single_parent_bias.append(
                        np.random.rand(1, self.layers[i][0]),
                    )
                    log.info(f"Bias Shape {p} {single_parent_bias[i-1].shape}")

            weights.append(np.array(single_parent_weight, dtype=object))

            if self.use_bias:
                bias.append(single_parent_bias)

        if self.use_bias:
            return weights, bias
        else:
            return weights, 0

    def init_propagations(self):
        """
        Genetic algorithm only uses the feedforward algorithm.
        Gets used after init_bias_usage and init_propagations. 
        """
        # Initializing FeedForward
        FeedForward.__init__(
                        self,
                        use_bias=self.use_bias, 
                        activation_functions=self.activation_functions
                        )

    def init_measures(self, weights):
        """
        Initialize the measures used for genetic algorithm.

        Parameters
        ----------
        weights: list[list[np.array]]
        """
        weight_length = len(weights)

        train_accuracy_results = np.zeros(weight_length)
        total_training_error = np.zeros(weight_length)

        # Catches the first iteration, when the number of children are
        # twice that of the parents
        if (self.number_of_parents == weight_length):
            verification_accuarcy_results = np.zeros(2*weight_length)
            total_verification_error = np.zeros(2*weight_length)

            test_accuarcy_results = np.zeros(2*weight_length)
            total_test_error = np.zeros(2*weight_length)

        else:
            verification_accuarcy_results = np.zeros(weight_length)
            total_verification_error = np.zeros(weight_length)

            test_accuarcy_results = np.zeros(weight_length)
            total_test_error = np.zeros(weight_length)

        return train_accuracy_results, total_training_error, \
                verification_accuarcy_results, total_verification_error, \
                    test_accuarcy_results, total_test_error
                
    def arith_crossover(self, lambda_val, parent_arr):
        """Arithmetic crossover.
        # TODO: Add description
        # Add different types of crossover

        Parameters
        ----------
        lambda_val : float
            Random value between 0 and 1
        parent_arr : list[list]
            List of flattened weights array

        Returns
        -------
        children
            Children array occuring after crossover.
        """
        # Each pair of parents create two children
        children = []
        for q in range(0,len(parent_arr),2):
            # First child
            k_ij = lambda_val*parent_arr[q] + (1 - lambda_val)*parent_arr[q+1]
            children.append(k_ij)
            # Second child
            k_ij = lambda_val*parent_arr[q+1] + (1 - lambda_val)*parent_arr[q]
            children.append(k_ij)

        return children

    def tournament_selection(self, param):  
        """Tournament selection method.
        # TODO: Add description
        # Add different Types of Selection

        Parameters
        ----------
        param : list[list[np.array]]
            Parents weights or Parents Bias 

        Returns
        -------
        parents
            Flattened weights/bias arrays, where each parent is
            one instance of the weights.
        """
        parents = []
        parents_structure = []
        # Length of flatten weights, we already have the dimension of the weights
        for z in range(0, len(param)): # Looping through all the parents 
            flat_parent_lengths = [0] 
            flat_parent = []

            for single in range(len(param[z])):
                # Getting the individual parent
                flat_parent.extend(param[z][single].flatten())
                flat_parent_lengths.append(len(flat_parent))

            flat_parent = np.array(flat_parent)
            parents.append(flat_parent)
            parents_structure.append(flat_parent_lengths)

        return parents, parents_structure

    def mutation(self, children):
        """Mutation function.
        TODO: Add a description

        Parameters
        ----------
        children : list[list]
            Children of parent weights/biases

        Returns
        -------
        children : list[list]
            Children that have random mutations performed on them.
        """
        # This would be the length of the parent - 1

        for f in range( 0, len(children) ):
            mutation_arr = np.random.random(children[f].shape) - 0.5
            mutation_arr = np.where(np.array(mutation_arr[f]) > 0, mutation_arr, 0)
            children[f] = np.array(children[f]) + mutation_arr

        return children
    
    def evaluation(self, param, children, parents_structure):
        """Restructures the param array(weights/biases) into their
        original shape, where the children get restructured into 
        a into the original shape. 

        Parameters
        ----------
        param : list[list[np.array]]
            Weights/bias array
        children : list[list]
            Children array that results from evolution process
        parents_structure : _type_
            Lengths determined by the flattening process during selection

        Returns
        -------
        final_param
            Children restructured into the input parameter(weights/bias) shape
        """
        # Reshaping back to the original
        new_param = param
        for z in range(len(parents_structure)):
            
            child_weights = []

            for single in range(0,len(parents_structure[z])-1):
                
                child_weights.append(
                    children[z][ # Was changed
                            parents_structure[z][single] : parents_structure[z][single+1]
                        ]
                        .reshape(
                            param[z][single].shape
                        )
                    )
                
            # Restructured array 
            new_param.append(child_weights)

        final_param = new_param

        return final_param

    def forward_prop_fit(self, 
                        X, Y, 
                        accuracy_results, 
                        total_training_error, 
                        weights, bias):
        """Forward propagation resulting in training error,
        accuracy results and layer output.

        Parameters
        ----------
        X : np.array
            Feature vector of data
        Y : np.array
            Labeled output of data
        accuracy_results : list[]
            Training accuracy per individual parent
        total_training_error : list[]
            Error per individual parent
        weights : list[list[np.array]]
            List of weight arrays
        bias : list[list[np.array]]
            List of bias arrays

        Returns
        -------
        total_training_error : list[]
            Error per individual parent
        accuracy_results : list[]
            Training accuracy per individual parent
        ff_results : list
            Output layer of network
        """

        bias_genetic = 0
        ff_results = []
        for p in range(len(weights)):

            weights_gentic = weights[p] # Make a copy of the array
            if self.use_bias:
                bias_genetic = bias[p]

            # Extracting the final layer output [-1]
            ff_results.append(
                self.feedforward(x=X, 
                    weights=weights_gentic, 
                    bias=bias_genetic)[-1]
                )

            # Getting loss according to error function
            total_training_error[p] += self.error_function(Y, ff_results[p])

            # Training Accuracy 
            if (np.argmax(ff_results[p]) == np.argmax(Y)):
                accuracy_results[p] += 1 # Used for genetic algorithm

        return total_training_error, accuracy_results, ff_results

    def optimize(self, error, accuracy, weights, bias):
        """Optimization function used to adjust weights
        and biases of the network.

        Parameters
        ----------
        error : list[]
            List of errors for each correspoding set
                of weights and biases
        accuracy : list[]
            List of accuracy of predictions for each
                corresponding weights and biases
        weights : list[list[np.array]]
            List of weights used corresponding to the number
                of parents
        bias : list[list[np.array]]
            List of bias used corresponding to the number
                of parents

        Returns
        -------
        Weights, Bias
            Weights and Biases resulting from evolution

        Raises
        ------
        NotImplementedError
        """
        # Which method we are using to improve the population
        if (self.fitness_eval == 'error'):
            fitness_measure = error
            best_performers = sorted(
                range(len(fitness_measure)), 
                    key=lambda x: fitness_measure[x])

        elif (self.fitness_eval == 'accuracy'):
            fitness_measure = accuracy
            best_performers = sorted(
                range(len(fitness_measure)), 
                    key=lambda x: fitness_measure[x])[::-1]

        else:
            raise NotImplementedError("Please choose an optimizer")
        # ================================================================ #
        #                       GENETIC ALGORITHM                          #
        # ================================================================ #
        # ---------------------------------------------------------------- #
        #                        Fitness Measure                           #
        # ---------------------------------------------------------------- #

        # Getting the strongest performing populations in order
        temp_weights = []
        if self.use_bias:
            temp_bias = []

        for a in best_performers: 
            temp_weights.append(weights[a])
            if self.use_bias:
                temp_bias.append(bias[a])

        # Reordering the best performers
        weights = temp_weights[0:self.number_of_parents]
        if self.use_bias:
            bias = temp_bias[0:self.number_of_parents]
        # ---------------------------------------------------------------- #
        #                        Parent Selection                          #
        # ---------------------------------------------------------------- #
        # Tournament selection
        # So the however many parents would fight against each other an only 
        # the best will remain

        parents_weight, parents_weight_structure = self.tournament_selection(weights)
        if self.use_bias:
            parents_bias, parents_bias_structure = self.tournament_selection(bias)

        
        parents_weight = np.array(parents_weight) # There will be a number of parents

        if self.use_bias:
            parents_bias = np.array(parents_bias) # There will be a number of parents

        # ---------------------------------------------------------------- #
        #                           Crossover                              #
        # ---------------------------------------------------------------- #
        # Arithmetic Crossover
        # TODO: Add other methods of crossover
        lambda_val = np.random.random()
        children_weights = self.arith_crossover(
            lambda_val=lambda_val, 
            parent_arr=parents_weight)

        if self.use_bias:
            children_bias = self.arith_crossover(
                lambda_val=lambda_val, 
                parent_arr=parents_bias)

        # ---------------------------------------------------------------- #
        #                           Mutation                               #
        # ---------------------------------------------------------------- #
        children_weights = self.mutation(children_weights)    

        if self.use_bias:
            children_bias = self.mutation(children_bias)

        # ---------------------------------------------------------------- #
        #                           Evaluation                             #
        # ---------------------------------------------------------------- #

        weights = self.evaluation(
            param=weights, 
            children=children_weights, 
            parents_structure=parents_weight_structure
        )

        if self.use_bias:
            bias = self.evaluation(
                param=bias, 
                children=children_bias, 
                parents_structure=parents_bias_structure
            )

        if self.use_bias:
            return weights, bias
        else:
            return weights, 0
    
