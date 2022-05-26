import numpy as np
from activations.activation_functions import relu, sigmoid, softmax, tanh
from common.early_stopping import EarlyStopping
from common.plotting_functions import Plots
from data_sources.proben1 import Proben1
from errors.error_functions import mse, rms, squared_error
from neural_network.neural_network import Network
from neural_network.optimizers.gradient.delta_bar_delta import DeltaBarDelta
from neural_network.optimizers.gradient.gradient_descent import GradientDescent
from neural_network.optimizers.gradient.gradient_descent_momentum import \
    GradientDescentWithMomentum
from neural_network.optimizers.gradient.rprop import Rprop

if __name__=='__main__':

    proben = Proben1()
    proben.download_data()
    proben.get_dataset_dirs() 

    (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = proben.load_data(data_set_name='cancer')[2]
    filename = proben.get_filenames(data_set_name='cancer')[2]

    print('-'*10)
    print(filename) 
    print('-'*10)

    # -------------------------------------------------------------------- #
    #                         START DATASET PREP                           #
    # -------------------------------------------------------------------- #
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])

    x_validate = x_validate.reshape(x_validate.shape[0], 1, x_validate.shape[1])
    y_validate = y_validate.reshape(y_validate.shape[0], 1, y_validate.shape[1])

    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])

    normalizing_factor_x = x_train.max()
    normalizing_factor_y = y_train.max()

    x_train /= normalizing_factor_x
    y_train /= normalizing_factor_y

    x_validate /= normalizing_factor_x
    y_validate /= normalizing_factor_y

    x_test /= normalizing_factor_x
    y_test /= normalizing_factor_y

    # --- Input and output layer size
    input_layer_size = x_train[0].shape[1]
    output_layer_size = y_train[0].shape[1]

    # optimizer_param=dict(name='heuristic', lower=-0.6, upper=0.3)
    optimizer_param=dict(name='xavier')
    # optimizer_param=dict(name='he')
    # optimizer_param=None

    # Normal Gradient Descent
    nn_train = Network(

        layers=[
            (input_layer_size,tanh),
            (16,tanh),
            (8,tanh),
            (output_layer_size,softmax)
        ],

        error_function=squared_error,

        use_bias=True,
        
        # optimizer = GeneticOptimizer(number_of_parents=4, 
        #                             fitness_eval='accuracy',
        #                             weights_initialization=optimizer_param),

        # optimizer= GradientDescent(learning_rate=0.5, weights_initialization=optimizer_param),

        # optimizer= GradientDescentWithMomentum(learning_rate=0.05, beta=0.9, weights_initialization=optimizer_param),

        optimizer= DeltaBarDelta(theta=0.1, mini_k=0.01, phi=0.1, weights_initialization=optimizer_param),

        # optimizer= Rprop(delta_max=50, delta_min=0, eta_plus=1.1, eta_minus=0.5, weights_initialization=optimizer_param),

        # training_params = EarlyStopping(alpha=10,
        #                                 pkt_threshold=0.1,
        #                                 k_epochs=5,
        #                                 )
    )

    nn_train.fit(
                x_train=x_train, 
                y_train=y_train, 
                x_test=x_test, 
                y_test=y_test, 
                x_validate=x_validate,
                y_validate=y_validate,
                epochs=150,
                batch_size=64, # If batch size equals 1, we have online learning
                shuffle_training_data=True,
                )      

    plots = Plots(nn_train)
    # plots.plot_epoch_error(ds_name="Cancer",save_dir="cancer")
    plots.plot_epoch_accuracy(ds_name="Cancer",save_dir="cancer1")

    # Make a prediction
    print(nn_train.predict(x_test[1])) 
    print(y_test[1]) 

