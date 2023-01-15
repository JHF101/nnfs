import numpy as np
from nnfs.activations.activation_functions import relu, sigmoid, softmax, tanh
from nnfs.common.early_stopping import EarlyStopping
from nnfs.common.plotting_functions import Plots
from nnfs.data_sources.proben1 import Proben1
from nnfs.errors.error_functions import mse, rms, squared_error
from nnfs.neural_network.neural_network import Network
from nnfs.neural_network.optimizers.gradient.delta_bar_delta import DeltaBarDelta
from nnfs.neural_network.optimizers.gradient.gradient_descent import GradientDescent
from nnfs.neural_network.optimizers.gradient.gradient_descent_momentum import (
    GradientDescentWithMomentum,
)
from nnfs.neural_network.optimizers.gradient.rprop import Rprop
from nnfs.neural_network.optimizers.gradient.rms_prop import RMSProp
from nnfs.neural_network.optimizers.gradient.adam import Adam

import matplotlib.pyplot as plt
import matplotlib.animation as animation

proben = Proben1()
proben.download_data()
proben.get_dataset_dirs()

(x_train, y_train), (x_validate, y_validate), (x_test, y_test) = proben.load_data(
    data_set_name="soybean"
)[1]
filename = proben.get_filenames(data_set_name="soybean")[1]


print("-" * 10)
print(filename)
print("-" * 10)

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

# optimizer_param=dict(name='heuristic', lower=-0.3, upper=0.3)
# optimizer_param=dict(name='xavier')
# optimizer_param=dict(name='he')
optimizer_param = None

# Normal Gradient Descent
nn_train = Network(
    layers=[
        (input_layer_size, tanh),
        (64, tanh),
        # (8,tanh),
        # (4,tanh),
        (output_layer_size, tanh),
    ],
    error_function=squared_error,
    use_bias=True,
    # optimizer = GeneticOptimizer(number_of_parents=4,
    #                             fitness_eval='accuracy',
    #                             weights_initialization=optimizer_param),
    # optimizer= GradientDescent(learning_rate=0.5, weights_initialization=optimizer_param),
    # optimizer= GradientDescentWithMomentum(learning_rate=0.09, beta=0.9, weights_initialization=optimizer_param),
    # optimizer= DeltaBarDelta(theta=0.1, mini_k=0.01, phi=0.1, weights_initialization=optimizer_param),
    # optimizer= Rprop(delta_max=50, delta_min=0, eta_plus=1.1, eta_minus=0.5, weights_initialization=optimizer_param),
    optimizer=RMSProp(
        learning_rate=0.0075, beta=0.99, weights_initialization=optimizer_param
    ),
    # optimizer= Adam(learning_rate=0.01, beta1=0.65, beta2=0.6, weights_initialization=optimizer_param),
    # training_params = EarlyStopping(alpha=15,
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
    epochs=100,
    batch_size=64,  # If batch size equals 1, we have online learning
    shuffle_training_data=True,
)


# plots = Plots(nn_train)
# plots.plot_epoch_error(save_dir="cancer")
# plots.plot_epoch_accuracy(save_dir="cancer1")
# plots.plot_confusion_matrix(save_dir="cancer2")

# Make a prediction
print(nn_train.predict(x_test[1]))
print(y_test[1])
