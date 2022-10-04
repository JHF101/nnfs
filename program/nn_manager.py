"""
Neural Network Logic
"""
from inspect import getmembers, isfunction, isclass, getargspec
import nnfs
from nnfs.activations import activation_functions
from nnfs.activations.activation_functions import linear, relu, sigmoid, softmax, tanh, elu, leaky_relu
from nnfs.common.early_stopping import EarlyStopping
from nnfs.errors.error_functions import mse, rms, squared_error, cross_entropy
from nnfs.errors import error_functions

from nnfs.neural_network.neural_network import Network
from nnfs.neural_network.optimizers.gradient.delta_bar_delta import DeltaBarDelta
from nnfs.neural_network.optimizers.gradient.gradient_descent import GradientDescent
from nnfs.neural_network.optimizers.gradient.gradient_descent_momentum import \
    GradientDescentWithMomentum
from nnfs.neural_network.optimizers.gradient.rprop import Rprop
from nnfs.neural_network.optimizers.gradient.rms_prop import RMSProp
from nnfs.neural_network.optimizers.gradient.adam import Adam
from nnfs.neural_network.optimizers import gradient, non_gradient
from nnfs.neural_network.optimizers.non_gradient.genetic import GeneticOptimizer
import streamlit as st

activation_func_select = {
    'tanh':tanh,
    'sigmoid':sigmoid,
    'relu':relu,
    'elu':elu,
    'leaky_relu':leaky_relu,
    'softmax':softmax,
    'linear':linear,
}

error_func_select = {
    'mse':mse,
    'rms':rms,
    'squared_error':squared_error,
    'cross_entropy':cross_entropy
}

optimizer_func_select = {
    'adam': Adam,
    'delta_bar_delta':DeltaBarDelta,
    'gradient_descent':GradientDescent,
    'gradient_descent_momentum':GradientDescentWithMomentum,
    'rms_prop':RMSProp,
    'rprop':Rprop,
    'genetic': GeneticOptimizer
}

optimizer_func_defaults = {
    'adam': {
        'learning_rate':0.01,
        'beta1':0.65,
        'beta2':0.6
    },
    'delta_bar_delta':{
        'theta':0.1,
        'mini_k':0.01,
        'phi':0.1
    },
    'gradient_descent':{
        'learning_rate':0.5
    },
    'gradient_descent_momentum':{
        'learning_rate':0.09,
        'beta':0.9
    },
    'rms_prop':{
        'learning_rate':0.001,
        'beta':0.99
    },
    'rprop':{
        'delta_max':50,
        'delta_min':0,
        'eta_plus':1.1,
        'eta_minus':0.5
    },
    'genetic': {
        'number_of_parents':4,
        'fitness_eval':'accuracy'
        }
}

# optimizer = GeneticOptimizer(number_of_parents=4,
#                             fitness_eval='accuracy',
#                             weights_initialization=optimizer_param),

# optimizer= GradientDescent(learning_rate=0.5, weights_initialization=optimizer_param),

# optimizer= GradientDescentWithMomentum(learning_rate=0.09, beta=0.9, weights_initialization=optimizer_param),

# optimizer= DeltaBarDelta(theta=0.1, mini_k=0.01, phi=0.1, weights_initialization=optimizer_param),

# optimizer= Rprop(delta_max=50, delta_min=0, eta_plus=1.1, eta_minus=0.5, weights_initialization=optimizer_param),

# optimizer= RMSProp(learning_rate=0.001, beta=0.99, weights_initialization=optimizer_param),

# optimizer= Adam(learning_rate=0.01, beta1=0.65, beta2=0.6, weights_initialization=optimizer_param),

def activation_func_manager(layer_num, column):
    with column:
        # ----- Activation functions ----- #
        activation_dropdown = st.selectbox(
                f"Activation for layer {layer_num}",
                ([act[0] for act in getmembers(activation_functions, isfunction)])
        )
    return activation_dropdown

def error_func_manager():
    # ----- Activation functions ----- #
    error_dropdown = st.selectbox(
            f"Error function",
            ([err[0] for err in getmembers(error_functions, isfunction)])
    )
    return error_dropdown

# --------------------------------------- #
#            Start Optimizer              #
# --------------------------------------- #
def optimizer_manager(initializer):
    """Creates the optimizer for the model"""
    # ----- Optimizers ----- #
    optimizers = [cls_name for cls_name, cls_obj in getmembers(gradient) if '__' not in cls_name and cls_name!='gradient_optimizer']
    optimizers.extend([cls_name for cls_name, obj_type in getmembers(non_gradient) if '__' not in cls_name])
    optimizers_select = st.selectbox(
        "Choose a optimizer function",
        (optimizers)
    )

    optimizer_func = optimizer_func_select[optimizers_select]
    # Initializer optimizer params and the default optimizer params
    init_params = initializer_optimizer(optimizer_func, **optimizer_func_defaults[optimizers_select])

    # Initializer Optimizer over here
    if optimizers_select=='genetic':
        optimizer = GeneticOptimizer(
            number_of_parents=int(init_params['number_of_parents']),
            fitness_eval=str(init_params['fitness_eval']),
            weights_initialization=initializer)
    elif optimizers_select=='gradient_descent':
        optimizer= GradientDescent(
            learning_rate=float(init_params['learning_rate']),
            weights_initialization=initializer)
    elif optimizers_select=='gradient_descent_momentum':
        optimizer= GradientDescentWithMomentum(
            learning_rate=float(init_params['learning_rate']),
            beta=float(init_params['beta']),
            weights_initialization=initializer)
    elif optimizers_select=='delta_bar_delta':
        optimizer= DeltaBarDelta(
            theta=float(init_params['theta']),
            mini_k=float(init_params['mini_k']),
            phi=float(init_params['phi']),
            weights_initialization=initializer)
    elif optimizers_select=='rprop':
        optimizer= Rprop(
            delta_max=float(init_params['delta_max']),
            delta_min=float(init_params['delta_min']),
            eta_plus=float(init_params['eta_plus']),
            eta_minus=float(init_params['eta_minus']),
            weights_initialization=initializer)
    elif optimizers_select=='rms_prop':
        optimizer= RMSProp(
            learning_rate=float(init_params['learning_rate']),
            beta=float(init_params['beta']),
            weights_initialization=initializer)
    elif optimizers_select=='adam':
        optimizer= Adam(
            learning_rate=float(init_params['learning_rate']),
            beta1=float(init_params['beta1']),
            beta2=float(init_params['beta2']),
            weights_initialization=initializer)
    return optimizer

def initializer_optimizer(obj, **kwargs):
    """Generates all of the initial parameters to the model
    **kwargs : dict
        Contains all of default parameters
    """
    name_params = []
    all_name_params = getargspec(obj).args
    for p in range(len(all_name_params)):
        if (all_name_params[p] == 'self'):
            pass
        elif (all_name_params[p] == 'weights_initialization'):
            pass
        else:
            name_params.append(all_name_params[p])

    init_params = dict()
    for i in name_params:
        # If default parameters are not passed in for the optimizer it is set to 0
        init_params[i] = st.text_input(f"Optimizer parameter: {i}", kwargs[i] if i in kwargs.keys() else 0)

    return init_params
# --------------------------------------- #
#              End Optimizer              #
# --------------------------------------- #


def early_stop_manager():
    # ----- Early Stopping ----- #
    if st.checkbox("Use early stopping"):
        alpha_val = st.number_input('Select an alpha', 5.0)
        pkt_threshold_val = st.number_input('Pkt threshold', 15)
        k_epochs_val = st.number_input('K Epochs', 5.0)
        training_params = EarlyStopping(alpha=alpha_val,
                                        pkt_threshold=pkt_threshold_val,
                                        k_epochs=k_epochs_val)
    else:
        training_params = None
    return training_params

def initializer_manager():
    # ----- Initializers ----- #
    optimizer_selector = st.selectbox(
        "Choose an initializer",
        ('heuristic', 'xavier', 'he', 'none')
    )
    if optimizer_selector == 'heuristic':
        slider_vals = st.slider(
            'Select a range of values',
            -1.0, 1.0, (-0.3, 0.3))
        optimizer_param=dict(name='heuristic', lower=slider_vals[0], upper=slider_vals[1])
    elif optimizer_selector == 'xavier':
        optimizer_param=dict(name='xavier')
    elif optimizer_selector == 'he':
        optimizer_param=dict(name='he')
    else:
        optimizer_param=None
    return optimizer_param

# -------------------------- #
#        Architecture        #
# -------------------------- #
def input_layer_manager(layer_size, colum1, colum2):
    with colum1:
        layer_size = st.number_input(f"Input layer", value=layer_size, disabled=True)
    activation_function = activation_func_manager(0, colum2)
    act_function = activation_func_select[activation_function]
    return (layer_size, act_function)

def hidden_layer_manager(layer_num, colum1, colum2):
    # TODO: Add check to see if it is the first or last layer
    with colum1:
        layer_size = st.number_input(f"Hidden layer {layer_num}", 0, 1024, step=1)
    activation_function = activation_func_manager(layer_num, colum2)

    act_function = activation_func_select[activation_function]

    return (layer_size, act_function)

def output_layer_manager(layer_size, layer_num, colum1, colum2):
    with colum1:
        layer_size = st.number_input(f"Output layer", value=layer_size, disabled=True)
    activation_function = activation_func_manager(layer_num, colum2)
    act_function = activation_func_select[activation_function]
    return (layer_size, act_function)

def neural_network_manager(**kwargs):
    layers = kwargs['layers']
    error_func = kwargs['error_func']
    bias_val = kwargs['bias']
    training_params = kwargs['training_params']
    optimizer = kwargs['optimizer']
    plot_config = kwargs['plot_config']

    # Normal Gradient Descent
    nn_train = Network(
        layers=layers,
        error_function=error_func,
        use_bias=bias_val,
        optimizer=optimizer,
        training_params = training_params,
        plots_config=plot_config
    )
    return nn_train

def training_manager(nn_train, **kwargs):

    epochs=kwargs['epochs']
    batch_size=kwargs['batch_size']
    shuffle_training_data=kwargs['shuffle_training_data']

    x_train=kwargs['x_train']
    y_train=kwargs['y_train']
    x_validate=kwargs['x_validate']
    y_validate=kwargs['y_validate']
    x_test=kwargs['x_test']
    y_test=kwargs['y_test']

    # TODO: Add ability to change percentage of dataset which is validations set
    # Where graph is getting plotted
    nn_train.fit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_validate=x_validate,
        y_validate=y_validate,
        epochs=epochs,
        batch_size=batch_size,
        shuffle_training_data=shuffle_training_data,
    )
    # nn_train.fit(**kwargs)

    return nn_train
