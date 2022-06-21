from modulefinder import Module
import streamlit as st

import numpy as np
from data_manager import data_manager

from nn_manager import early_stop_manager, error_func_manager, initializer_manager, hidden_layer_manager, input_layer_manager, neural_network_manager, optimizer_manager, output_layer_manager, training_manager


st.set_page_config(layout="wide")

# ---------------------------------- #
#               SideBar              #
# ---------------------------------- #

with st.sidebar:
    # Data Sources

    # Dataset selection
    datasets = data_manager()
    # ----------------------------- #
    #      Network Construction     #
    # ----------------------------- #

    error_func = error_func_manager()
    initializer = initializer_manager()

    optimizer = optimizer_manager(initializer)

    training_params = early_stop_manager()

    num_of_hidden_layers = int(st.number_input('Number of hidden layers', step=1))

    col1, col2 = st.columns(2)
    layers = []

    # Input Layers
    layers.append(input_layer_manager(datasets['input_layer_size'], col1, col2))

    # Hidden Layers
    for i in range(num_of_hidden_layers):
        layers.append(hidden_layer_manager(i+1, col1, col2))

    # Output Layer
    layers.append(output_layer_manager(datasets['output_layer_size'], len(layers)+1, col1, col2))

    use_bias = st.checkbox("Use bias")

    generate_plots = st.checkbox("Generate plots")

    nn_train = neural_network_manager(
        layers=layers,
        error_func=error_func,
        bias=use_bias,
        training_params=training_params,
        optimizer=optimizer,
        generate_plots=generate_plots
    )

    # TODO: Import library based on it's name
        # TODO: Get the input parameters that are required

if st.button('Train the model'):
    training_manager(
        nn_train,
        **datasets
    )


# TODO: Add button to save model

# TODO: Add text input to load in model

# TODO: Add table returning the result of the predictions of the models