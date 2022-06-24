from modulefinder import Module
import streamlit as st

import numpy as np
from data_manager import data_manager
from nn_visualizer import DrawNN

from nn_manager import (
    early_stop_manager,
    error_func_manager,
    initializer_manager,
    hidden_layer_manager,
    input_layer_manager,
    neural_network_manager,
    optimizer_manager,
    output_layer_manager,
    training_manager,
    error_func_select
)

# st.set_page_config(layout="wide")

# ---------------------------------- #
#               SideBar              #
# ---------------------------------- #

with st.sidebar:
    # Data Sources
    with st.form(key='data_set_form'):
        # Dataset selection
        datasets = data_manager()
        # ----------------------------- #
        #      Network Construction     #
        # ----------------------------- #
        submit_button = st.form_submit_button(label='Submit')

    with st.form(key='error_func_form'):
        error_func = error_func_manager()
        actual_error_func =error_func_select[error_func]
        submit_button = st.form_submit_button(label='Submit')

    initializer = initializer_manager()

    # with st.form(key='optimizer_form'):
    optimizer = optimizer_manager(initializer)

    training_params = early_stop_manager()

    with st.form(key='hidden_layer_form'):
        num_of_hidden_layers = int(st.number_input('Number of hidden layers', step=1))
        submit_button = st.form_submit_button(label='Submit')

    with st.form(key='architecture_form'):
        col1, col2 = st.columns(2)
        layers = []

        # Input Layers
        layers.append(input_layer_manager(datasets['input_layer_size'], col1, col2))

        # Hidden Layers
        for i in range(num_of_hidden_layers):
            layers.append(hidden_layer_manager(i+1, col1, col2))

        # Output Layer
        layers.append(output_layer_manager(datasets['output_layer_size'], len(layers)+1, col1, col2))
        submit_button = st.form_submit_button(label='Submit')

    with st.form(key='use_bias_form'):
        use_bias = st.checkbox("Use bias")
        submit_button = st.form_submit_button(label='Submit')

    with st.form(key='gen_plots_form'):
        generate_plots = st.checkbox("Generate plots")
        submit_button = st.form_submit_button(label='Submit')

if st.button("Show Neural Network Architecture"):
    network=DrawNN([i[0] for i in layers])
    network.draw()


with st.form(key='confirm_nn_form'):

    nn_train = neural_network_manager(
        layers=layers,
        error_func=actual_error_func,
        bias=use_bias,
        training_params=training_params,
        optimizer=optimizer,
        generate_plots=generate_plots
    )
    submit_button = st.form_submit_button(label='Submit')


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