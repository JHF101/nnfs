from modulefinder import Module
import streamlit as st
from itertools import count

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

def counter(_count=count(1)):
    return next(_count)

# ---------------------------------- #
#               SideBar              #
# ---------------------------------- #
with st.sidebar:
    # Data Sources
    st.write(f'Step {counter()}: Select a data source.')
    # Dataset selection
    datasets = data_manager()

    # ----------------------------- #
    #      Network Construction     #
    # ----------------------------- #
    with st.form(key='batch_size'):
        st.write(f'Step {counter()}: Select batch size.')
        batch_size = st.number_input("Select batch size", 1, step=1)
        submit_button = st.form_submit_button(label='Submit')

    with st.form(key='epochs'):
        st.write(f'Step {counter()}: Select number of epochs the model should be trained for.')
        epochs = st.number_input("Select number of epochs.", 1, step=1)
        submit_button = st.form_submit_button(label='Submit')

    with st.form(key='error_func_form'):
        st.write(f'Step {counter()}: Select a loss/error function for the network.')
        error_func = error_func_manager()
        actual_error_func =error_func_select[error_func]
        submit_button = st.form_submit_button(label='Submit')

    st.write(f'Step {counter()}: Select a weight initializer for the network.')
    initializer = initializer_manager()

    st.write(f'Step {counter()}: Select an optimizer for the network.')
    # with st.form(key='optimizer_form'):
    optimizer = optimizer_manager(initializer)

    st.write(f'Step {counter()}: Select if early stopping should be used.')
    training_params = early_stop_manager()

    with st.form(key='hidden_layer_form'):
        st.write(f'Step {counter()}: Select the number of hidden layers to be used by the model')
        num_of_hidden_layers = int(st.number_input('Number of hidden layers', step=1))
        submit_button = st.form_submit_button(label='Submit')

    with st.form(key='architecture_form'):
        st.write(f'Step {counter()}: Define the architecture of the network, by selecting the number of neurons in the hidden layer and the activation functions used for that layer.')
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
        st.write(f'Step {counter()}: Select if bias should be used in the network.')
        use_bias = st.checkbox("Use bias")
        submit_button = st.form_submit_button(label='Submit')

    with st.form(key='shuffle_training_data'):
        st.write(f'Step {counter()}: Select whether training data should be shuffled per epoch.')
        shuffle_training_data = st.checkbox("Shuffle training data?")
        submit_button = st.form_submit_button(label='Submit')

    with st.form(key='gen_plots_form'):
        st.write(f'Step {counter()}: Select whether plots should be generated while training or not.')
        generate_plots = st.checkbox("Generate plots")
        submit_button = st.form_submit_button(label='Submit')

if st.button("Show Neural Network Architecture"):
    network=DrawNN([i[0] for i in layers])
    network.draw()


with st.form(key='confirm_nn_form'):

    st.write("Please make sure that the following is correct before submitting:")
    # st.write(f"Layers = {[i, v.__name__ for i,v in zip(layers)]}")
    st.write(f"Error Function = {error_func}")
    st.write(f"Batch size = {batch_size}")
    st.write(f"Number of epochs = {epochs}")
    st.write(f"Bias usage = {use_bias}")
    st.write(f"Optimizer = {optimizer}")
    st.write(f"Generate Plots = {generate_plots}")
    st.write(f"Shuffle Training Data = {shuffle_training_data}")


    nn_train = neural_network_manager(
        layers=layers,
        error_func=actual_error_func,
        bias=use_bias,
        training_params=training_params,
        optimizer=optimizer,
        generate_plots=generate_plots
    )

    submit_button = st.form_submit_button(label='Submit')

if st.button('Train the model'):
    st.write(f'Step {counter()}: Start training the model when all of the parameters have been set')

    training_manager(
        nn_train,
        batch_size=batch_size,
        shuffle_training_data=shuffle_training_data,
        epochs=epochs,
        **datasets
    )


# TODO: Add button to save model

# TODO: Add text input to load in model

# TODO: Add table returning the result of the predictions of the models