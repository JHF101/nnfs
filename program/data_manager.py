from nnfs.data_sources.proben1 import Proben1
from nnfs.data_sources.mnist import MNIST
import streamlit as st
import os
from nnfs.utils.utils import to_categorical

def proben1_manager():
    datasets_list = ['cancer', 'card', 'diabetes', 'gene', 'glass', 'heart', 'horse', 'mushroom', 'soybean', 'thyroid']
    proben_dataset = st.selectbox('Select a dataset', datasets_list)
    proben_number = int(st.number_input('Select one of the sub datasets', step=1))

    proben = Proben1()
    proben.download_data()
    proben.get_dataset_dirs()

    (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = proben.load_data(data_set_name=proben_dataset)[proben_number]
    filename = proben.get_filenames(data_set_name=proben_dataset)[proben_number]

    return x_train, y_train, x_validate, y_validate, x_test, y_test

def mnist_manager():

    mnist = MNIST()
    mnist.download_data()

    train_size = st.number_input('Select the number of samples you want to train on out of the 60000', 1, 60000)
    test_size = st.number_input('Select the number of samples you want to train on out of the 10000', 1, 10000)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    dataset_concat = {
        'train_size':train_size,
        'test_size':test_size
    }
    # return x_train[:train_size], y_train[:train_size], None, None, x_test[:test_size], y_test[:test_size]
    return (x_train[:train_size], y_train[:train_size], None, None, x_test[:test_size], y_test[:test_size]), dataset_concat

def data_manager():
    """Manages the selection and data manipulations that happens to the
    dataset to be passed into the model.

    Returns
    -------
    result : Dict()
        Dictionary containing the dataset, input and output based on features
        and labels
    """
    # TODO : Add ability to normalize datasets or do other manipulations

    # TODO: Add more datasets
    datasets = {
        'proben1': 0,
        'mnist' : 1,
    }

    datasets_selector = st.selectbox('Select a DataSource', datasets.keys())

    with st.form(key='datasets_forms'):
        if datasets_selector=='proben1':
            dataset = proben1_manager()
        elif datasets_selector=='mnist':
            dataset, dataset_concat = mnist_manager()
        submit_button = st.form_submit_button(label='Submit')

    (x_train, y_train, x_validate, y_validate, x_test, y_test) = dataset

    # -------------------------------------------------------------------- #
    #                         START DATASET PREP                           #
    # -------------------------------------------------------------------- #
    if datasets_selector == 'mnist':

        x_train = x_train.reshape(dataset_concat['train_size'], 1, 28*28)
        x_test = x_test.reshape(dataset_concat['test_size'], 1, 28*28)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # --- Input and output layer size
        input_layer_size = 28*28#x_train[0].shape[0]
        output_layer_size = 10 #y_train[0].shape[0]

    if datasets_selector == 'proben1':
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])

        x_validate = x_validate.reshape(x_validate.shape[0], 1, x_validate.shape[1])
        y_validate = y_validate.reshape(y_validate.shape[0], 1, y_validate.shape[1])

        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])

    # TODO: Add the ability to normalize the data
    # normalizing_factor_x = x_train.max()
    # normalizing_factor_y = y_train.max()

    # x_train /= normalizing_factor_x
    # y_train /= normalizing_factor_y

    # x_validate /= normalizing_factor_x
    # y_validate /= normalizing_factor_y

    # x_test /= normalizing_factor_x
    # y_test /= normalizing_factor_y

        # --- Input and output layer size
        input_layer_size = x_train[0].shape[1]
        output_layer_size = y_train[0].shape[1]

    result = {
        "x_train":x_train,
        "y_train":y_train,
        "x_validate":x_validate,
        "y_validate":y_validate,
        "x_test":x_test,
        "y_test":y_test,
        "input_layer_size":input_layer_size,
        "output_layer_size":output_layer_size
    }

    return result