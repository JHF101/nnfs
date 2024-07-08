from nnfs.data_sources.proben1 import Proben1
from nnfs.data_sources.mnist import MNIST
import streamlit as st
from nnfs.utils.utils import to_categorical, min_max_scaler, standard_scaler
import numpy as np


def proben1_manager():
    datasets_list = [
        "cancer",
        "card",
        "diabetes",
        "gene",
        "glass",
        "heart",
        "horse",
        "mushroom",
        "soybean",
        "thyroid",
    ]
    proben_dataset = st.selectbox("Select a dataset", datasets_list)
    proben_number = int(st.number_input("Select one of the sub datasets", step=1))

    proben = Proben1()
    proben.download_data()
    proben.get_dataset_dirs()

    (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = proben.load_data(
        data_set_name=proben_dataset
    )[proben_number]
    _ = proben.get_filenames(data_set_name=proben_dataset)[proben_number]

    return x_train, y_train, x_validate, y_validate, x_test, y_test


def mnist_manager():

    mnist = MNIST()

    train_size = st.number_input(
        "Select the number of samples you want to train on out of the 60000", 1, 60000
    )
    test_size = st.number_input(
        "Select the number of samples you want to train on out of the 10000", 1, 10000
    )

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    dataset_concat = {"train_size": train_size, "test_size": test_size}
    # return x_train[:train_size], y_train[:train_size], None, None, x_test[:test_size], y_test[:test_size]
    return (
        x_train[:train_size],
        y_train[:train_size],
        None,
        None,
        x_test[:test_size],
        y_test[:test_size],
    ), dataset_concat


def data_manager():
    """Manages the selection and data manipulations that happens to the
    dataset to be passed into the model.

    Returns
    -------
    result : Dict()
        Dictionary containing the dataset, input and output based on features
        and labels
    """

    # TODO: Add more datasets
    datasets = {
        "proben1": {"id": 0, "url": "https://github.com/jeffheaton/proben1"},
        "mnist": {
            "id": 1,
            "url": "http://yann.lecun.com/exdb/mnist/",
        },  # If the site is back online
    }

    options = [key if key != "mnist" else "mnist (disabled)" for key in datasets.keys()]
    datasets_selector = st.selectbox("Select a DataSource", options)

    if datasets_selector == "mnist (disabled)":
        st.warning(
            "The MNIST dataset is currently disabled because the site is offline."
        )
        st.stop()  # Stops the execution of the rest of the code

    selected_dataset = (
        datasets_selector if datasets_selector != "mnist (disabled)" else "mnist"
    )

    # Display dataset link
    dataset_url = datasets[selected_dataset]["url"]
    st.markdown(f"[Go to {selected_dataset} dataset site]({dataset_url})")

    with st.form(key="datasets_forms"):
        if datasets_selector == "proben1":
            dataset = proben1_manager()
        elif datasets_selector == "mnist":
            dataset, dataset_concat = mnist_manager()

        _ = st.form_submit_button(label="Submit")

    (x_train, y_train, x_validate, y_validate, x_test, y_test) = dataset

    # Combine all data splits for scaling
    combined_data = np.concatenate((x_train, x_validate, x_test), axis=0)

    scalers = {
        "None": 0,
        "MinMax": 1,
        "Standard": 1,
    }
    scaler = st.selectbox("Select a Scaling Method for the Dataset", scalers.keys())

    if scaler != "None":
        # Apply scaler if specified
        if scaler == "MinMax":
            combined_data = min_max_scaler(combined_data)
        elif scaler == "Standard":
            combined_data = standard_scaler(combined_data)

        # Split back into individual data splits
        split_idx1 = len(x_train)
        split_idx2 = len(x_train) + len(x_validate)

        x_train = combined_data[:split_idx1]
        x_validate = combined_data[split_idx1:split_idx2]
        x_test = combined_data[split_idx2:]

    # -------------------------------------------------------------------- #
    #                         START DATASET PREP                           #
    # -------------------------------------------------------------------- #
    if datasets_selector == "mnist":

        x_train = x_train.reshape(dataset_concat["train_size"], 1, 28 * 28)
        x_test = x_test.reshape(dataset_concat["test_size"], 1, 28 * 28)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # --- Input and output layer size
        input_layer_size = 28 * 28  # x_train[0].shape[0]
        output_layer_size = 10  # y_train[0].shape[0]
        dataset_type = "image"

    elif datasets_selector == "proben1":
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])

        x_validate = x_validate.reshape(x_validate.shape[0], 1, x_validate.shape[1])
        y_validate = y_validate.reshape(y_validate.shape[0], 1, y_validate.shape[1])

        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])
        dataset_type = "numerical"

        # --- Input and output layer size
        input_layer_size = x_train[0].shape[1]
        output_layer_size = y_train[0].shape[1]

    result = {
        "x_train": x_train,
        "y_train": y_train,
        "x_validate": x_validate,
        "y_validate": y_validate,
        "x_test": x_test,
        "y_test": y_test,
        "input_layer_size": input_layer_size,
        "output_layer_size": output_layer_size,
        "dataset_type": dataset_type,
    }

    return result
