from nnfs.data_sources.proben1 import Proben1
import streamlit as st

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

# def mnist_manager():

def data_manager():

    datasets = {
        'proben1':proben1_manager(),
    }
    datasets_selector = st.selectbox('Select a DataSource', datasets.keys())

    x_train, y_train, x_validate, y_validate, x_test, y_test = datasets[datasets_selector]

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