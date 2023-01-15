from data_sources.proben1 import Proben1

proben = Proben1()
proben.get_dataset_dirs()

(x_train, y_train), (x_validate, y_validate), (x_test, y_test) = proben.load_data(
    data_set_name="cancer"
)[2]
filename = proben.get_filenames(data_set_name="cancer")[2]
