import os
import numpy as np
import codecs
import wget
import gzip
import glob
import shutil

from nnfs.utils.logs import create_logger
log = create_logger(__name__)

class MNIST:
    def __init__(self):
        self.WORKING_DIR = os.getcwd() + '/'
        self.DATASET_DIR = self.WORKING_DIR+'mnist/'

    def byte_to_int(self, byte):
        return int(codecs.encode(byte, 'hex'), 16)

    def download_data(self):
        try:
            os.mkdir(self.DATASET_DIR)
            wget.download('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', out=self.DATASET_DIR)
            wget.download('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', out=self.DATASET_DIR)
            wget.download('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', out=self.DATASET_DIR)
            wget.download('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', out=self.DATASET_DIR)
            log.info("Making directory")
        except:
            log.warning("The file location already exists")

        file_names = glob.glob(self.DATASET_DIR+'*.gz')
        for file in file_names:
            with gzip.open(file, 'rb') as f_in:
                with open(file[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        # Removing .gz file
        files = glob.glob(self.WORKING_DIR + '*.gz')
        for f in files:
            log.info(f"Deleting {f}")
            os.remove(f)

    def load_data(self):
        """
        Loads the MNIST Data into the directory specified

        Parameters
        ----------
        directory : str
            The location in which the dataset should be saved

        Returns
        -------
        tuple(tuples)
            Returns the MNIST dataset in the that tensorflow does:
                (x train, y train), (x test, y test)
        """
        training_set_label_file = test_set_label_file = 2049
        training_set_image_file = test_set_image_file = 2051

        training_set_size = 60000
        test_set_size = 10000

        # If file does not exist then download the data
        if not (os.path.exists(self.DATASET_DIR)):
            log.info("Downloading the dataset")
            self.download_data()

        # Determine way to get this
        files = os.listdir(self.DATASET_DIR)

        # Create a dictionary to store train images, train labels, test images and test labels
        dataset_dict = {}
        for file in files:
            if file.endswith('ubyte'):
                with open(self.DATASET_DIR+file, 'rb') as f:
                    data = f.read()
                    magic_number = self.byte_to_int(data[0:4]) # Image/Label
                    length_of_arr = self.byte_to_int(data[4:8]) # Length of array

                    # Just for completeness
                    if ((magic_number == training_set_image_file) or (magic_number == test_set_image_file)):
                        data_type = 'images'
                        row_count = self.byte_to_int(data[8:12]) # Number of rows
                        col_count = self.byte_to_int(data[12:16]) # Number of columns
                        parsed_data = np.frombuffer(data, dtype=np.uint8, offset=16) # read the pixel data
                        parsed_data = parsed_data.reshape(length_of_arr, row_count, col_count)

                    # Just for completeness
                    elif ((magic_number == training_set_label_file) or (magic_number == test_set_label_file)):
                        data_type = 'labels'
                        parsed_data = np.frombuffer(data, dtype=np.uint8, offset=8) # read the label data
                        parsed_data = parsed_data.reshape(length_of_arr)

                    if (length_of_arr == test_set_size):
                        set_type = 'test'
                    elif (length_of_arr == training_set_size):
                        set_type = 'train'
                    dataset_dict[set_type + '_' + data_type] = parsed_data

        # Structure from how tensorflow does it
        return ((dataset_dict['train_images'], dataset_dict['train_labels']),
                (dataset_dict['test_images'], dataset_dict['test_labels']))


