from glob import glob
import os
import numpy as np
import wget
import re

class Proben1:

    def __init__(self):
        self.WORKING_DIR = os.getcwd() + '/'
        self.DATASET_DIR = self.WORKING_DIR+'proben1/'
    
    def download_data(self):
        import glob
        import zipfile
        # Get the github link
        wget.download('https://github.com/jeffheaton/proben1/archive/refs/heads/master.zip')
        file_names = glob.glob('*.zip')
        try:
            os.mkdir(self.DATASET_DIR)
        except:
            print("The file location already exists")

        for file in file_names:
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(self.DATASET_DIR)
        # Removing .zip file
        files = glob.glob(self.WORKING_DIR + '*.zip')
        for f in files:
            os.remove(f)

    def get_dataset_dirs(self):
        
        # Find all of the directories in here
        directories = [x[0] for x in os.walk(self.WORKING_DIR)]

        # Find all of the dt files
        file_dirs = [] # Array of .dt files with their directory on the computer
        for dir in directories:
            if 'proben1' in dir:
                file_names = glob(dir+'/*.dt')
                # Checking to see if the directory contains any dt files
                if len(file_names) > 0:
                    file_dirs.append(
                        {
                            "directory":file_names,
                            "dataset": os.path.split(dir)[1],
                            "path": dir,
                        }
                    )
        
        self.file_dirs = file_dirs
    
    def get_filenames(self, data_set_name):
        filenames = []
        for dirs in self.file_dirs:
            if data_set_name in dirs['dataset']:
                for i in range(len(dirs['directory'])):
                    filenames.append( os.path.basename(dirs['directory'][i]))
        return filenames

    def load_data(self, data_set_name):
        # Returns an array of tuples of the processed data 
        # The user can the choose which element of the array that 
        # They would like to use for training
        for dirs in self.file_dirs:
            if data_set_name in dirs['dataset']:
                return self.file_parser(dirs['directory'])        

    def file_parser(self, file_names):
        """Gets files that correspond to the dataset file types and
        the proceeds to return the data in a standardized format similar
        to that of keras.datasets

        Parameters
        ----------
        file_names : Str
            The filenames of the dataset of interest.

        References
        ----------
        https://stackoverflow.com/questions/15502619/correctly-reading-text-from-windows-1252cp1252-file-in-python#:~:text=import%20os%0Aimport,lines)%0A%20%20%20%20%20%20%20%20f2.close()
        """
        data_set_output_array = []
        # Can take any list of filenames which can then be used
        for j in range(len(file_names)):
            with open(file_names[j], 'r', encoding='cp1252') as f:
                read_line = f.read().split('\n') # Splits the file into lines

            idx_counter = 0
            data_set_counter = 0 # Used to structure the training, validation and test splits
            data_state = True
            x_train_temp, y_train_temp, x_validate_temp, y_validate_temp, x_test_temp, y_test_temp = [], [], [], [], [], []
            for t in range(0, len(read_line)):

                # --------------------------------------------------- #
                #           Determine Dataset Structure               #
                # --------------------------------------------------- #
                if (idx_counter <= 6):
                    structure_value = int(read_line[t].split('=')[-1]) 
                    if idx_counter == 0:
                        bool_in = structure_value
                        print("Bool_in",bool_in)
                    elif idx_counter == 1:
                        real_in = structure_value
                        print("Real_in",real_in)
                    elif idx_counter == 2:
                        bool_out = structure_value
                        print("Bool_out",bool_out)
                    elif idx_counter == 3:
                        real_out = structure_value
                        print("Real_out",real_out)
                    elif idx_counter == 4:
                        training_examples = structure_value
                        print("Training Examples",training_examples)
                    elif idx_counter == 5:
                        validation_examples = structure_value
                        print("Validation Examples",validation_examples)
                    elif idx_counter == 6:
                        test_examples = structure_value
                        print("Test Examples",test_examples)

                # --------------------------------------------------- #
                #           Structure the Dataset                     #
                # --------------------------------------------------- #
                else:
                    if data_state:
                        # Inputs
                        data_set_input_length = 0
                        if (bool_in != 0):
                            print("It has boolean inputs")
                            data_set_input_length = bool_in
                        elif (real_in != 0):
                            print("It has Real valued inputs")
                            data_set_input_length = real_in
                        else: 
                            raise Exception("There is no input data length")

                        data_set_output_length = 0
                        if (bool_out != 0):
                            print("It has boolean outputs")
                            data_set_output_length = bool_out
                        elif (real_out != 0):
                            print("It has Real valued outputs")
                            data_set_output_length = real_out
                        else: 
                            raise Exception("There is no output data length")
                        data_state = False

                    # Splitting by removing the spaces
                    data = re.split("(?<=\\S) ", read_line[t])

                    if data == ['']:
                        continue

                    for check in data:
                        if (check=='') or (check==' '):
                            print(data)
                            raise Exception(f"Space found in {data_set_counter} of dt file")

                    data = [float(i) for i in data]

                    # Split data into x and y
                    x_data = data[0:data_set_input_length]
                    y_data = data[data_set_input_length:]

                    # Checks
                    if len(y_data)!=data_set_output_length:
                        raise Exception(f"The length of y_data {len(y_data)}!= {data_set_output_length}")

                    if (data_set_counter < training_examples):
                        x_train_temp.append(x_data)
                        y_train_temp.append(y_data)
                    elif (training_examples)<=data_set_counter<(training_examples+validation_examples):
                        x_validate_temp.append(x_data)
                        y_validate_temp.append(y_data)
                    elif (training_examples+validation_examples)<=data_set_counter<(training_examples+validation_examples+test_examples):
                        x_test_temp.append(x_data)
                        y_test_temp.append(y_data)
                    else:
                        raise Exception(f"The index {data_set_counter} is greater than {training_examples+validation_examples+test_examples}")

                    data_set_counter += 1

                idx_counter += 1 

            # Dataset returned in same format keras datasets
            data_set_output_array.append(
                    [(
                        np.array(x_train_temp), np.array(y_train_temp)),
                    (
                        np.array(x_validate_temp), np.array(y_validate_temp)),
                    (
                        np.array(x_test_temp), np.array(y_test_temp))]
                )

        return data_set_output_array


