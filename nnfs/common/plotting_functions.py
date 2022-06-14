import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sn
import numpy as np
import streamlit as st

plt.style.use('fivethirtyeight')

from nnfs.utils.logs import create_logger
log = create_logger(__name__)

def random_color():
    rgbl=[255,0,0]
    np.random.shuffle(rgbl)
    return tuple(rgbl)

class Plots:
    def __init__(self):
        self.fig_error, self.ax_error = plt.subplots(figsize=(16, 9))
        self.st_err = st.pyplot(self.fig_error)

        self.fig_accuracy, self.ax_accuracy = plt.subplots(figsize=(16, 9))
        self.st_acc = st.pyplot(self.fig_accuracy)

        self.fig_confusion, self.ax_confusion = plt.subplots(figsize=(16, 9))
        self.st_confusion = st.pyplot(self.fig_confusion)

    def update_data(self, self_data):
        self.optimizer = self_data.optimizer
        self.weights = self_data.weights
        self.activation_functions = self_data.activation_functions
        self.error_function = self_data.error_function
        self.epoch_error_testing_plot = self_data.epoch_error_testing_plot
        self.epoch_error_training_plot = self_data.epoch_error_training_plot
        self.epoch_error_validation_plot = self_data.epoch_error_validation_plot
        self.optimal_error_position = self_data.optimal_error_position

    def plot_epoch_error(self, ds_name, save_dir):
        if (self.optimizer.optimizer_type == 'gradient'):
            # Naming
            architecture = ''
            for i in range(0,len(self.weights)):
                                                                    # Activation function
                architecture += str(self.weights[i].shape[1]) + "-" + str(self.activation_functions[i+1].__name__)[0] + "+"
            architecture = architecture[:-1] # Remove to the last plus
            architecture += " " + self.error_function.__name__
            # if (self.learning_rate>0):
            #     architecture += " " + "lr: " + str(self.learning_rate)
            # Creating indeces
            epochs_idx = [i+1 for i in range(len(self.epoch_error_testing_plot))]

            self.ax_error.plot(epochs_idx, self.epoch_error_testing_plot, 'b', label="Testing")

            self.ax_error.plot(epochs_idx, self.epoch_error_training_plot, 'r', label="Training")

            val_set_len = len(self.epoch_error_validation_plot)
            test_set_len = len(self.epoch_error_training_plot)
            if (val_set_len > 0) and (test_set_len==val_set_len):
                self.ax_error.plot(epochs_idx, self.epoch_error_validation_plot, 'g', label="Validation")
                # if self.early_stopping is not None:
                print("optimal error",(self.optimal_error_position))
                if self.optimal_error_position>0:
                    self.ax_error.plot(self.optimal_error_position,
                            self.epoch_error_validation_plot[self.optimal_error_position-1],
                            'ko',
                            label="Min validation error",
                            markersize=12,linewidth=2)

            self.ax_error.set_title(ds_name+' '+str.capitalize(self.optimizer.optimizer_name)+' '+architecture)
            self.ax_error.set_ylabel("Error")
            self.ax_error.set_xlabel("Epochs")
            self.ax_error.legend()
            self.st_err.empty()
            self.st_err = st.pyplot(self.fig_error)
            self.ax_error.clear()

    def plot_epoch_accuracy(self, ds_name, save_dir):
        if (self.optimizer.optimizer_type == 'gradient'):
            # Naming
            architecture = ''
            for i in range(0, len(self.weights)):
                                                                    # Activation function
                architecture += str(self.weights[i].shape[1]) + "-" + str(self.activation_functions[i+1].__name__)[0] + "+"
            architecture = architecture[:-1] # Remove to the last plus
            architecture += " "+self.error_function.__name__

            # if (self.learning_rate>0):
            #     architecture += " " + "lr: " + str(self.learning_rate)
            epochs_idx = [i+1 for i in range(len(self.epoch_error_testing_plot))]

            self.ax_accuracy.plot(epochs_idx, self.epoch_testing_accuracy_plot,'b', label="Testing")

            self.ax_accuracy.plot(epochs_idx, self.epoch_training_accuracy_plot, 'r', label="Training")

            val_set_len = len(self.epoch_error_validation_plot)
            test_set_len = len(self.epoch_error_training_plot)

            if (val_set_len > 0) and (test_set_len == val_set_len):
                self.ax_accuracy.plot(

                    epochs_idx,
                    self.epoch_validation_accuracy_plot,
                    'g',
                    label="Validation")
                # if self.early_stopping is not None:
                if self.optimal_error_position>0:
                    self.ax_accuracy.plot(
                        self.optimal_error_position,
                        self.epoch_testing_accuracy_plot[self.optimal_error_position-1],
                        'ko',
                        label="Min validation error",
                        markersize=12,linewidth=2)
        else:
            architecture = ''
            for i in range(0, len(self.weights)):
                # Getting the structure of weights
                architecture += str(self.weights[0][i].shape[1]) + "-" + str(self.activation_functions[i+1].__name__)[0] + "+"
            architecture = architecture[:-1] # Remove to the last plus
            # architecture += " "+self.error_function.__name__

            epochs_idx = [i for i in range(len(self.epoch_testing_accuracy_plot))]

            for i in range(1):#len(self.epoch_error_testing_plot[0])):
                temp_arr1 = []

                for j in range(len(self.epoch_testing_accuracy_plot)):
                    temp_arr1.append(self.epoch_testing_accuracy_plot[j][i])

                self.ax_accuracy.plot(epochs_idx, temp_arr1, random_color(), label=f"Testing {i}")

            # for i in range(1):#len(self.epoch_training_accuracy_plot[0])):
            #     temp_arr2 = []
            #     for j in range(len(self.epoch_training_accuracy_plot)):
            #         temp_arr2.append(self.epoch_training_accuracy_plot[j][i])
            #     log.warn(temp_arr2)

            #     plt.plot(epochs_idx, temp_arr2, 'r',  label=f"Training {i}")

            # val_set_len = len(self.epoch_testing_accuracy_plot)
            # test_set_len = len(self.epoch_error_training_plot)
            # if (val_set_len > 0) and (test_set_len==val_set_len):
            #     for i in range(1):#len(self.epoch_testing_accuracy_plot[0])):
            #         temp_arr2 = []
            #         for j in range(len(self.epoch_testing_accuracy_plot)):
            #             temp_arr2.append(self.epoch_testing_accuracy_plot[j][i])
            #         log.warn(temp_arr2)

            #         plt.plot(epochs_idx, temp_arr2, random_color(),  label=f"Training {i}")


        self.ax_accuracy.set_title(ds_name+' '+str.capitalize(self.optimizer.optimizer_name)+' '+architecture)
        self.ax_accuracy.set_ylabel("Accuracy")
        self.ax_accuracy.set_xlabel("Epochs")
        self.ax_accuracy.set_ylim(0,1)
        self.ax_accuracy.legend()
        self.st_acc.empty()
        self.st_acc = st.pyplot(self.fig_accuracy)
        self.ax_accuracy.clear()

    # @staticmethod
    def plot_confusion_matrix(self, ds_name, save_dir):
        df = pd.DataFrame(self.confusion_matrix[-1])
        # sn.set(font_scale=1.4)
        conf = sn.heatmap(df, annot=True, annot_kws={"size": 16}, fmt='g', ax=self.ax_confusion, cbar=False)
        self.ax_confusion.set_title(f'Confusion Matrix for {ds_name}')
        self.ax_confusion.set_xlabel("Predictions")
        self.ax_confusion.set_ylabel("Actual Values")

        self.st_confusion.empty()
        self.st_confusion = st.pyplot(self.fig_confusion)
        self.ax_confusion.clear()
        
