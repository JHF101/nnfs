import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sn
import numpy as np

plt.style.use('fivethirtyeight')

from utils.logs import create_logger
log = create_logger(__name__)

def random_color():
    rgbl=[255,0,0]
    np.random.shuffle(rgbl)
    return tuple(rgbl)

class Plots:
    def __init__(self, object):
        self.obj = object
        # pass

    def plot_epoch_error(self, ds_name, save_dir):
        if (self.obj.optimizer.optimizer_type == 'gradient'):
            # Naming
            figure(figsize=(16, 9), dpi=80)
            architecture = ''
            for i in range(0,len(self.obj.weights)):                    
                                                                    # Activation function 
                architecture += str(self.obj.weights[i].shape[1]) + "-" + str(self.obj.activation_functions[i+1].__name__)[0] + "+" 
            architecture = architecture[:-1] # Remove to the last plus
            architecture += " " + self.obj.error_function.__name__  
            # if (self.obj.learning_rate>0):
            #     architecture += " " + "lr: " + str(self.obj.learning_rate)
            # Creating indeces
            epochs_idx = [i+1 for i in range(len(self.obj.epoch_error_testing_plot))]

            plt.plot(epochs_idx, self.obj.epoch_error_testing_plot, 'b', label="Testing")

            plt.plot(epochs_idx, self.obj.epoch_error_training_plot, 'r', label="Training")
            
            val_set_len = len(self.obj.epoch_error_validation_plot)
            test_set_len = len(self.obj.epoch_error_training_plot)
            if (val_set_len > 0) and (test_set_len==val_set_len):
                plt.plot(epochs_idx, self.obj.epoch_error_validation_plot, 'g', label="Validation")
                if self.obj.early_stopping is not None:
                    plt.plot(self.obj.optimal_error_position, 
                            self.obj.epoch_error_validation_plot[self.obj.optimal_error_position-1], 
                            'ko', 
                            label="Min validation error", 
                            markersize=12,linewidth=2)
            
            plt.title(ds_name+' '+str.capitalize(self.obj.optimizer.optimizer_name)+' '+architecture)
            plt.ylabel("Error")
            plt.ylim(0,1)
            plt.xlabel("Epochs")
            plt.legend()
            plt.grid()
            if (save_dir is not None):
                plt.style.use('fivethirtyeight')
                plt.savefig(save_dir)
            plt.clf()
            # plt.close()
    
    def plot_epoch_accuracy(self, ds_name, save_dir): 
        
        figure(figsize=(16, 9), dpi=80)

        if (self.obj.optimizer.optimizer_type == 'gradient'):
            # Naming
            architecture = ''
            for i in range(0, len(self.obj.weights)):                    
                                                                    # Activation function 
                architecture += str(self.obj.weights[i].shape[1]) + "-" + str(self.obj.activation_functions[i+1].__name__)[0] + "+" 
            architecture = architecture[:-1] # Remove to the last plus    
            architecture += " "+self.obj.error_function.__name__   
            
            # if (self.obj.learning_rate>0):
            #     architecture += " " + "lr: " + str(self.obj.learning_rate)
            epochs_idx = [i+1 for i in range(len(self.obj.epoch_error_testing_plot))]

            plt.plot(epochs_idx, self.obj.epoch_testing_accuracy_plot,'b', label="Testing")
            
            plt.plot(epochs_idx, self.obj.epoch_training_accuracy_plot, 'r', label="Training")
            
            val_set_len = len(self.obj.epoch_error_validation_plot)
            test_set_len = len(self.obj.epoch_error_training_plot)

            if (val_set_len > 0) and (test_set_len == val_set_len):
                plt.plot(
                    epochs_idx, 
                    self.obj.epoch_validation_accuracy_plot, 
                    'g', 
                    label="Validation")
                if self.obj.early_stopping is not None:
                    plt.plot(
                        self.obj.optimal_error_position, 
                        self.obj.epoch_testing_accuracy_plot[self.obj.optimal_error_position-1], 
                        'ko', 
                        label="Min validation error", 
                        markersize=12,linewidth=2)
        else:
            # architecture = ''
            # for i in range(0, len(self.obj.weights)):                    
            #     # Getting the structure of weights
            #     architecture += str(self.obj.weights[0][i].shape[1]) + "-" + str(self.obj.activation_functions[i+1].__name__)[0] + "+" 
            # architecture = architecture[:-1] # Remove to the last plus    
            # architecture += " "+self.obj.error_function.__name__   

            epochs_idx = [i for i in range(len(self.obj.epoch_testing_accuracy_plot))]

            # log.info("Trial indexing")
            # log.info(self.obj.epoch_testing_accuracy_plot)

            for i in range(1):#len(self.obj.epoch_error_testing_plot[0])):
                temp_arr1 = []

                for j in range(len(self.obj.epoch_testing_accuracy_plot)):
                    temp_arr1.append(self.obj.epoch_testing_accuracy_plot[j][i])

                log.info(f"{epochs_idx}, {temp_arr1}")
                plt.plot(epochs_idx, temp_arr1, random_color(), label=f"Testing {i}")
                log.warn(f"Executing for the {i} time")

            # for i in range(1):#len(self.obj.epoch_training_accuracy_plot[0])):
            #     temp_arr2 = []
            #     for j in range(len(self.obj.epoch_training_accuracy_plot)):
            #         temp_arr2.append(self.obj.epoch_training_accuracy_plot[j][i])
            #     log.warn(temp_arr2)
                
            #     plt.plot(epochs_idx, temp_arr2, 'r',  label=f"Training {i}")

            # val_set_len = len(self.obj.epoch_testing_accuracy_plot)
            # test_set_len = len(self.obj.epoch_error_training_plot)
            # if (val_set_len > 0) and (test_set_len==val_set_len):
            #     for i in range(1):#len(self.obj.epoch_testing_accuracy_plot[0])):
            #         temp_arr2 = []
            #         for j in range(len(self.obj.epoch_testing_accuracy_plot)):
            #             temp_arr2.append(self.obj.epoch_testing_accuracy_plot[j][i])
            #         log.warn(temp_arr2)
                    
            #         plt.plot(epochs_idx, temp_arr2, random_color(),  label=f"Training {i}")


        # plt.title(ds_name+' '+str.capitalize(self.obj.optimizer.optimizer_name)+' '+architecture)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.ylim(0,1)
        plt.legend()
        plt.grid()
        # plt.show()
        if (save_dir is not None):
            plt.style.use('fivethirtyeight')
            plt.savefig(save_dir)
        plt.clf()
        plt.close()
            
    def plot_confusion_matrix(self, save_dir):
        figure(figsize=(16, 9), dpi=80)
        df = pd.DataFrame(self.obj.confusion_matrix)
        # sn.set(font_scale=1.4)
        sn.heatmap(df, annot=True, annot_kws={"size": 16}, fmt='g') 
        plt.xlabel("Predictions")
        plt.ylabel("Actual Values")
        # plt.savefig('B7_confusion_matrix.png')
        if (save_dir is not None):
            # plt.style.use('seaborn')
            plt.savefig(save_dir)
        plt.cla()
        plt.clf()
        plt.close()