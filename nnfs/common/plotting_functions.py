from distutils.command.config import config
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sn
import numpy as np
import streamlit as st
import re
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from nnfs.utils.logs import create_logger
log = create_logger(__name__)

def random_color():
    rgbl=[255,0,0]
    np.random.shuffle(rgbl)
    return tuple(rgbl)

class Plots:
    def __init__(self):
        # TODO: Sort out layout
        self.err_plot_space, self.acc_plot_space = st.empty(), st.empty()

        with self.err_plot_space:
            self.st_err = None

        with self.acc_plot_space:
            self.st_acc = None

        self.st_confusion = None

        self.acc_layout = go.Layout(
            plot_bgcolor="#FFF",
            xaxis=dict(
                title="epochs",
                linecolor="#BCCCDC",
                showgrid=False
            ),
            yaxis=dict(
                title="accuracy",
                linecolor="#BCCCDC",
                showgrid=False,
            ),
            legend=dict(
                itemclick="toggleothers",
                itemdoubleclick="toggle",
            )
        )
        self.err_layout = go.Layout(
            plot_bgcolor="#FFF",
            xaxis=dict(
                title="epochs",
                linecolor="#BCCCDC",
                showgrid=False
            ),
            yaxis=dict(
                title="error",
                linecolor="#BCCCDC",
                showgrid=False,
            ),
            legend=dict(
                itemclick="toggleothers",
                itemdoubleclick="toggle",
            )
        )
        self.config = {"displayModeBar": False, "showTips": False}

    def update_data(self, self_data):
        self.optimizer = self_data.optimizer
        self.weights = self_data.weights
        self.activation_functions = self_data.activation_functions
        self.error_function = self_data.error_function

        self.epoch_error_testing_plot = self_data.epoch_error_testing_plot
        self.epoch_error_training_plot = self_data.epoch_error_training_plot
        self.epoch_error_validation_plot = self_data.epoch_error_validation_plot

        self.epoch_testing_accuracy_plot = self_data.epoch_testing_accuracy_plot
        self.epoch_training_accuracy_plot = self_data.epoch_training_accuracy_plot
        self.epoch_validation_accuracy_plot = self_data.epoch_validation_accuracy_plot

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

            # Creating indeces
            epochs_idx = [i+1 for i in range(len(self.epoch_error_testing_plot))]

            val_set_len = len(self.epoch_error_validation_plot)
            test_set_len = len(self.epoch_error_training_plot)

            fig = go.Figure(layout=self.err_layout)

            fig.add_trace(
                go.Scatter(
                    x=epochs_idx,
                    y=self.epoch_error_training_plot,
                    mode='lines',
                    name='Training'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=epochs_idx,
                    y=self.epoch_error_testing_plot,
                    mode='lines',
                    name='Testing'
                )
            )

            if (val_set_len > 0) and (test_set_len==val_set_len):
                fig.add_trace(
                    go.Scatter(
                        x=epochs_idx,
                        y=self.epoch_error_validation_plot,
                        mode='lines',
                        name='Validation'
                    )
                )

            fig.update_layout(title=ds_name+' '+str.capitalize(self.optimizer.optimizer_name)+' '+architecture)

            with self.err_plot_space:
                self.st_err = st.plotly_chart(fig, use_container_width=True, config=self.config)


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
            epochs_idx = [i+1 for i in range(len(self.epoch_testing_accuracy_plot))]

            val_set_len = len(self.epoch_validation_accuracy_plot)
            test_set_len = len(self.epoch_testing_accuracy_plot)


            fig = go.Figure(layout=self.acc_layout)
            fig.add_trace(
                go.Scatter(
                    x=epochs_idx,
                    y=self.epoch_training_accuracy_plot,
                    mode='lines',
                    name='Training'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=epochs_idx,
                    y=self.epoch_testing_accuracy_plot,
                    mode='lines',
                    name='Testing'
                )
            )

            if (val_set_len > 0) and (test_set_len==val_set_len):
                fig.add_trace(
                    go.Scatter(
                        x=epochs_idx,
                        y=self.epoch_validation_accuracy_plot,
                        mode='lines',
                        name='Validation'
                    )
                )

            fig.update_layout(title=ds_name+' '+str.capitalize(self.optimizer.optimizer_name)+' '+architecture)

            with self.acc_plot_space:
                self.st_acc = st.plotly_chart(fig, use_container_width=True, config=self.config)

    def plot_confusion_matrix(self, confusion_matrix):
        df = pd.DataFrame(confusion_matrix[-1])
        dfc = df#.corr()
        z = dfc.values.tolist()
        z_text = [[str(y) for y in x] for x in z]
        fig = ff.create_annotated_heatmap(z,
                                            x=list(df.columns),
                                            y=list(df.columns),
                                            annotation_text=z_text, colorscale='agsunset')
        fig['data'][0]['showscale'] = True

        self.st_confusion = st.plotly_chart(fig, use_container_width=True)

    # def clear_plots(self):
    #     with self.err_plot_space:
    #         self.st_err = None

    #     with self.acc_plot_space:
    #         self.st_acc = None

    #     self.st_confusion = None

