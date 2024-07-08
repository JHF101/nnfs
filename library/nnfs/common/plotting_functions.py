import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

from nnfs.utils.logs import create_logger

log = create_logger(__name__)

# TODO: Add standard plotly support without need for streamlit


def random_color():
    rgbl = [255, 0, 0]
    np.random.shuffle(rgbl)
    return tuple(rgbl)


class Plots:
    def __init__(self, plots_config={"type": "plotly"}):
        self.plot_config = plots_config
        if self.plot_config.get("type") == "streamlit":
            # TODO: Sort out layout
            self.err_plot_space, self.acc_plot_space = st.empty(), st.empty()

            with self.err_plot_space:
                self.st_err = None

            with self.acc_plot_space:
                self.st_acc = None

            self.st_confusion = None

        if self.plot_config.get("accuracy"):
            self.acc_layout = go.Layout(
                plot_bgcolor="#FFF",
                xaxis=dict(title="epochs", linecolor="#BCCCDC", showgrid=False),
                yaxis=dict(
                    title="accuracy",
                    linecolor="#BCCCDC",
                    showgrid=False,
                ),
                legend=dict(
                    itemclick="toggleothers",
                    itemdoubleclick="toggle",
                ),
            )
        if self.plot_config.get("error"):
            self.err_layout = go.Layout(
                plot_bgcolor="#FFF",
                xaxis=dict(title="epochs", linecolor="#BCCCDC", showgrid=False),
                yaxis=dict(
                    title="error",
                    linecolor="#BCCCDC",
                    showgrid=False,
                ),
                legend=dict(
                    itemclick="toggleothers",
                    itemdoubleclick="toggle",
                ),
            )
        self.config = {"displayModeBar": False, "showTips": False}
        self.architecture = ""

    def update_data(self, self_data):
        """Receives the data from neural network which can then be used by the
        rest of the class.

        Parameters
        ----------
        self_data : any
            neural_network class variables
        """
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

        if self.optimizer.optimizer_type == "non-gradient":
            self.number_of_parents = self.optimizer.number_of_parents

        if self.optimizer.optimizer_type == "gradient":
            if hasattr(self, "learning_rate"):
                self.optimizer.learning_rate = self.optimizer.learning_rate
            else:
                self.optimizer.learning_rate = 0

        self.generate_fig_title()

    def plot_epoch_error(self):
        """Plots the loss during training of the network."""

        # Creating indeces
        epochs_idx = [i + 1 for i in range(len(self.epoch_error_testing_plot))]

        val_set_len = len(self.epoch_error_validation_plot)
        test_set_len = len(self.epoch_error_training_plot)

        fig = go.Figure(layout=self.err_layout)

        fig = self.add_traces_to_figure(
            fig=fig,
            x_data=epochs_idx,
            y_data=self.epoch_error_training_plot,
            legend="Training",
        )

        fig = self.add_traces_to_figure(
            fig=fig,
            x_data=epochs_idx,
            y_data=self.epoch_error_testing_plot,
            legend="Testing",
        )

        # Check if there actually is a validation set
        if (val_set_len > 0) and (test_set_len == val_set_len):
            fig = self.add_traces_to_figure(
                fig=fig,
                x_data=epochs_idx,
                y_data=self.epoch_error_validation_plot,
                legend="Validation",
            )

        fig.update_layout(title=self.architecture)

        if self.plot_config.get("type") == "streamlit":
            with self.err_plot_space:
                # Plot the streamlit graph
                self.st_err = st.plotly_chart(
                    fig, use_container_width=True, config=self.config
                )
        elif self.plot_config.get("type") == "plotly":
            pio.write_image(fig, file=self.plot_config.get("error"), format="png")
            print(f"saved image in {self.plot_config.get('error')}")

    def plot_epoch_accuracy(self):
        """Plots the accuracy of the predictions of the model during training."""

        epochs_idx = [i + 1 for i in range(len(self.epoch_testing_accuracy_plot))]

        val_set_len = len(self.epoch_validation_accuracy_plot)
        test_set_len = len(self.epoch_testing_accuracy_plot)

        # Creating the figures
        fig = go.Figure(layout=self.acc_layout)

        fig = self.add_traces_to_figure(
            fig=fig,
            x_data=epochs_idx,
            y_data=self.epoch_training_accuracy_plot,
            legend="Training",
        )

        fig = self.add_traces_to_figure(
            fig=fig,
            x_data=epochs_idx,
            y_data=self.epoch_testing_accuracy_plot,
            legend="Testing",
        )

        # Check if there actually is a validation set
        if (val_set_len > 0) and (test_set_len == val_set_len):
            fig = self.add_traces_to_figure(
                fig=fig,
                x_data=epochs_idx,
                y_data=self.epoch_validation_accuracy_plot,
                legend="Validation",
            )

        fig.update_layout(title=self.architecture)

        if self.plot_config["type"] == "streamlit":
            with self.acc_plot_space:
                # Plot the streamlit graph
                self.st_acc = st.plotly_chart(
                    fig, use_container_width=True, config=self.config
                )
        elif self.plot_config["type"] == "plotly":
            pio.write_image(fig, file=self.plot_config.get("accuracy"), format="png")

    def add_traces_to_figure(self, fig, x_data, y_data, legend):
        """Add plotly traces to the figure

        - Can take in single dimensional data for x and y (gradient)
        - Can take in multidimensional data for x and y (genetic)
        Parameters
        ----------
        fig : go.Figure
            The figure that the traces will be plotted n
        x_data : np.ndarray
            x-axis data
        y_data : np.ndarray
            y-axis data
        legend : str
            The legend of the trace

        Returns
        -------
        fig : go.Figure
            The graph with additional traces plotted on it.
        """
        # Check whether it's a list
        non_gradient_bool = isinstance(y_data[0], np.ndarray)
        if non_gradient_bool:
            y_data = pd.DataFrame(y_data)
            for dim in range(0, self.number_of_parents):
                # Plotting the cols
                col_data = y_data[dim]
                fig.add_trace(
                    go.Scatter(
                        x=x_data, y=col_data, mode="lines", name=f"{legend} {dim}"
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(x=x_data, y=y_data, mode="lines", name=f"{legend}")
            )

        return fig

    def generate_fig_title(self):
        """Generates the figure title describing the architecture
        of the network
        """
        # Naming
        architecture = ""
        if self.optimizer.optimizer_type == "non-gradient":
            loop_value = self.number_of_parents
        else:
            loop_value = len(self.weights)

        for i in range(0, loop_value):
            if self.optimizer.optimizer_type == "non-gradient":
                architecture += str(self.number_of_parents)
            else:
                weight_dim = self.weights[i].shape[1]
                # Activation function
                architecture += (
                    str(weight_dim)
                    + "-"
                    + str(self.activation_functions[i + 1].__name__)[0]
                    + "+"
                )

        # Remove to the last plus
        architecture = architecture[:-1]
        architecture += " " + self.error_function.__name__
        if self.optimizer.optimizer_type == "gradient":
            if self.optimizer.learning_rate > 0:
                architecture += " " + "lr: " + str(self.learning_rate)

        self.architecture = (
            str.capitalize(self.optimizer.optimizer_name) + " " + architecture
        )

    def plot_confusion_matrix(self, confusion_matrix):
        """Plots the confusion matrix of predicted outputs versus
        ground truth valued outputs of the network.

        Parameters
        ----------
        confusion_matrix : object
            Confusion matrix returned from seaborn.
        """
        df = pd.DataFrame(confusion_matrix)
        z = df.values.tolist()
        z_text = [[str(y) for y in x] for x in z]
        fig = ff.create_annotated_heatmap(
            z,
            x=list(df.columns),
            y=list(df.columns),
            annotation_text=z_text,
            colorscale="agsunset",
        )
        fig["data"][0]["showscale"] = True

        if self.plot_config["type"] == "streamlit":
            # Plot the streamlit graph
            self.st_confusion = st.plotly_chart(fig, use_container_width=True)
        elif self.plot_config["type"] == "plotly":
            fig.write_image(self.plot_config["confusion_matrix"])
