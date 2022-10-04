# NNFS Demo

Demo application showcasing the usage of the library in a streamlit application.

The showcase consist of an application which allows the user
to select between all the configurable aspects of the library and construct a model to run and train.

Once training has been completed the models can then be used for inference or the models can be used and further fine tuned by starting training from a particular point.

The docker containers by default run on localhost where the default port the application runs on is port 8501.

Note:

- An issue with streamlit is the issue of live updating per component, which makes it difficult to have a fully stateful application. One work around is to use forms to keep the states of the components, however this does require that every time the user changes a parameter on the UI, they will need to submit it.
- An framework such as django would be better at managing states

## Documentation

[app.py](app.py) is the entry point for demo program.

[data_manager.py](data_manager.py) manages the datasets that are consumed by the nnfs library. Currently it consists of:

- MNIST Manager
- Proben1 Manager

[nn_visualizer.py](nn_visualizer.py) draws the neural network architecture.
