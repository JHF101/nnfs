# Neural Networks from Scratch

Library that implements a neural network for scratch, with a number of different gradient and non-gradient optimizers.

## Installation

Using python virtual environments (Windows / Linux):

    pip install virtualenv / pip3 install virtualenv

Then create the virtual environment:

    virtualenv venv

Followed by activating the virtual environment (Windows / Linux):

    venv/Scripts/activate / source venv/bin/activate

## Usage

### Architecture

The following is an example of how to set up the neural network model.

```python
    nn_train = Network(
        # Architecure
        layers=[
            (input_layer_size, sigmoid), # Input Layer         , activation function object
            (16, sigmoid),               # First Hidden Layer  , activation function object
            (16, sigmoid),               # Second Hidden Layer , activation function object
            (output_layer_size, softmax) # Output Layer        , activation function object
        ],

        error_function=squared_error,    # Error function object

        use_bias=True,

        # Optimizer used in
        optimizer= GradientDescent(
            learning_rate=0.5,
            weights_initialization=weights_initialization
            ),

        # Overfitting
        training_params = EarlyStopping(
            alpha=10,
            pkt_threshold=0.1,
            k_epochs=5,
            )
    )
```

Where the weights initialization is the method of weight initialization for the neural network.

```python
    weights_initialization=dict(name='heuristic', lower=-0.3, upper=0.3)
    weights_initialization=dict(name='xavier')
    weights_initialization=dict(name='he')
    weights_initialization=None
```

The other possible optimizer are

1. Genetic Oprimizer

   ```python
           optimizer = GeneticOptimizer(
               number_of_parents=4,
               fitness_eval='accuracy',
               weights_initialization=weights_initialization
               )
   ```

2. Gradient descent

   ```python
           optimizer= GradientDescent(
               learning_rate=0.5,
               weights_initialization=weights_initialization
               )
   ```

3. Gradient Descent with Momentum

   ```python
           optimizer= GradientDescentWithMomentum(
               learning_rate=0.05,
               beta=0.9,
               weights_initialization=weights_initialization
               )
   ```

4. Delta-Bar-Delta

   ```python
           optimizer= DeltaBarDelta(
               theta=0.1,
               mini_k=0.01,
               phi=0.1,
               weights_initialization=weights_initialization
               )
   ```

5. Resillient Prop
   ```python
           optimizer= Rprop(
               delta_max=50,
               delta_min=0,
               eta_plus=1.1,
               eta_minus=0.5,
               weights_initialization=weights_initialization
               ),
   ```

### Training

Training the network can be done by passing an array in an array of training, testing and validation (optional):
```python
        nn_train.fit(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=100,
            batch_size=8,
            shuffle_training_data=True,
            # x_validate=x_validate,
            # y_validate=y_validate,
            # --- or --- #
            # validations_set_percent=0.1,
        )
```

Where `x_train`, `x_test` and `x_validate` is a 1 by x numpy vector of the features, `y_train`, `y_test` and `y_validate` is a 1 by x numpy vector of the labeled data. `epochs` are the number of times the entire dataset will be passed over, where `batch_size` is the number of training samples to work through before the internal parameters of the model are updated. The validation sets can be added by adding `validations_set_percent`(0<`validations_set_percent`<1), which takes a percentage of the entire training set and creates a validation set. `shuffle_training_data` reorders the input features and labels during the training loops.

### Prediction

Prediction can be done by using the Network object and passing in the an unlabeled feature:
```python
    prediction=nn_train.predict(
        x_test[1]
        )
```

### Saving a model

A trained model can then be saved by using the save_model function:
```python
    nn_train.save_model('test.pickle')
```

The model is saved in the pickle format.

### Loading a model

Loading a model can be done by instantiating a `Network` object and setting `load_model=True`. One can then use the load_model function to load the model.
```python
    new_nn = Network(
        load_model=True
    )

    new_nn.load_model(
        'test.pickle',
    )
```

This will use the model as it was saved. There are however some other parameters that be adjusted to fine tune models for better performance.

### Model Fine Tuning

When loading a model from a pre-trained model(loaded from a .pickle file), the optimizer can be replaced by another optimizer.
    ```python
    new_nn = Network(
        load_model=True
        optimizer = GradientDescent(learning_rate=0.1),
    )
    ```

Further fine-tuning can be performed by adding a removing bias for further training and also modifying the training functions for each layer in the network when loading the network model:

    ```python

        new_nn.load_model(
        'test.pickle',
        use_bias=True,
        activation_functions=[
            sigmoid,
            sigmoid,
            tanh,
            softmax
        ]
    )
    ```

Note: Only gradient based optimizers can fine-tune other gradient based models.

After the model has been loaded and/or fine tuned, all of the other functions are available to it.

## Features

### Optimizers

- [Gradient Descent](nnfs/neural_network/optimizers/gradient/gradient_descent.py)
- [Gradient Descent with Momentum](nnfs/neural_network/optimizers/gradient/gradient_descent_momentum.py)
- [Delta-Bar-Delta](nnfs/neural_network/optimizers/gradient/delta_bar_delta.py)
- [Resilient Propagation (RProp)](nnfs/neural_network/optimizers/gradient/rprop.py)
- [RMSProp](nnfs/neural_network/optimizers/gradient/rms_prop.py)
- [Adaptive Moment Estimation](nnfs/neural_network/optimizers/gradient/adam.py)
- [Genetic Optimizer](nnfs/neural_network/optimizers/non_gradient/genetic.py)

### Initializers

The [initializers](nnfs/neural_network/optimizers/initializers.py) are:

- Heuristic (User specified upper and lower ranges)
- Xavier
- He
- None (Randomized between -0.5 and 0.5)

### Activation functions

The [activation functions](nnfs/activations/activation_functions.py) are:

- sigmoid
- tanh
- relu
- softmax
- linear

### Error/Loss functions

The [error functions](nnfs/errors/error_functions.py) are:

- mse
- squared error
- cross_entropy (Bug)

# References

- https://ml-cheatsheet.readthedocs.io/
- https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
- https://machinelearningmastery.com/cross-entropy-for-machine-learning/
- https://www.v7labs.com/blog/neural-networks-activation-functions
- https://mlfromscratch.com/activation-functions-explained/#/
- https://deepnotes.io/softmax-crossentropy

## Sphinx

- https://towardsdatascience.com/documenting-python-code-with-sphinx-554e1d6c4f6d

## Datasets

- https://github.com/jeffheaton/proben1
- http://yann.lecun.com/exdb/mnist/

```

```
