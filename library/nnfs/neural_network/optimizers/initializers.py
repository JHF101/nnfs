import numpy as np

class Initializers:
    """
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    """

    def __init__(self) -> None:
        pass

    def heuristic(self, **kwargs):
        # Lower
        # Upper
        # dim0
        # dim1
        return np.random.uniform(low=kwargs['lower'],
                        high=kwargs['upper'],
                        size=(kwargs['dim0'],kwargs['dim1']))

    def xavier(self, **kwargs):
        prev_layer = kwargs['dim0']
        next_layer = kwargs['dim1']
        if 'normalized' in kwargs.keys():
            normalized=kwargs['normalized']
        else:
            normalized=False

        if normalized:
            main_term = (np.sqrt(6.0) / np.sqrt(prev_layer+next_layer))
            lower = - main_term
            upper =  main_term

        else:
            main_term = (1.0 / np.sqrt(prev_layer))
            lower = - main_term
            upper = main_term

        initalized = np.random.rand(prev_layer * next_layer)

        scaled_initialized = lower + initalized * (upper - lower)

        return scaled_initialized.reshape((prev_layer,next_layer))

    def he(self, **kwargs):
        prev_layer = kwargs['dim0']
        next_layer = kwargs['dim1']

        # Range of weights
        ranges = np.sqrt(2.0 / prev_layer)

        # Generate random numbers
        intialized = np.random.randn(prev_layer*next_layer)

        # Scaled to desired interval
        scaled_initialized = intialized * ranges

        return scaled_initialized.reshape((prev_layer,next_layer))



