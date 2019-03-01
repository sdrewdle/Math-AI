"""
Defines relu activation function
"""

import numpy as np


class ReLU(object):
    """
    ReLU class to handle the forward and backward propagation of the ReLU
    non-linearity layers between the linear layers.
    """

    def __init__(self):
        self.input = None

    def __repr__(self):
        return "<Activation.ReLU>"

    def __call__(self, X):
        self.input = X
        return np.maximum(X, 0.0)

    def forward_prop(self, X):
        """
        Preforms the forward propagation, saving the necessary data for
        backward propagation.
        """
        self.input = X
        return np.maximum(X, 0.0)

    def backward_prop(self, Y):
        """
        Preforms backward propagation, and chain rule. The new gradient is then
        returned.
        """
        Y[self.input < 0] = 0
        return Y

    def update_weights(self, count):
        """
        This is just a necessity of the architecture, because there are other
        activation functions that will have "weights" to update, so all of them
        need this function, because I'm lazy.
        """
        pass
