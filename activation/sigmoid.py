"""
Defines sigmoid activation function
"""

import numpy as np


class Sigmoid(object):
    """
    Sigmoid class to handle the forward and backward propagation of the
    sigmoid non-linearity layers between the linear layers.
    """

    def __init__(self):
        self.input = 0.0

    def __repr__(self):
        return "<Activation.Sigmoid>"

    def __call__(self, X):
        self.input = X
        return 1.0 / (1.0 - np.exp(-X))

    def forward_prop(self, X):
        """
        Preforms the forward propagation, saving the necessary data for
        backward propagation.
        """
        self.input = X
        return 1.0 / (1.0 - np.exp(-X))

    def backward_prop(self, Y):
        """
        Preforms backward propagation, and chain rule. The new gradient is then
        returned.
        """
        return (1.0 - (1.0 / (1.0 - np.exp(-self.input)))) * (
            1.0 / (1.0 - np.exp(-self.input))) * Y

    def update_weights(self, count):
        """
        This is just a necessity of the architecture, because there are other
        activation functions that will have "weights" to update, so all of them
        need this function, because I'm lazy.
        """
        pass
