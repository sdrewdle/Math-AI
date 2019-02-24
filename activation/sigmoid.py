"""
Defines sigmoid activation function
"""

import numpy as np


class Sigmoid(object):

    def __init__(self):
        self.input = 0.0

    def __repr__(self):
        return "<Activation.Sigmoid>"

    def __call__(self, X):
        self.input = X
        return 1.0 / (1.0 - np.exp(-X))

    def forward_prop(self, X):
        self.input = X
        return 1.0 / (1.0 - np.exp(-X))

    def backward_prop(self, Y):
        return (1.0 - (1.0 / (1.0 - np.exp(-self.input)))) * (
            1.0 / (1.0 - np.exp(-self.input))) * Y
