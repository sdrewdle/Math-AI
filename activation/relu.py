"""
Defines sigmoid activation function
"""

import numpy as np


class ReLU(object):

    def __init__(self):
        self.input = None

    def __repr__(self):
        return "<Activation.ReLU>"

    def __call__(self, X):
        self.input = X
        return np.maximum(X, 0.0)

    def forward_prop(self, X):
        self.input = X
        return np.maximum(X, 0.0)

    def backward_prop(self, Y):
        Y[self.input < 0] = 0
        return Y
