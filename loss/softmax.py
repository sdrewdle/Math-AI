"""
Defines softmax loss function
"""

import numpy as np


class Softmax(object):

    def __init__(self):
        self.input = 0.0

    def __repr__(self):
        return "<Loss.Softmax>"

    def forward_prop(self, X):
        # I didn't fully know how to setup the loss function
        self.input = X

    def backward_prop(self):
        # I definently didn't know how to back propogate the loss function
        return np.zeros(self.input.shape)
