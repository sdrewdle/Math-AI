"""Defines a dense layer class"""
import numpy as np


class Dense(object):
    """Dense layer class"""

    def __init__(self,
                 source_layer,
                 neurons,
                 activation=None,
                 weight_init=None,
                 bias_init=None,
                 name=None):
        self.source_layer = source_layer
        self.output_size = neurons
        self.activation = activation() if activation is not None else None
        self.name = name
        self.grad_weights = None
        self.input = None
        if weight_init is None:
            self.weights = np.random.randn(
                self.output_size, self.source_layer.output_size) * np.sqrt(
                    2.0 / self.output_size)
        elif isinstance(weight_init, np.ndarray):
            self.weights = weight_init
        else:
            self.weights = np.fromfunction(weight_init, (self.output_size))
        if bias_init is None:
            self.bias = np.random.randn()
        elif isinstance(bias_init, float):
            self.bias = bias_init
        else:
            self.bias = bias_init()

    def __repr__(self):
        return "<Layer.Dense({}) {},{},{}>".format(
            self.name, self.source_layer.name, self.output_size,
            self.activation)

    def repr(self):
        return "<Layer.Dense {},{} {}>".format(self.output_size,
                                               self.activation,
                                               self.source_layer.repr())

    def forward_prop(self):
        self.input = self.source_layer.forward_prop()
        if self.activation:
            return (self.activation(self.weights @ self.input) + self.bias)
        return (self.weights @ self.input) + self.bias

    def backward_prop(self, dD):
        # TODO Verify that this is the proper backprop for a dense layer. I
        #      don't think it is.
        if self.activation:
            dD = self.activation.backward_prop(dD)
        self.grad_weights = dD[None].T @ self.input[None]
        self.source_layer.backward_prop(self.weights.T @ dD)
