"""
Defines a network class that handles all of the construction and training of
the network
"""
from enum import Enum

import layer
import activation


class Network(object):
    INPUT = 1
    DENSE = 2

    SIGMOID = 11
    RELU = 12

    def __init__(self, architecture=[]):
        self.architecture = architecture
        self.layers = []
        self.layer_count = {self.INPUT: 0, self.DENSE: 0}
        for layer in architecture:
            if isinstance(layer, dict):
                ltype = layer['type']
                layer.pop('type', None)
                self.add_layer(ltype, **layer)
            else:
                self.add_layer(**layer)

    def add_layer(self, layer_type, **kwargs):
        if not self.layers and layer_type != self.INPUT:
            print("First layer must be an input layer!")
            return
        if 'activation' in kwargs:
            if kwargs['activation'] == self.SIGMOID:
                kwargs['activation'] = activation.sigmoid.Sigmoid
            elif kwargs['activation'] == self.RELU:
                kwargs['activation'] = activation.relu.ReLU
        if layer_type == self.INPUT:
            self.layers.append(
                layer.input.Input(
                    **kwargs, name="i{}".format(self.layer_count[layer_type])))
        elif layer_type == self.DENSE:
            self.layers.append(
                layer.dense.Dense(
                    self.layers[-1],
                    **kwargs,
                    name="d{}".format(self.layer_count[layer_type])))

    def __repr__(self):
        return "[NeuralNetwork {}]".format(len(self.layers))

    def repr(self):
        return "[NeuralNetwork {}]".format(self.layers[-1].repr() if self.
                                           layers else None)
