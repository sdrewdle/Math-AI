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

    def __init__(self, architecture=[], loss_function=None):
        """
        Initializes a neural network structure, with a given architecture and
        loss function. This initialization is where the hyperparameters will
        be set. But for now I'm keeping things simple.
        """
        self.architecture = architecture
        self.layers = []
        self.loss_function = None
        self.layer_count = {self.INPUT: 0, self.DENSE: 0}
        for layer in architecture:
            if isinstance(layer, dict):
                ltype = layer['type']
                layer.pop('type', None)
                self.add_layer(ltype, **layer)
            else:
                self.add_layer(**layer)

    def add_layer(self, layer_type, **kwargs):
        """
        Adds a new layer to the network, linking the input of the new layer
        to the output of the previous layer. This should make things very nice
        for users, so one can initialize and then add layers as desired. Do
        note that python makes these links by reference, so changing something
        in the list is the same as changing the value in the layer chain. This
        means that this works nicely.
        """
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

    def train(self, data, labels):
        """
        This method implement Stochastic gradient descent. Really it would be
        better to abstractly handle this, and let the user chose the training
        method, but this is easy, and its a first try.

        This doesn't actual do the gradient decent, this is only a method to
        handle a single batch of data. It takes the data, evaluates the loss
        and gradients for all of the elements of the data, then it averages
        those gradients and makes the change to the layers. The full decent
        should be calling this method many many times.
        """
        if self.loss_function is None:
            print("Training requires a loss function!")
            return
        for iteration in range(len(data)):
            self.layers[0].load_data(data[iteration])
            loss = self.loss_function.forward_prop(
                self.layers[-1].forward_prop(), labels[iteration])
            self.layers[-1].backward_prop(
                self.loss_function.backward_prop(loss))
        self.layers[-1].update_weights(len(data))

    def predict(self, X):
        """
        This is the end goal. A user would use this function to actually use
        the network, after you train it, you want it to "predict" the label
        for the input. This just handles the forward propagation, and will
        return the scores/probabilities that the network outputs.
        """
        self.layers[0].load_data(X)
        scores = self.layers[-1].forward_prop()
        # I should convert from the scores to the probabilities
        return scores

    def __repr__(self):
        return "[NeuralNetwork {}]".format(len(self.layers))

    def repr(self):
        return "[NeuralNetwork {}]".format(self.layers[-1].repr() if self.
                                           layers else None)
