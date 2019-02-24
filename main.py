#!/usr/bin/env python3
"""
Startup script for ML projects
"""

# import matplotlib.pyplot as plt
from network import Network
# import data
# import layer
# import activation

def main():
    """
    Function that is called on startup of the ML project code
    """
    nn = Network()
    nn.add_layer(Network.INPUT, shape=785)
    nn.add_layer(Network.DENSE, neurons=500)
    nn.add_layer(Network.DENSE, neurons=10)
    print(nn)
    print(nn.repr())
    # data.mnist.load_all(flatten=True)
    # X, Y, Xt, Yt = data.mnist.load_all(flatten=True)
    # print(X.shape, Y.shape, Xt.shape, Yt.shape)
    # x = layer.input.Input(784, 'X')
    # l0 = layer.dense.Dense(x, 500, activation.relu.ReLU, name="l0")
    # print(l0.repr())
    # x.load_data(X[0])
    # out = l0.forward_prop()
    # l0.backward_prop(np.random.rand(*out.shape))


if __name__ == "__main__":
    main()
