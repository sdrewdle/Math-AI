"""Defines an input layer class"""

class Input(object):
    """Input layer class"""

    def __init__(self, shape=None, name=None):
        self.shape = shape
        self.name = name
        self.output_size = shape

    def __repr__(self):
        return "<Layer.input({}) {}>".format(self.name, self.shape)

    def repr(self):
        return "<Layer.input {} {}>".format(self.name, self.shape)

    def load_data(self, data):
        self.data = data

    def set_data(self, data):
        self.data = data

    def forward_prop(self):
        return self.data

    def backward_prop(self, dD):
        pass

    def update_weights(self, count):
        pass
