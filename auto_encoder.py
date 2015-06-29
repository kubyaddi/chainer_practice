#/usr/bin/env python

from chainer import Variable, Function, FunctionSet, optimizers
import chainer.functions as F

class AutoEncoder:
    """
    Constract AutoEncoder by #input and #hidden
    """
    def __init__(self, xn, hn):
        self.model = FunctionSet(encode=F.Linear(xn, hn), decode=F.Linear(hn, xn))

    def encode(self, x, train=True):
        h = F.dropout(F.relu(self.model.encode(x)), train=train)
        return h

    def decode(self, h, train=True):
        y = F.dropout(F.relu(self.model.decode(h)), train=train)
        return y

    def train_once(self, x_data):
        x = Variable(x_data)
        h = self.encode(x)
        y = self.decode(h)
        return F.mean_squared_error(x, y)#, F.accuracy(x, y)

    def reconstract(self, x_data):
        x = Variable(x_data)
        h = self.encode(x, train=False)
        y = self.decode(h, train=False)
        return y.data


