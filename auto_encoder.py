#/usr/bin/env python

from chainer import Variable, Function, FunctionSet, optimizers
import chainer.functions as F

class AutoEncoder:
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

if __name__ == '__main__':
	from sklearn.datasets import fetch_mldata
	import numpy as np

	batchsize = 100
	n_epoch = 5
	n_units = 100

	mnist = fetch_mldata('MNIST original')
	mnist.data = mnist.data.astype(np.float32)
	mnist.data /= 255
	mnist.target = mnist.target.astype(np.int32)

	N = 60000
	x_train, x_test = np.split(mnist.data, [N])
	y_train, y_test = np.split(mnist.target, [N])
	N_test = y_test.size

	ae1 = AutoEncoder(784, n_units)

	optimizer1 = optimizers.Adam()
	optimizer1.setup(ae1.model.collect_parameters())

	for epoch in xrange(1, n_epoch+1):
		print 'epoch', epoch
		perm = np.random.permutation(N)
		for i in xrange(0, N, batchsize):
			sum_loss = 0
			x_batch = x_train[perm[i:i+batchsize]]
			optimizer1.zero_grads()
			loss = ae1.train_once(x_batch)
			loss.backward()
			optimizer1.update()
			sum_loss += float(loss.data) * batchsize
		print 'train mean loss={}'.format(sum_loss/N)

	ae2 = AutoEncoder(n_units, n_units)

	optimizer2 = optimizers.Adam()
	optimizer2.setup(ae2.model.collect_parameters())

	for epoch in xrange(1, n_epoch+1):
		print 'epoch', epoch
		perm = np.random.permutation(N)
		for i in xrange(0, N, batchsize):
			sum_loss = 0
			x_batch = x_train[perm[i:i+batchsize]]
			optimizer2.zero_grads()
			h = ae1.encode(Variable(x_batch), train=False)
			loss = ae2.train_once(h.data)
			loss.backward()
			optimizer2.update()
			sum_loss += float(loss.data) * batchsize
		print 'train mean loss={}'.format(sum_loss/N)

