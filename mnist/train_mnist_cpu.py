#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST
This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.
"""
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F

batchsize = 100
n_epoch   = 20
n_units   = 1000

# Prepare dataset
print 'fetch MNIST dataset'
mnist = fetch_mldata('MNIST original')
mnist.data   = mnist.data.astype(np.float32)
mnist.data  /= 255
mnist.target = mnist.target.astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist.data,   [N])
y_train, y_test = np.split(mnist.target, [N])
N_test = y_test.size

# Prepare multi-layer perceptron model
model = FunctionSet(l1=F.Linear(784, n_units),
	            l2=F.Linear(n_units, n_units),
	            l3=F.Linear(n_units, 10))
# Neural net architecture
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y  = model.l3(h2)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())
# Learning loop
for epoch in xrange(1, n_epoch+1):
    print 'epoch', epoch
    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]
        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
	loss.backward()
        optimizer.update()
        sum_loss     += float(loss.data) * batchsize
        sum_accuracy += float(acc.data) * batchsize
	print i
    
    print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)
    
    # evaluation
    sum_accuracy = 0
    sum_loss     = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]
        
        loss, acc = forward(x_batch, y_batch, train=False)
	sum_loss     += float(loss.data) * batchsize
        sum_accuracy += float(acc.data) * batchsize
    
    print 'test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test)

