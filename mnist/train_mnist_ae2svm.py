#/usr/bin/env python

import numpy as np
from chainer import Variable, Function, FunctionSet, optimizers
import chainer.functions as F
from sklearn.datasets import fetch_mldata
from sklearn import svm

from auto_encoder import AutoEncoder

batchsize = 100
n_epoch = 5
n_units = 100

def fetch_mnist():
    import os.path
    if(os.path.exists("MNISTorg.pklz")):
        import gzip
        try:
            import cPickle as pickle
        except:
            import pickle
        mf = gzip.open("MNISTorg.pklz")
        dat = pickle.load(mf)
        return dat
    else:
        return fetch_mldata("MNIST original")

mnist = fetch_mnist()
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255
mnist.target = mnist.target.astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist.data, [N])
y_train, y_test = np.split(mnist.target, [N])
N_test = y_test.size

#First layer
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

#1layer SVM
f_train1 = ae1.encode(Variable(x_train), train=False).data
#clf1 = svm.SVC()
clf1 = svm.LinearSVC()
clf1.fit(f_train1, y_train)
f_test1 = ae1.encode(Variable(x_test), train=False).data
print "1layer test mean score={}".format(clf1.score(f_test1, y_test))

#Second layer
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

#2layer SVM
f_train2 = ae2.encode(Variable(f_train1), train=False).data
#clf2 = svm.SVC()
clf2 = svm.LinearSVC()
clf2.fit(f_train2, y_train)
f_test2 = ae2.encode(Variable(f_test1), train=False).data
print "2layer test mean score={}".format(clf2.score(f_test2, y_test))

