import numpy as np
import matplotlib.pyplot as plt

GRAY_SCALE_RANGE = 255

import pickle

data_filename = 'data.pkl'
print('Loading data from file \'' + data_filename + '\' ...')
with open(data_filename, 'rb') as f:
    train_labels = pickle.load(f)
    train_images = pickle.load(f)
    test_labels = pickle.load(f)
    test_images = pickle.load(f)
    num_pixel = pickle.load(f)
print('Data loading complete.')

import tensorflow as tf

train_images = np.array(train_images)
train_images.resize(train_images.size // num_pixel, num_pixel)
test_images = np.array(test_images)
test_images.resize(test_images.size // num_pixel, num_pixel)
test_labels = np.array(test_labels)
train_labels = np.array(train_labels)

train_labels_ten = np.zeros((train_labels.size, 10))
test_labels_ten = np.zeros((test_labels.size, 10))
for i in range(10):
    train_labels_ten[:, i] = train_labels == i
    test_labels_ten[:, i] = test_labels == i

## normalization
train_images = train_images / GRAY_SCALE_RANGE
test_images = test_images / GRAY_SCALE_RANGE

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

L = [num_pixel, 400, 300, 200, 100, 30, 10]
LAYERS = len(L) - 1

learning_rate = 1e-2
iterations = 50000
batch_size = 100
regular_lambda = 0 #1e-4

W = list(range(LAYERS + 1))
b = list(range(LAYERS + 1))
for i in range(LAYERS):
    W[i + 1] = weight_variable([L[i], L[i + 1]])
    b[i + 1] = bias_variable([L[i + 1]])

x = tf.placeholder(tf.float32, [None, num_pixel])
yt = list(range(LAYERS + 1))
yt[0] = x
for i in range(LAYERS):
    yt[i + 1] = tf.nn.relu(tf.matmul(yt[i], W[i + 1]) + b[i + 1])
y = tf.nn.softmax(yt[LAYERS])
y_ = tf.placeholder("float", [None, 10])

l2_loss = 0
for i in range(LAYERS):
    l2_loss += tf.nn.l2_loss(W[i + 1]) + tf.nn.l2_loss(b[i + 1])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
cost_function = cross_entropy + regular_lambda * l2_loss

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
init = tf.global_variables_initializer()

print('Start training ...')
print('Neural Network Layers:', L)
print('Learning Rate:', learning_rate)
print('Iterations:', iterations)
print('Batch Size:', batch_size)
print('Regularization lambda:', regular_lambda)

sess = tf.Session()
sess.run(init)

def new_batch(batch_size):
    batch_idx = np.random.choice(range(train_images.shape[0]), size = batch_size, replace = False)
    batch_x = np.zeros((batch_size, num_pixel))
    batch_y_ = np.zeros((batch_size, 10))
    for i in range(batch_size):
        batch_x[i] = train_images[batch_idx[i]]
        batch_y_[i] = train_labels_ten[batch_idx[i]]
    return batch_x, batch_y_

if False:
    sess.run(train_step, feed_dict = {x: train_images, y_: train_labels_ten})
else:
    for i in range(iterations):
        batch_x, batch_y_ = new_batch(batch_size)
        sess.run(train_step, feed_dict = {x: batch_x, y_: batch_y_})
        if i % (iterations // 100) == 0:
            print('Process: {}%'.format((i // (iterations // 100) + 1) * 1))
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print('Accuracy:', sess.run(accuracy, feed_dict = {x: test_images, y_: test_labels_ten}))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print('Accuracy:', sess.run(accuracy, feed_dict = {x: test_images, y_: test_labels_ten}))
