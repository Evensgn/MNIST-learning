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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

CONV_L = [1, 32, 64]
CONV_LAYERS = len(CONV_L) - 1

DENSE_L = [7 * 7 * 64, 1024, 10]
DENSE_LAYERS = len(DENSE_L) - 1

learning_rate = 2e-3
iterations = 50000
batch_size = 100
regular_lambda = 1e-4
drop_keep_prob = 1.0

keep_prob = tf.placeholder("float")
x = tf.placeholder(tf.float32, [None, num_pixel])
y_ = tf.placeholder("float", [None, 10])

conv_W = list(range(CONV_LAYERS + 1))
conv_b = list(range(CONV_LAYERS + 1))
for i in range(CONV_LAYERS):
    conv_W[i + 1] = weight_variable([5, 5, CONV_L[i], CONV_L[i + 1]])
    conv_b[i + 1] = bias_variable([CONV_L[i + 1]])

conv_yt = list(range(CONV_LAYERS + 1))
conv_yt[0] = tf.reshape(x, [-1, 28, 28, CONV_L[0]])
for i in range(CONV_LAYERS):
    conv_yt[i + 1] = tf.nn.relu(conv2d(conv_yt[i], conv_W[i + 1]) + conv_b[i + 1])
    conv_yt[i + 1] = max_pool_2x2(conv_yt[i + 1])

dense_W = list(range(DENSE_LAYERS + 1))
dense_b = list(range(DENSE_LAYERS + 1))
for i in range(DENSE_LAYERS):
    dense_W[i + 1] = weight_variable([DENSE_L[i], DENSE_L[i + 1]])
    dense_b[i + 1] = bias_variable([DENSE_L[i + 1]])

dense_yt = list(range(DENSE_LAYERS + 1))
dense_yt[0] = tf.reshape(conv_yt[CONV_LAYERS], [-1, DENSE_L[0]])
for i in range(DENSE_LAYERS):
    if i == DENSE_LAYERS - 1:
        dense_yt[i] = tf.nn.dropout(dense_yt[i], keep_prob)
    dense_yt[i + 1] = tf.nn.relu(tf.matmul(dense_yt[i], dense_W[i + 1]) + dense_b[i + 1])

y = tf.nn.softmax(dense_yt[DENSE_LAYERS])

l2_loss = 0
for i in range(LAYERS):
    l2_loss += tf.nn.l2_loss(W[i + 1]) + tf.nn.l2_loss(b[i + 1])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
cost_function = cross_entropy + regular_lambda * l2_loss

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
init = tf.global_variables_initializer()

print('Start training ...')
print('Neural Network Layers:', L)
print('Learning Rate:', learning_rate)
print('Iterations:', iterations)
print('Batch Size:', batch_size)
print('Regularization lambda:', regular_lambda)
print('Dropout Keep Probability:', drop_keep_prob)

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
        sess.run(train_step, feed_dict = {x: batch_x, y_: batch_y_, keep_prob: drop_keep_prob})
        if i % (iterations // 100) == 0:
            print('Process: {}%'.format((i // (iterations // 100) + 1) * 1))
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print('Accuracy:', sess.run(accuracy, feed_dict = {x: test_images, y_: test_labels_ten, keep_prob: 1.0}))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print('Accuracy:', sess.run(accuracy, feed_dict = {x: test_images, y_: test_labels_ten, keep_prob: 1.0}))
