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

x = tf.placeholder(tf.float32, [None, num_pixel])
W = tf.Variable(tf.zeros([num_pixel, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

def next_batch(batch_size):
    batch_idx = np.random.choice(range(train_images.shape[0]), size = batch_size, replace = False)
    batch_x = np.zeros((batch_size, num_pixel))
    batch_y_ = np.zeros((batch_size, 10))
    for i in range(batch_size):
        batch_x[i, :] = train_images[batch_idx[i], :]
        batch_y_[i] = train_labels_ten[batch_idx[i]]
    return batch_x, batch_y_

for i in range(5000):
    batch_x, batch_y_ = next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_x, y_: batch_y_})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print('Accuracy:', sess.run(accuracy, feed_dict = {x: test_images, y_: test_labels_ten}))
