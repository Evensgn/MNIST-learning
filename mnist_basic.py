import gzip
import numpy as np
import struct
import matplotlib.pyplot as plt

GRAY_SCALE_RANGE = 255

def read_labels_from_file(filename):
    print('Reading labels from file \'' + filename + '\' ...')
    with gzip.open(filename, 'rb') as f:
        buf = f.read()
    index = 0
    magic_num, num_label = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    labels = []
    # num_label = 1000
    for i in range(num_label):
        ## struct.unpack_from is inefficient
        # labels.append(int(struct.unpack_from('>B', buf, index)[0]))
        # index += struct.calcsize('>B')
        labels.append(buf[index])
        index += 1
    print('Read labels :', num_label)
    return labels

def read_images_from_file(filename):
    print('Reading images from file \'' + filename + '\' ...')
    with gzip.open(filename, 'rb') as f:
        buf = f.read()
    index = 0
    magic_num, num_image, num_row, num_column = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    num_pixel = num_row * num_column
    images = []
    # num_image = 1000
    for i in range(num_image):
        img = []
        for j in range(num_pixel):
            ## struct.unpack_from is inefficient
            # im.append(int(struct.unpack_from('>B', buf, index)[0]))
            # index += struct.calcsize('>B')
            img.append(buf[index])
            index += 1
        images.append(img)
        ## to show images
        ## show image using matplotlib.plt
        if False and i < 3:
            img = np.array(img)
            img.resize(num_row, num_column)
            plt.imshow(img, cmap = 'gray')
            plt.show()
        ## show image using cv2
        if False and i < 3:
            img = np.array(img, dtype = np.uint8)
            img.resize(num_row, num_column)
            f = lambda i, j: img[i // 10, j // 10]
            imglarge = np.fromfunction(f, (num_row * 10, num_column * 10), dtype = np.uint8)
            # imglarge = cv2.copyMakeBorder(imglarge, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value = 255)
            cv2.imshow('Image Sample', imglarge)
            cv2.waitKey(0)
    print('Read images :', num_image)
    return images, num_pixel

## read data from file
train_labels = read_labels_from_file('train-labels-idx1-ubyte.gz')
train_images, num_pixel = read_images_from_file('train-images-idx3-ubyte.gz')
test_labels = read_labels_from_file('t10k-labels-idx1-ubyte.gz')
test_images, num_pixel = read_images_from_file('t10k-images-idx3-ubyte.gz')

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

for i in range(10000):
    batch_x, batch_y_ = next_batch(300)
    sess.run(train_step, feed_dict = {x: batch_x, y_: batch_y_})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print('Accuracy:', sess.run(accuracy, feed_dict = {x: test_images, y_: test_labels_ten}))
