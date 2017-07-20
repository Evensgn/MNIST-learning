import gzip
import numpy as np
import struct
import matplotlib.pyplot as plt
import cv2

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

def show_large_img(img):
    f = lambda i, j: img[i // 10, j // 10]
    imglarge = np.fromfunction(f, (img.shape[0] * 10, img.shape[1] * 10), dtype = np.uint8)        
    cv2.imshow('Image', imglarge)
    cv2.waitKey(0)

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
        if True and i < 3:
            img = np.array(img, dtype = np.uint8)
            img.resize(num_row, num_column)
            show_large_img(img)
    print('Read images :', num_image)
    return images, num_pixel

## read data from file
train_labels = read_labels_from_file('train-labels-idx1-ubyte.gz')
train_images, num_pixel = read_images_from_file('train-images-idx3-ubyte.gz')
test_labels = read_labels_from_file('t10k-labels-idx1-ubyte.gz')
test_images, num_pixel = read_images_from_file('t10k-images-idx3-ubyte.gz')

import pickle

data_filename = 'data.pkl'
print('Dumping data to file \'' + data_filename + '\' ...')
with open(data_filename, 'wb') as f:
    pickle.dump(train_labels, f)
    pickle.dump(train_images, f)
    pickle.dump(test_labels, f)
    pickle.dump(test_images, f)
    pickle.dump(num_pixel, f)
print('Data dumping complete.')
