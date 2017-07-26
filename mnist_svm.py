import numpy as np
import matplotlib.pyplot as plt

GRAY_SCALE_RANGE = 255

import pickle

data_filename = 'data_deskewed.pkl'
print('Loading data from file \'' + data_filename + '\' ...')
with open(data_filename, 'rb') as f:
    train_labels = pickle.load(f)
    train_images = pickle.load(f)
    test_labels = pickle.load(f)
    test_images = pickle.load(f)
    num_pixel = pickle.load(f)
print('Data loading complete.')

train_images = np.array(train_images)
train_images.resize(train_images.size // num_pixel, num_pixel)
test_images = np.array(test_images)
test_images.resize(test_images.size // num_pixel, num_pixel)
test_labels = np.array(test_labels)
train_labels = np.array(train_labels)

## normalization
train_images = train_images / GRAY_SCALE_RANGE
test_images = test_images / GRAY_SCALE_RANGE

from sklearn import svm, metrics

# clf = svm.SVC(gamma = 0.001)
clf = svm.SVC(kernel = 'linear')

clf.fit(train_images[:1000], train_labels[:1000])

prediction = clf.predict(test_images)

print("Classification report for classifier %s:\n%s\n"
	  % (clf, metrics.classification_report(test_labels, prediction)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, prediction))