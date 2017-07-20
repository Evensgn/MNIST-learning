import numpy as np
import matplotlib.pyplot as plt
import cv2

NUM_ROW = 28
NUM_COLUMN = 28
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

# deskewing function by Satya Mallick
# from http://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * NUM_ROW * skew], [0, 1, 0]]) # require that NUM_ROW == NUM_COLUMN
    img = cv2.warpAffine(img, M, (NUM_ROW, NUM_ROW), flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def show_large_img(img):
    f = lambda i, j: img[i // 10, j // 10]
    imglarge = np.fromfunction(f, (img.shape[0] * 10, img.shape[1] * 10), dtype = np.uint8)        
    cv2.imshow('Image', imglarge)
    cv2.waitKey(0)

print('Deskewing train images:', len(train_images))
for i in range(len(train_images)):
	img = train_images[i]
	img = np.array(img, dtype = np.uint8)
	img.resize(NUM_ROW, NUM_COLUMN)
	if True and i < 3:
		show_large_img(img)
	img = deskew(img)
	if True and i < 3:
		show_large_img(img)
	train_images[i] = img

print('Deskewing test images:', len(test_images))
for i in range(len(test_images)):
	img = test_images[i]
	img = np.array(img, dtype = np.uint8)
	img.resize(NUM_ROW, NUM_COLUMN)
	if True and i < 3:
		show_large_img(img)
	img = deskew(img)
	if True and i < 3:
		show_large_img(img)
	test_images[i] = img

data_filename = 'data_deskewed.pkl'
print('Dumping data to file \'' + data_filename + '\' ...')
with open(data_filename, 'wb') as f:
    pickle.dump(train_labels, f)
    pickle.dump(train_images, f)
    pickle.dump(test_labels, f)
    pickle.dump(test_images, f)
    pickle.dump(num_pixel, f)
print('Data dumping complete.')
