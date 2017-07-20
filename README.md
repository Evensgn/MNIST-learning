# MNIST-learning

Zhou Fan (@Evensgn)

Some machine learning exercises on MNIST data set.

## Usage
Download four MNIST data files (*.gz) from [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/).

Then run the code [read_data.py](read_data.py) to generate pickle file 'data.pkl'.

## Softmax Regression (SGD)
**Code:** [mnist_basic.py](mnist_basic.py)

Parameter | Value
------------ | ----
Learning Rate | 1e-2 
Iterations | 5000
Batch Size | 100
**Accuracy** | **91%**

## Full-Connect Neural Network (SGD)
**Code:** [mnist_nn.py](mnist_nn.py)

Parameter | Value
------------ | ----
Network Layers | 784-580-400-300-200-100-30-10
Learning Rate | 2e-3
Iterations | 40000
Batch Size | 100
Lambda in Regularization | 1e-3
Dropout Keep Probability | 0.9
**Accuracy** | **98.4%**

## Convolutional Neural Network (SGD)
**Code:** [mnist_cnn.py](mnist_cnn.py)

Parameter | Value
----------- | ------
Convolutional Layers | [1, 32, 64]
Densely Connected Layers|  [3136, 1024, 10]
Learning Rate | 0.0005
Iterations | 20000
Batch Size | 50
Regularization lambda | 1e-06
Dropout Keep Probability | 0.5
**Accuracy** | **99.3%**
