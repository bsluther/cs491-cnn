import numpy as np
from CNN import CNN

input_shape = (32, 32, 3)
num_classes = 10
lenet5 = CNN(input_shape, num_classes)

ex_filter = np.array([[0, 1], [2, 3]]).reshape((2, 2, 1))

batch_size = 1
num_filters = 1
input_channels = 1
input_height = 3
input_width = 3
filters = np.zeros(
    (num_filters, ex_filter.shape[0], ex_filter.shape[1], input_channels)
)
filters[0] = ex_filter
biases = np.zeros(input_channels)
stride = 1
padding = 1

ex_X = np.zeros((batch_size, input_height, input_width, input_channels))
ex_input = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).reshape((3, 3, 1))
ex_X[0] = ex_input

out = lenet5.convolve(ex_X, filters, biases, stride, padding)
bp = 1
