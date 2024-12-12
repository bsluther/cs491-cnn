import numpy as np

# from CNN import CNN

a = np.array(
    [
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[110, 120, 130], [140, 150, 160], [170, 180, 190]],
            [[210, 220, 230], [240, 250, 260], [270, 280, 290]],
        ],
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
            [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
        ],
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
            [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
        ],
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
            [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
        ],
    ]
)
b = a[:, 1, :, :]
print(a.shape)
print(b)

# input_shape = (32, 32, 3)
# num_classes = 10
# lenet5 = CNN(input_shape, num_classes)

# ex_filter = np.array([[0, 1], [2, 3]]).reshape((2, 2, 1))

# batch_size = 1
# num_filters = 1
# input_channels = 1
# input_height = 3
# input_width = 3
# filters = np.zeros(
#     (num_filters, ex_filter.shape[0], ex_filter.shape[1], input_channels)
# )
# filters[0] = ex_filter
# biases = np.zeros(input_channels)
# stride = 1
# padding = 1

# ex_X = np.zeros((batch_size, input_height, input_width, input_channels))
# ex_input = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).reshape((3, 3, 1))
# ex_X[0] = ex_input

# out = lenet5.convolve(ex_X, filters, biases, stride, padding)
# bp = 1
