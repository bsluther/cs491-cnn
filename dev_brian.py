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
# print(a.shape)
# print(b)


def convolve2d(x, filter, padding):
    if padding > 0:
        x = np.pad(
            x,
            ((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
    in_height, in_width = x.shape
    f_height, f_width = filter.shape
    out_height = in_height - f_height + 1
    out_width = in_width - f_width + 1
    output = np.zeros((out_height, out_width))
    for row_offset in range(out_height):
        for col_offset in range(out_width):
            region = x[
                row_offset : row_offset + f_height, col_offset : col_offset + f_width
            ]
            output[row_offset, col_offset] = np.sum(region * filter)
    return output


x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
f = np.array([[0, 1], [2, 3]])
res = convolve2d(x, f, 1)
# print(res)
# print(x / 2)


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
