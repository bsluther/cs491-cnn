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
b = a[1]
# print(a.shape)
# print(b.reshape(b.shape[0], -1))


def make_convolution_row(F, input_len):
    filter_len = F.shape[0]
    padded = np.pad(F, ((0, 0), (0, input_len - filter_len)), "constant")
    flattened = padded.flatten()
    with_zeros = np.append(flattened, np.zeros(input_len * (input_len - filter_len)))
    return with_zeros


def make_convolution_matrix(F, input_len):
    filter_len = F.shape[0]
    output_len = input_len - filter_len + 1

    C = np.empty((0, input_len * input_len))

    conv_template = make_convolution_row(F, input_len)

    for i in range(0, output_len):
        for j in range(0, output_len):
            rolled = np.roll(conv_template, i * input_len + j)
            C = np.vstack([C, rolled])

    return C


def convolve_2d_mtx(x, filter, padding):
    x_padded = x
    if padding > 0:
        x_padded = np.pad(
            x,
            ((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
    n = x_padded.shape[0]
    m = filter.shape[0]
    c = make_convolution_matrix(filter, x_padded.shape[0])
    result = c @ x_padded.flatten()
    return result.reshape((n - m + 1, n - m + 1))


x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
f = np.array([[0, 1], [2, 3]])
# print(convolve_2d_mtx(x, f, 0))


def example_slides():
    X = np.array(
        [
            [0, 4, 2, 1, 3],
            [-1, 0, 1, -2, 2],
            [3, 1, 2, 0, 1],
            [0, 1, 4, 1, 2],
            [2, 3, 1, 1, 0],
        ]
    )
    K = np.array([[1, 0], [-1, 2]])

    conv_mtx = make_convolution_matrix(K, X.shape[0])

    f_bar = X.flatten()
    o = conv_mtx @ f_bar + np.ones(conv_mtx.shape[0])
    print(o)
