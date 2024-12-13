import numpy as np
from CNN import CNN, convolve2d

# Checking correctness of convolution function

net = CNN((32, 32, 3), 10)

x_0 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
x = np.dstack((x_0.copy(), x_0.copy(), x_0.copy()))

f_0 = np.array([[0.0, 1.0], [2.0, 3.0]])
f = np.dstack((f_0.copy(), f_0.copy(), f_0.copy()))

res1 = net.convolve(x.reshape((1, 3, 3, 3)), f.reshape((1, 2, 2, 3)), np.zeros(3))
# print(res1)


def convolve3d(a, b):
    a_height = a.shape[0]
    a_depth = a.shape[2]
    b_height = b.shape[0]
    b_depth = b.shape[2]
    if a_depth != b_depth:
        return None
    result = np.zeros((a_height - b_height + 1, a_height - b_height + 1))
    for depth in range(a_depth):
        result += convolve2d(a[:, :, depth], b[:, :, depth], 0)
    return result


# print(x.shape, f.shape)
def test_backprop_input_gradient():
    net = CNN((32, 32, 3), 10)

    # Make test data
    x_0 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype="float64")
    x = np.dstack((x_0.copy(), x_0.copy(), x_0.copy()))

    f_0 = np.array([[0, 1], [2, 3]], dtype="float64")
    f = np.dstack((f_0.copy(), f_0.copy(), f_0.copy()))

    y = convolve3d(x, f)
    dL_dY = np.ones((2, 2))

    # Compute the gradient for test data
    dL_dX = net.backprop_input_gradient(
        x.reshape((1, 3, 3, 3)),
        dL_dY.reshape((1, 2, 2, 1)),
        f.reshape((1, 2, 2, 3)),
        1,
        1,
    )

    expected_0 = np.array([[0, 1, 1], [2, 6, 4], [2, 5, 3]], dtype="float64")
    expected = np.dstack((expected_0.copy(), expected_0.copy(), expected_0.copy()))[
        np.newaxis, ...
    ]
    result = "Passed" if np.allclose(dL_dX, expected, 0e-7) else "Failed"
    print(f"Test 'backprop_input_gradient': {result}")
    # print(expected[0, :, :, 0])
    # print(dL_dX[0, :, :, 0])


test_backprop_input_gradient()
