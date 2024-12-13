import numpy as np
from CNN import CNN, convolve2d

# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring

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


def make_test_case_1():
    # Make test data
    x_0 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype="float64")
    x = np.dstack((x_0.copy(), x_0.copy(), x_0.copy()))

    f_0 = np.array([[0, 1], [2, 3]], dtype="float64")
    f = np.dstack((f_0.copy(), f_0.copy(), f_0.copy()))

    y = convolve3d(x, f)
    dL_dY = np.ones((2, 2))

    return x, f, y, dL_dY


def test_convolve_case_1():
    net = CNN((32, 32, 3), 10)
    x, f, _, _ = make_test_case_1()

    y = net.convolve(
        x.reshape((1, 3, 3, 3)),
        f.reshape((1, 2, 2, 3)),
        np.zeros(3),
        stride=1,
        padding=0,
    )

    expected = np.array([[57, 75], [111, 129]], dtype="float64").reshape((1, 2, 2, 1))

    result = "Passed" if np.allclose(y, expected, 0e-7) else "Failed"
    print(f"Test 'convolve' (case 1): {result}")
    # print(y[0, :, :, 0])
    # print(expected[0, :, :, 0])


def test_backprop_kernel_gradient_case_1():
    net = CNN((32, 32, 3), 10)

    x, f, y, dL_dY = make_test_case_1()

    # Compute gradient with respect to kernel for test data
    dL_dK = net.backprop_kernel_gradient(
        x.reshape((1, 3, 3, 3)),
        dL_dY.reshape((1, 2, 2, 1)),
        f.reshape((1, 2, 2, 3)),
        1,
        1,
    )

    # Expected results calculcated by hand
    expected_0 = np.array([[8, 12], [20, 24]], dtype="float64")
    expected = np.dstack((expected_0.copy(), expected_0.copy(), expected_0.copy()))[
        np.newaxis, ...
    ]

    result = "Passed" if np.allclose(dL_dK, expected, 0e-7) else "Failed"
    print(f"Test 'backprop_kernel_gradient' (case 1): {result}")
    # print(dL_dK[0, :, :, 0])
    # print(expected[0, :, :, 0])


# print(x.shape, f.shape)
def test_backprop_input_gradient_case_1():
    net = CNN((32, 32, 3), 10)

    x, f, y, dL_dY = make_test_case_1()

    # Compute the gradient with respect to input for test data
    dL_dX = net.backprop_input_gradient(
        x.reshape((1, 3, 3, 3)),
        dL_dY.reshape((1, 2, 2, 1)),
        f.reshape((1, 2, 2, 3)),
        1,
        1,
    )

    # Expected results calculated by hand
    expected_0 = np.array([[0, 1, 1], [2, 6, 4], [2, 5, 3]], dtype="float64")
    expected = np.dstack((expected_0.copy(), expected_0.copy(), expected_0.copy()))[
        np.newaxis, ...
    ]

    result = "Passed" if np.allclose(dL_dX, expected, 0e-7) else "Failed"
    print(f"Test 'backprop_input_gradient' (case 1): {result}")
    # print(expected[0, :, :, 0])
    # print(dL_dX[0, :, :, 0])


def make_test_case_2():
    x_0 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype="float64")
    x_1 = np.array([[1, 4, 6], [2, 0, 7], [3, 5, 8]], dtype="float64")
    x = np.dstack((x_0, x_1)).reshape((1, 3, 3, 2))
    f0_0 = np.array([[0, 1], [2, 3]], dtype="float64")
    f0_1 = np.array([[1, 3], [2, 4]], dtype="float64")
    f0 = np.dstack((f0_0, f0_1))
    f1_0 = np.array([[5, 1], [0, 2]], dtype="float64")
    f1_1 = np.array([[1, 2], [1, 3]], dtype="float64")
    f1 = np.dstack((f1_0, f1_1))
    f = np.array([f0, f1]).reshape((2, 2, 2, 2))

    y_0 = np.array([[36, 75], [65, 106]], dtype="float64")
    y_1 = np.array([[20, 54], [53, 84]], dtype="float64")
    y = np.dstack((y_0, y_1)).reshape((1, 2, 2, 2))

    dL_dY_0 = np.array([[1, 0], [1, 1]], dtype="float64")
    dL_dY_1 = np.array([[2, 1], [0, 1]], dtype="float64")
    dL_dY = np.dstack((dL_dY_0, dL_dY_1)).reshape((1, 2, 2, 2))

    return x, f, y, dL_dY


def test_case_2_convolve():
    net = CNN((3, 3, 2), 10)
    x, f, y_expected, _ = make_test_case_2()
    y_result = net.convolve(x, f, np.zeros(2), stride=1, padding=0)

    # print(y_expected[0, :, :, 1])
    # print(y_result[0, :, :, 1])
    assert np.allclose(y_expected, y_result, 1e-7), "Test case 2 convolution failed"
    print("Test case 2, convolution: passed")


def test_case_2_input_gradient():
    net = CNN((3, 3, 2), 10)
    x, f, _, dL_dY = make_test_case_2()

    expected_dL_dX_0 = np.array([[10, 8, 1], [2, 13, 4], [2, 5, 5]], dtype="float64")
    expected_dL_dX_1 = np.array([[3, 8, 2], [5, 16, 8], [2, 7, 7]], dtype="float64")
    expected_dL_dX = np.dstack((expected_dL_dX_0, expected_dL_dX_1)).reshape(
        (1, 3, 3, 2)
    )

    result_dL_dX = net.backprop_input_gradient(x, dL_dY, f, stride=1, padding=1)

    # print(expected_dL_dX[0, :, :, 1])
    # print(result_dL_dX[0, :, :, 1])
    assert np.allclose(
        expected_dL_dX, result_dL_dX
    ), "Test case 2 input gradient failed"
    print("Test case 2, input gradient: passed")


def test_case_1_kernel_gradient():
    net = CNN((3, 3, 2), 10)
    x, f, _, dL_dY = make_test_case_2()

    expected_dL_dK_0 = np.array([[[7, 10], [16, 19]], [[3, 11], [10, 13]]])
    expected_dL_dK_0 = np.dstack(
        np.array([[[7, 10], [16, 19]], [[3, 11], [10, 13]]], dtype="float64")
    )
    expected_dL_dK_1 = np.array([[[5, 9], [17, 21]], [[6, 21], [9, 15]]])
    expected_dL_dK_1 = np.dstack(
        np.array([[[5, 9], [17, 21]], [[6, 21], [9, 15]]], dtype="float64")
    )

    expected_dL_dK = np.array([expected_dL_dK_0, expected_dL_dK_1])

    result_dL_dK = net.backprop_kernel_gradient(x, dL_dY, f, stride=1, padding=0)

    assert np.allclose(
        expected_dL_dK, result_dL_dK
    ), "Test case 2 kernel gradient failed"
    print("Test case 2, kernel gradient: passed")


test_convolve_case_1()
test_backprop_kernel_gradient_case_1()
test_backprop_input_gradient_case_1()
make_test_case_2()
test_case_2_convolve()
test_case_2_input_gradient()
test_case_1_kernel_gradient()
