import numpy as np
from CNN import CNN, convolve2d

# Checking correctness of convolution function

net = CNN((32, 32, 3), 10)

x_0 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

x = np.dstack((x_0.copy(), x_0.copy(), x_0.copy()))

f_0 = np.array([[0, 1], [2, 3]])
f = np.dstack((f_0.copy(), f_0.copy(), f_0.copy()))

res1 = net.convolve(x.reshape((1, 3, 3, 3)), f.reshape((1, 2, 2, 3)), np.zeros(3))
print(res1)


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


print(x.shape, f.shape)
print(convolve3d(x, f))
