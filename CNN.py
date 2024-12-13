import numpy as np


class CNN:
    def __init__(self, input_shape, num_classes):
        """
        Initialize the LeNet-5 model structure as described in the assignment.
        Parameters:
        - input_shape: tuple, the shape of the input images (height, width, channels).
        - num_classes: int, the number of classes in the output layer.
        """

        self.output = None
        self.flattened = None
        self.pool2_out = None
        self.relu2_out = None
        self.pool1_out = None
        self.relu1_out = None

        self.input_shape = input_shape
        self.num_classes = num_classes

        # I am defining the layers of LeNet-5 here. Starting with convolution layers.
        # The first convolutional layer has 6 filters of size 5x5x(input_channels).
        # self.conv1_filters = np.random.randn(6, 5, 5, input_shape[2]) * 0.01
        self.conv1_filters = np.random.normal(
            0, np.sqrt(2 / (6 * 5 * 5 * input_shape[2])), size=(6, 5, 5, input_shape[2])
        )
        self.conv1_biases = np.zeros(6)
        self.conv1_out = None

        # The second convolutional layer has 16 filters of size 5x5x6.
        # self.conv2_filters = np.random.randn(16, 5, 5, 6) * 0.01
        self.conv2_filters = np.random.normal(
            0, np.sqrt(2 / (16 * 5 * 5 * 6)), size=(16, 5, 5, 6)
        )
        self.conv2_biases = np.zeros(16)
        self.conv2_out = None

        # Fully connected layers are defined with their weights and biases.
        # Sizes are based on the output of the previous layers.
        # self.fc1_weights = np.random.randn(120, 400) * 0.01
        self.fc1_weights = np.random.normal(0, np.sqrt(2 / 400), size=(120, 400))
        self.fc1_biases = np.zeros(120)
        self.fc1_out = None

        # self.fc2_weights = np.random.randn(84, 120) * 0.01
        self.fc2_weights = np.random.normal(0, np.sqrt(2 / 120), size=(84, 120))
        self.fc2_biases = np.zeros(84)
        self.fc2_out = None

        # self.output_weights = np.random.randn(num_classes, 84) * 0.01
        self.output_weights = np.random.normal(
            0, np.sqrt(2 / 84), size=(num_classes, 84)
        )
        self.output_biases = np.zeros(num_classes)

    def relu(self, x):
        """Apply ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU used during backpropagation."""
        return (x > 0).astype(float)

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def max_pool(self, x, pool_size, stride):
        """
        Perform max pooling operation.
        - x: Input data.
        - pool_size: Tuple (height, width) of the pooling window.
        - stride: Stride of the pooling window.
        """
        # Calculate output dimensions after pooling.
        batch_size, height, width, channels = x.shape
        pooled_height = (height - pool_size[0]) // stride + 1
        pooled_width = (width - pool_size[1]) // stride + 1

        # Initializing pooled output with zeros.
        pooled = np.zeros((batch_size, pooled_height, pooled_width, channels))

        # Iterating over each pooling window.
        for b in range(batch_size):
            for h in range(0, height - pool_size[0] + 1, stride):
                for w in range(0, width - pool_size[1] + 1, stride):
                    for c in range(channels):
                        window = x[b, h : h + pool_size[0], w : w + pool_size[1], c]
                        pooled[b, h // stride, w // stride, c] = np.max(window)

        return pooled

    def convolve(self, x, filters, biases, stride=1, padding=0):
        """
        Perform convolution operation.
        - x: Input data (batch_size, height, width, channels).
        - filters: Filters to apply (num_filters, filter_height, filter_width, input_channels).
        - biases: Biases for each filter.
        - stride: Stride of the convolution.
        - padding: Number of zero-padding rows/columns to add to the input.
        """
        # Adding padding to input.
        if padding > 0:
            x = np.pad(
                x,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        batch_size, height, width, channels = x.shape
        num_filters, filter_height, filter_width, _ = filters.shape

        # Calculating the output dimensions.
        output_height = (height - filter_height) // stride + 1
        output_width = (width - filter_width) // stride + 1

        # Initializing output tensor.
        output = np.zeros((batch_size, output_height, output_width, num_filters))

        # Iterating over the input to perform convolution.
        for b in range(batch_size):
            for h in range(0, height - filter_height + 1, stride):
                for w in range(0, width - filter_width + 1, stride):
                    for f in range(num_filters):
                        region = x[b, h : h + filter_height, w : w + filter_width, :]
                        output[b, h // stride, w // stride, f] = (
                            np.sum(region * filters[f]) + biases[f]
                        )

        return output

    def forward(self, x):
        """
        Perform forward propagation through the LeNet-5 model.
        - x: Input data (batch_size, height, width, channels).
        """
        # First convolutional layer + ReLU activation.
        self.conv1_out = self.convolve(
            x, self.conv1_filters, self.conv1_biases, stride=1, padding=0
        )
        self.relu1_out = self.leaky_relu(self.conv1_out)

        # First max pooling layer.
        self.pool1_out = self.max_pool(self.relu1_out, pool_size=(2, 2), stride=2)

        # Second convolutional layer + ReLU activation.
        self.conv2_out = self.convolve(
            self.pool1_out, self.conv2_filters, self.conv2_biases, stride=1, padding=0
        )
        self.relu2_out = self.leaky_relu(self.conv2_out)

        # Second max pooling layer.
        self.pool2_out = self.max_pool(self.relu2_out, pool_size=(2, 2), stride=2)

        # Flatten the output for the fully connected layers.
        self.flattened = self.pool2_out.reshape(self.pool2_out.shape[0], -1)

        # First fully connected layer + ReLU activation.
        self.fc1_out = self.relu(
            np.dot(self.fc1_weights, self.flattened.T).T + self.fc1_biases
        )

        # Second fully connected layer + ReLU activation.
        self.fc2_out = self.relu(
            np.dot(self.fc2_weights, self.fc1_out.T).T + self.fc2_biases
        )

        # Output layer (identity activation).
        self.output = np.dot(self.output_weights, self.fc2_out.T).T + self.output_biases

        # for debugging purpose
        # print("Shape after first conv:", x.shape)
        # print("Shape after first max pool:", x.shape)
        # print("Shape after second conv:", x.shape)
        # print("Shape after second max pool:", x.shape)

        return self.output

    # backpropagation method
    def backprop(self, x, y_true, learning_rate=0.001):
        """
        Perform backpropagation and update the weights for all layers.
        - x: Input data (batch_size, height, width, channels).
        - y_true: True labels (batch_size, num_classes).
        - learning_rate: Learning rate for gradient descent.
        """

        # Softmax activation
        probs = self.softmax(self.output)

        # compute cross-entropy loss
        loss, dL_dlogits = self.cross_entropy(probs, y_true)

        # gradients for output layer
        dL_doutput_weights = np.dot(dL_dlogits.T, self.fc2_out)
        dL_doutput_biases = np.sum(dL_dlogits, axis=0)
        dL_dfc2_out = np.dot(dL_dlogits, self.output_weights)

        # gradients for second fully connected layer
        dL_dfc2_out_relu = dL_dfc2_out * self.relu_derivative(self.fc2_out)
        dL_dfc2_weights = np.dot(dL_dfc2_out_relu.T, self.fc1_out)
        dL_dfc2_biases = np.sum(dL_dfc2_out_relu, axis=0)
        dL_dfc1_out = np.dot(dL_dfc2_out_relu, self.fc2_weights)

        # gradients for first fully connected layer
        dL_dfc1_out_relu = dL_dfc1_out * self.relu_derivative(self.fc1_out)
        dL_dfc1_weights = np.dot(dL_dfc1_out_relu.T, self.flattened)
        dL_dfc1_biases = np.sum(dL_dfc1_out_relu, axis=0)
        dL_dflattened = np.dot(dL_dfc1_out_relu, self.fc1_weights)
        print(f"dL_dfc1_out min = {dL_dfc1_out.min()}, max = {dL_dfc1_out.max()}")

        # gradients for flattened input
        dL_dpool2_out = dL_dflattened.reshape(self.pool2_out.shape)

        # backprop through max pool 2
        dL_drelu2_out = self.backprop_max_pool(
            dL_dpool2_out, self.relu2_out, pool_size=(2, 2), stride=2
        )

        # backprop through ReLU 2
        dL_dconv2_out = dL_drelu2_out * self.leaky_relu_derivative(self.conv2_out)

        # backprop through convolution 2
        dL_dpool1_out, dL_dconv2_filters, dL_dconv2_biases = self.back_prop_single_conv(
            self.pool1_out, dL_dconv2_out, self.conv2_filters
        )
        print(f"dL_dpool2_out min={dL_dpool1_out.min()}, max={dL_dpool1_out.max()}")

        # backprop through max pool 1
        dL_drelu1_out = self.backprop_max_pool(
            dL_dpool1_out, self.relu1_out, pool_size=(2, 2), stride=2
        )

        # backprop through ReLU 1
        dL_dconv1_out = dL_drelu1_out * self.leaky_relu_derivative(self.conv1_out)

        # backprop through convolution 1
        _, dL_dconv1_filters, dL_dconv1_biases = self.back_prop_single_conv(
            x, dL_dconv1_out, self.conv1_filters
        )

        # update
        self.output_weights -= learning_rate * dL_doutput_weights
        self.output_biases -= learning_rate * dL_doutput_biases
        self.fc2_weights -= learning_rate * dL_dfc2_weights
        self.fc2_biases -= learning_rate * dL_dfc2_biases
        self.fc1_weights -= learning_rate * dL_dfc1_weights
        self.fc1_biases -= learning_rate * dL_dfc1_biases
        self.conv2_filters -= learning_rate * dL_dconv2_filters
        self.conv2_biases -= learning_rate * dL_dconv2_biases
        self.conv1_filters -= learning_rate * dL_dconv1_filters
        self.conv1_biases -= learning_rate * dL_dconv1_biases

        return loss

    def backprop_max_pool(self, dL_dout, x, pool_size=(2, 2), stride=2):
        # initialize the gradient
        dL_in = np.zeros_like(x)

        batch_size, height, width, channels = x.shape
        out_height, out_width = dL_dout.shape[1:3]

        # iterate through each input in the batch
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        # compute the starting indices of the window in the input
                        h_start = h_out * stride
                        w_start = w_out * stride
                        h_end = min(h_start + pool_size[0], height)
                        w_end = min(w_start + pool_size[1], width)

                        # the current window
                        window = x[b, h_start:h_end, w_start:w_end, c]

                        # find the index of the maximum value in the window
                        max_idx = np.unravel_index(np.argmax(window), window.shape)

                        # map the gradient back to the maximum value position in the input
                        dL_in[
                            b, h_start + max_idx[0], w_start + max_idx[1], c
                        ] += dL_dout[b, h_out, w_out, c]

        return dL_in

    def backprop_kernel_gradient(self, input_data, dL_dY, kernels, stride=1, padding=1):
        dL_dK = np.zeros_like(kernels)  # gradient with respect to the kernels
        batch_size = input_data.shape[0]
        in_depth = input_data.shape[3]
        num_kernels = kernels.shape[0]
        for batch in range(batch_size):
            for j in range(num_kernels):
                for i in range(in_depth):
                    dL_dK[j, :, :, i] += convolve2d(
                        input_data[batch, :, :, i], dL_dY[batch, :, :, j], 0
                    )
        dL_dK = dL_dK / batch_size
        return dL_dK
        ### ALTERNATE dL/dK
        # for j in range(kernels.shape[3]):
        #     for i in range(kernels.shape[0]):
        #         res = self.convolve(
        #             input_data[:, :, :, i],
        #             dL_dY[:, :, :, j],
        #             np.zeros(1),
        #             stride=1,
        #             padding=0,
        #         )
        # dL_dK = self.convolve(
        #     dL_dY, input_data, biases=np.zeros(input_data.shape[1]), stride=1, padding=0
        # )
        ###

    def backprop_input_gradient(self, input_data, dL_dY, kernels, stride=1, padding=1):
        padding = kernels.shape[1] - 1

        dL_dX = np.zeros_like(input_data)  # gradient with respect to the input

        rotated_kernels = np.zeros_like(kernels)

        # rotate all the kernels by 180 degrees
        for kernel_indx in range((kernels.shape[0])):
            current_kernel = kernels[kernel_indx]
            rotated_kernels[kernel_indx] = np.rot90(
                current_kernel, 2, (0, 1)
            )  # rotate the kernel by 180 degrees

        batch_size = input_data.shape[0]
        in_depth = input_data.shape[3]
        num_kernels = kernels.shape[0]
        for batch in range(batch_size):
            for i in range(in_depth):
                for j in range(num_kernels):
                    dL_dX[batch, :, :, i] += convolve2d(
                        dL_dY[batch, :, :, j], rotated_kernels[j, :, :, i], padding
                    )
        return dL_dX
        ### ALTERNATE dL/dX
        # Build up dL/dX one channel at a time
        # for i in range((input_data.shape[3])):
        #     # Pick the ith channel out of every kernel
        #     expanded = np.expand_dims(rotated_kernels[:, :, :, i], axis=3)
        #     # (p x n x n x 1) -> (1 x n x n x p) to match the shape convolve expects
        #     swapped = np.swapaxes(expanded, 0, 3)
        #     res = self.convolve(
        #         dL_dY,
        #         swapped,
        #         biases=np.zeros(kernels.shape[0]),
        #         stride=stride,
        #         padding=padding2,
        #     )
        #     dL_dX[:, :, :, i] = np.squeeze(res)

    def back_prop_single_conv(self, input_data, dL_dY, kernels, stride=1, padding=1):

        padding2 = kernels.shape[1] - 1

        dL_dX = np.zeros_like(input_data)  # gradient with respect to the input
        dL_dK = np.zeros_like(kernels)  # gradient with respect to the kernels
        dL_db = np.sum(
            dL_dY, axis=(0, 1, 2)
        )  # gradient with respect to the biases (One per kernel)

        rotated_kernels = np.zeros_like(kernels)

        # rotate all the kernels by 180 degrees
        for kernel_indx in range((kernels.shape[0])):
            current_kernel = kernels[kernel_indx]
            rotated_kernels[kernel_indx] = np.rot90(
                current_kernel, 2, (0, 1)
            )  # rotate the kernel by 180 degrees

        # compute the dL/dX by convolving dL/dY with all the rotated kernels
        # dL_dX = self.convolve(
        #     dL_dY,  # we convolve over the output
        #     rotated_kernels,
        #     biases=np.zeros(kernels.shape[0]),
        #     stride=stride,
        #     padding=padding2,
        # )

        other_dL_dX = self.backprop_input_gradient(
            input_data, dL_dY, kernels, stride, padding
        )
        dL_dX = other_dL_dX
        # comparison = np.allclose(dL_dX, other, 1e-100)

        # get lengths and dimensions to iterate over
        b, h_in, w_in, c_in = (
            input_data.shape
        )  # b = batch size, h = input height, w = input width , c = input channels
        f, k_h, k_w, _ = (
            kernels.shape
        )  # f - filters, h - kernal height, w - keneral width
        _, h_out, w_out, _ = dL_dY.shape  # h - output height , w- output width

        # padded = np.pad(
        #     input_data, (0, 0), (padding, padding), (padding, padding), (0, 0)
        # )  # add padding that was done during the forward pass, index matching

        ### CHANGED
        padded = np.pad(
            input_data,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            "constant",
            constant_values=0,
        )

        # dL/dk (f, k_h,w_h,c) = sigma b,h_out,w_out over x(b,h_out*stride+k_h,w_out*stride+k_w,c)*dl/dy(b,h_out,w_out,f)
        # for f in range(f):  # loop kernal
        #     for k_h_input in range(k_h):  # loop height
        #         for k_w_input in range(k_w):  # loop width
        #             for c in range(c_in):  # loop input
        #                 val = 0.0  # gradient accumulation
        #                 for b in range(b):
        #                     for h_out_i in range(h_out):
        #                         for w_out_i in range(
        #                             w_out
        #                         ):  # for the output compute the corresponding corrd
        #                             h_in = h_out_i * stride + k_h_input
        #                             w_in = w_out_i * stride + k_w_input
        #                             val += (
        #                                 padded[b, h_in, w_in, c]
        #                                 * dL_dY[b, h_out_i, w_out_i, f]
        #                             )
        #                 dL_dK[f, k_h_input, k_w_input, c] = (
        #                     val  # store grad for position
        #                 )

        other_dL_dK = self.backprop_kernel_gradient(
            input_data, dL_dY, kernels, stride, padding
        )
        dL_dK = other_dL_dK
        # comparison = np.allclose(dL_dK, other_dL_dX, 1e-100)
        bp = 1
        return dL_dX, dL_dK, dL_db

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy(self, probs, y_true):

        batch_size = y_true.shape[0]

        # convert one-hot encoded y_true to class indices
        if len(y_true.shape) > 1:
            y_true_indices = np.argmax(y_true, axis=1)
        else:
            y_true_indices = y_true

        # advoid modifying probs directly to prevent side effects
        dL_dlogits = np.copy(probs)

        # select the probabilities corresponding to the true classes
        correct_probs = probs[range(batch_size), y_true_indices]

        # compute the cross-entropy loss
        loss = -np.sum(np.log(correct_probs)) / batch_size

        # compute the gradient with respect to logits
        dL_dlogits[range(batch_size), y_true_indices] -= 1
        dL_dlogits /= batch_size

        return loss, dL_dlogits

    # for f in range(kernels.shape[0]):
    #     dL_dK[kernel_indx] = self.convolve( # we convolve over the input for the current partial
    #         input_data,
    #         dL_dY[kernel_indx],
    #         biases = None,
    #         stride=stride,
    #         padding=padding
    #     )

    # need to rotate the kernel by 180 degrees


# Trying to debug backprop_single_conv so made something simpler than the batched/3d conv function
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
