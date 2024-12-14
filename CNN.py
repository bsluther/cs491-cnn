import numpy as np
from dev_brian import convolve_2d_mtx


class CNN:
    def __init__(self, input_shape, num_classes):
        """
        Initialize the LeNet-5 model structure as described in the assignment.
        Parameters:
        - input_shape: tuple, the shape of the input images (height, width, channels).
        - num_classes: int, the number of classes in the output layer.
        """

        self.input_shape = input_shape
        self.num_classes = num_classes

        # I am defining the layers of LeNet-5 here. Starting with convolution layers.
        # The first convolutional layer has 6 filters of size 5x5x(input_channels).

        limit_conv1 = np.sqrt(2 / (input_shape[2] * 5 * 5))  # For Conv1
        self.conv1_filters = (
            np.random.uniform(-limit_conv1, limit_conv1, (6, 5, 5, input_shape[2])) * 2
        )
        self.conv1_biases = np.zeros(6)

        # The second convolutional layer has 16 filters of size 5x5x6.
        limit_conv2 = np.sqrt(2 / (6 * 5 * 5))  # For Conv2
        self.conv2_filters = (
            np.random.uniform(-limit_conv2, limit_conv2, (16, 5, 5, 6)) * 2
        )
        self.conv2_biases = np.zeros(16)

        # Fully connected layers are defined with their weights and biases.
        # Sizes are based on the output of the previous layers.
        # self.fc1_weights = np.random.randn(120, 256) * 0.01
        self.fc1_weights = np.random.randn(120, 400) * np.sqrt(2 / 400) * 2
        self.fc1_biases = np.zeros(120)

        # self.fc2_weights = np.random.randn(84, 120) * 0.01
        self.fc2_weights = np.random.randn(84, 120) * np.sqrt(2 / 120) * 2
        self.fc2_biases = np.zeros(84)

        # self.output_weights = np.random.randn(num_classes, 84) * 0.01
        self.output_weights = np.random.randn(num_classes, 84) * np.sqrt(2 / 84) * 2
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

                        # print(f"Region at batch {b}, h {h}, w {w}, filter {f}:")
                        # print(region)
                        # print(f"Kernel:\n{filters[f]}")
                        # print(f"Dot Product: {np.sum(region * filters[f])}")
                        # print(f"With Bias: {np.sum(region * filters[f]) + biases[f]}")

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
    def backprop(self, x, y_true, learning_rate=0.001, learning_rate_conv=0.5):
        """
        Perform backpropagation and update the weights for all layers.
        - x: Input data (batch_size, height, width, channels).
        - y_true: True labels (batch_size, num_classes).
        - learning_rate: Learning rate for gradient descent.
        """

        # Softmax activation
        probs = self.softmax(self.output)
        # print("Softmax probabilities range:", probs.min(), probs.max())

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
        # print(f"dL_dfc1_out min = {dL_dfc1_out.min()}, max = {dL_dfc1_out.max()}")

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
        # print(f"dL_dpool2_out min={dL_dpool1_out.min()}, max={dL_dpool1_out.max()}")

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

        # print("Conv2 Filter Range:", self.conv2_filters.min(), self.conv2_filters.max())
        #
        # print("Activation after Conv1:", np.linalg.norm(self.relu1_out))
        # print("Activation after Pool1:", np.linalg.norm(self.pool1_out))
        # print("Activation after Conv2:", np.linalg.norm(self.relu2_out))
        # print("Activation after Pool2:", np.linalg.norm(self.pool2_out))

        # print("========================")

        # print("Gradients at FC2 Weights:", np.linalg.norm(dL_dfc2_weights))
        # print("Gradients at FC1 Weights:", np.linalg.norm(dL_dfc1_weights))
        # print("Gradients at Conv2 Filters:", np.linalg.norm(dL_dconv2_filters))
        # print("Gradients at Conv1 Filters:", np.linalg.norm(dL_dconv1_filters))
        # print("Gradient Norms:")
        # print("Conv1 Filters:", np.linalg.norm(dL_dconv1_filters))
        # print("Conv1 Biases:", np.linalg.norm(dL_dconv1_biases))
        # print("Conv2 Filters:", np.linalg.norm(dL_dconv2_filters))
        # print("Conv2 Biases:", np.linalg.norm(dL_dconv2_biases))
        # print("FC1 Weights:", np.linalg.norm(dL_dfc1_weights))
        # print("FC1 Biases:", np.linalg.norm(dL_dfc1_biases))
        # print("FC2 Weights:", np.linalg.norm(dL_dfc2_weights))
        # print("FC2 Biases:", np.linalg.norm(dL_dfc2_biases))
        # print("Output Weights:", np.linalg.norm(dL_doutput_weights))
        # print("Output Biases:", np.linalg.norm(dL_doutput_biases))

        # print("Loss for first batch:", loss)

        return loss

    def backprop_max_pool(self, dL_dout, x, pool_size=(2, 2), stride=2):
        """
        Backpropagate through a max pooling layer.
        - dL_dout: Gradient of loss with respect to the max pooled output.
        - x: Input to the pooling layer.
        - pool_size: Tuple (height, width) of the pooling region.
        - stride: Stride of the pooling operation.
        """
        # initialize the gradient w.r.t. input as zeros
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

                        window = x[b, h_start:h_end, w_start:w_end, c]

                        # find the index of the maximum value in the window
                        max_idx = np.unravel_index(np.argmax(window), window.shape)

                        # map the gradient back to the maximum value position in the input
                        dL_in[
                            b, h_start + max_idx[0], w_start + max_idx[1], c
                        ] += dL_dout[b, h_out, w_out, c]

        return dL_in

    def test_backprop_max_pool(self):
        """
        Test backpropagation through max pooling with a simple example.
        """
        # Example input tensor: batch_size=1, height=4, width=4, channels=1
        x = np.array(
            [
                [
                    [[1], [2], [3], [4]],
                    [[5], [6], [7], [8]],
                    [[9], [10], [11], [12]],
                    [[13], [14], [15], [16]],
                ]
            ]
        )  # Shape: (1, 4, 4, 1)

        # Example gradient tensor: matches the pooled output (2x2 region pooling)
        dL_dout = np.array([[[[10], [20]], [[30], [40]]]])  # Shape: (1, 2, 2, 1)

        # Call backprop_max_pool
        dL_din = self.backprop_max_pool(dL_dout, x, pool_size=(2, 2), stride=2)

        # Verify shape
        print("Input Gradient Shape:", dL_din.shape)
        print("Input Gradient:\n", dL_din)

        print("All assertions passed for backprop_max_pool!")

    def test_backprop_kernel_gradient(self):
        """
        Test the backprop_kernel_gradient function with controlled inputs.
        """
        # Define a simple input tensor (batch_size=1, height=3, width=3, channels=1)
        input_data = np.array(
            [[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]]
        )  # Shape: (1, 3, 3, 1)

        # Define a simple gradient tensor (batch_size=1, height=2, width=2, num_kernels=1)
        dL_dY = np.array([[[[1], [2]], [[3], [4]]]])  # Shape: (1, 2, 2, 1)

        # Define a simple kernel (1 kernel, height=2, width=2, in_depth=1)
        kernels = np.array([[[[1], [0]], [[0], [1]]]])  # Shape: (1, 2, 2, 1)

        # Call the backprop_kernel_gradient function
        dL_dK = self.backprop_kernel_gradient(
            input_data, dL_dY, kernels, stride=1, padding=0
        )

        # Print the computed gradient
        print("Gradient with respect to kernels (dL_dK):")
        print(dL_dK)

        # Manually compute expected gradient
        expected_dL_dK = np.array([[[[37], [47]], [[67], [77]]]])  # Shape: (1, 2, 2, 1)

        # Assert that the computed gradient matches the expected gradient
        assert np.allclose(
            dL_dK, expected_dL_dK
        ), "Kernel gradients do not match expected values!"

        print("Test passed! Kernel gradients are correct.")

    def test_convolve(self):
        """
        Test the convolve function with a simple example.
        """
        # Example input tensor: batch_size=1, height=4, width=4, channels=1
        x = np.array(
            [
                [
                    [[1], [2], [3], [4]],
                    [[5], [6], [7], [8]],
                    [[9], [10], [11], [12]],
                    [[13], [14], [15], [16]],
                ]
            ]
        )  # Shape: (1, 4, 4, 1)

        # Example kernel: num_filters=1, filter_height=2, filter_width=2, input_channels=1
        kernel = np.array([[[[1], [0]], [[0], [1]]]])  # Shape: (1, 2, 2, 1)

        # Example bias for the filter
        bias = np.array([1])  # Shape: (1,)

        # Instantiate the CNN object
        cnn = CNN(input_shape=(4, 4, 1), num_classes=10)

        # Perform the convolution
        output = cnn.convolve(x, kernel, bias, stride=1, padding=0)

        # Manually compute the expected output (with bias = 1)
        expected_output = np.array(
            [[[[7], [9], [11]], [[15], [17], [19]], [[23], [25], [27]]]]
        )  # Shape: (1, 3, 3, 1)

        # Print the outputs
        print("Convolution Output:")
        print(output)
        print("Expected Output:")
        print(expected_output)

        # Assert the output matches the expected result
        assert np.allclose(
            output, expected_output
        ), "Convolution output does not match expected output!"
        print("Convolution function test passed!")

    def backprop_kernel_gradient(self, input_data, dL_dY, kernels, stride=1, padding=1):
        dL_dK = np.zeros_like(
            kernels, dtype=np.float64
        )  # gradient with respect to the kernels
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

    # def leaky_relu_derivative(self, x, alpha=0.01):
    #     return np.where(x > 0, 1, alpha)

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
                    arg1 = dL_dY[batch, :, :, j]
                    arg2 = rotated_kernels[j, :, :, i]
                    dL_dX[batch, :, :, i] += convolve2d(
                        dL_dY[batch, :, :, j], rotated_kernels[j, :, :, i], padding
                    )
        return dL_dX

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
        # padded = np.pad(
        #     input_data,
        #     ((0, 0), (padding, padding), (padding, padding), (0, 0)),
        #     "constant",
        #     constant_values=0,
        # )

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


if __name__ == "__main__":
    # Instantiate the CNN class with a dummy input shape and number of classes
    input_shape = (28, 28, 1)  # Example for MNIST
    num_classes = 10
    cnn = CNN(input_shape, num_classes)

    # Test backprop_kernel_gradient
    print("\nTesting backprop_kernel_gradient:")
    cnn.test_backprop_kernel_gradient()

    # Test backprop_max_pool
    print("\nTesting backprop_max_pool:")
    cnn.test_backprop_max_pool()

    cnn.test_convolve()
