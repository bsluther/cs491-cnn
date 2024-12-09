import numpy as np

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
        self.conv1_filters = np.random.randn(6, 5, 5, input_shape[2]) * 0.01
        self.conv1_biases = np.zeros(6)
        
        # The second convolutional layer has 16 filters of size 5x5x6.
        self.conv2_filters = np.random.randn(16, 5, 5, 6) * 0.01
        self.conv2_biases = np.zeros(16)
        
        # Fully connected layers are defined with their weights and biases.
        # Sizes are based on the output of the previous layers.
        self.fc1_weights = np.random.randn(120, 576) * 0.01
        self.fc1_biases = np.zeros(120)
        
        self.fc2_weights = np.random.randn(84, 120) * 0.01
        self.fc2_biases = np.zeros(84)
        
        self.output_weights = np.random.randn(num_classes, 84) * 0.01
        self.output_biases = np.zeros(num_classes)

    def relu(self, x):
        """Apply ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU used during backpropagation."""
        return (x > 0).astype(float)

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
                        window = x[b, h:h + pool_size[0], w:w + pool_size[1], c]
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
            x = np.pad(x, 
                       ((0, 0), (padding, padding), (padding, padding), (0, 0)), 
                       mode='constant', constant_values=0)
        
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
                        region = x[b, h:h + filter_height, w:w + filter_width, :]
                        output[b, h // stride, w // stride, f] = np.sum(region * filters[f]) + biases[f]
        
        return output

    def forward(self, x):
        """
        Perform forward propagation through the LeNet-5 model.
        - x: Input data (batch_size, height, width, channels).
        """
        # First convolutional layer + ReLU activation.
        x = self.convolve(x, self.conv1_filters, self.conv1_biases, stride=1, padding=2)
        x = self.relu(x)
        
        # First max pooling layer.
        x = self.max_pool(x, pool_size=(2, 2), stride=2)
        
        # Second convolutional layer + ReLU activation.
        x = self.convolve(x, self.conv2_filters, self.conv2_biases, stride=1, padding=0)
        x = self.relu(x)
        
        # Second max pooling layer.
        x = self.max_pool(x, pool_size=(2, 2), stride=2)
        
        # Flatten the output for the fully connected layers.
        x = x.reshape(x.shape[0], -1)
        
        # First fully connected layer + ReLU activation.
        x = self.relu(np.dot(self.fc1_weights, x.T).T + self.fc1_biases)
        
        # Second fully connected layer + ReLU activation.
        x = self.relu(np.dot(self.fc2_weights, x.T).T + self.fc2_biases)
        
        # Output layer (identity activation).
        x = np.dot(self.output_weights, x.T).T + self.output_biases

        # for debugging purpose
        print("Shape after first conv:", x.shape)
        print("Shape after first max pool:", x.shape)
        print("Shape after second conv:", x.shape)
        print("Shape after second max pool:", x.shape)

        return x
