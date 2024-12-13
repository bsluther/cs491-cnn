import numpy as np

class SimplifiedCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Initialize Conv1
        limit_conv1 = np.sqrt(2 / (input_shape[2] * 5 * 5))
        self.conv1_filters = np.random.uniform(-limit_conv1, limit_conv1, (6, 5, 5, input_shape[2]))
        self.conv1_biases = np.zeros(6)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def convolve(self, x, filters, biases, stride=1, padding=0):
        if padding > 0:
            x = np.pad(
                x,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        batch_size, height, width, channels = x.shape
        num_filters, filter_height, filter_width, _ = filters.shape
        output_height = (height - filter_height) // stride + 1
        output_width = (width - filter_width) // stride + 1
        output = np.zeros((batch_size, output_height, output_width, num_filters))
        for b in range(batch_size):
            for h in range(0, height - filter_height + 1, stride):
                for w in range(0, width - filter_width + 1, stride):
                    for f in range(num_filters):
                        region = x[b, h:h+filter_height, w:w+filter_width, :]
                        output[b, h//stride, w//stride, f] = np.sum(region * filters[f]) + biases[f]
        return output

    def forward(self, x):
        # Conv1 + ReLU
        self.conv1_out = self.convolve(x, self.conv1_filters, self.conv1_biases)
        self.relu1_out = self.relu(self.conv1_out)
        return self.relu1_out

    def compute_loss_and_gradients(self, x, y_true):
        # Forward pass
        output = self.forward(x)
        # Flatten output for simple cross-entropy
        logits = output.reshape(output.shape[0], -1)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        # Cross-entropy loss
        batch_size = y_true.shape[0]
        loss = -np.sum(np.log(probs[np.arange(batch_size), y_true])) / batch_size

        # Backprop gradients
        dL_dlogits = probs
        dL_dlogits[np.arange(batch_size), y_true] -= 1
        dL_dlogits /= batch_size
        dL_dconv1_out = dL_dlogits.reshape(self.conv1_out.shape)
        dL_dconv1_out *= self.relu_derivative(self.conv1_out)

        # Gradients for conv1 filters and biases
        dL_dconv1_filters = np.zeros_like(self.conv1_filters)
        dL_dconv1_biases = np.sum(dL_dconv1_out, axis=(0, 1, 2))

        batch_size, height, width, _ = x.shape
        _, filter_height, filter_width, _ = self.conv1_filters.shape
        for b in range(batch_size):
            for h in range(height - filter_height + 1):
                for w in range(width - filter_width + 1):
                    region = x[b, h:h+filter_height, w:w+filter_width, :]
                    for f in range(self.conv1_filters.shape[0]):
                        dL_dconv1_filters[f] += region * dL_dconv1_out[b, h, w, f]

        return loss, dL_dconv1_filters, dL_dconv1_biases

    def update_weights(self, dL_dconv1_filters, dL_dconv1_biases, learning_rate=0.01):
        self.conv1_filters -= learning_rate * dL_dconv1_filters
        self.conv1_biases -= learning_rate * dL_dconv1_biases


# Test the simplified model
if __name__ == "__main__":
    # Define inputs and labels
    input_shape = (1, 28, 28, 1)  # Batch size 1, height 28, width 28, channels 1
    num_classes = 10
    model = SimplifiedCNN(input_shape, num_classes)

    # Dummy input and label
    x = np.random.randn(*input_shape)
    y_true = np.array([1])  # Example: true label is class 1

    # Train for a few iterations
    for epoch in range(10):
        loss, dL_dconv1_filters, dL_dconv1_biases = model.compute_loss_and_gradients(x, y_true)
        model.update_weights(dL_dconv1_filters, dL_dconv1_biases)
        print(f"Epoch {epoch + 1}, Loss: {loss}")