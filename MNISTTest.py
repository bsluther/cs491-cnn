# Suppose CNN is defined as in your previous code
import numpy as np

from CNN import CNN
from MNISTDataLoader import MNISTDataLoader

# Instantiate the MNIST data loader
mnist_loader = MNISTDataLoader()
x_train, y_train = mnist_loader.get_train_data()
x_test, y_test = mnist_loader.get_test_data()

# Create an instance of your CNN for MNIST
# MNIST input shape: (28, 28, 1), and 10 output classes
input_shape = (28, 28, 1)
num_classes = 10
lenet5 = CNN(input_shape, num_classes)

# Now train the CNN on MNIST data, for example:
batch_size = 64
epochs = 1


def create_batches(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]


for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_batches = create_batches(x_train, y_train, batch_size)

    total_loss = 0
    batch_count = 0

    for x_batch, y_batch in train_batches:
        # Forward pass
        lenet5.forward(x_batch)
        # Backprop and update weights
        loss = lenet5.backprop(x_batch, y_batch, learning_rate=0.001)
        total_loss += loss
        batch_count += 1
        print(f"Current loss: {total_loss}")

    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

# Evaluate on test data
test_batches = create_batches(x_test, y_test, batch_size)
correct_predictions = 0
total_samples = 0
for x_batch, y_batch in test_batches:
    outputs = lenet5.forward(x_batch)
    predictions = np.argmax(outputs, axis=1)
    labels = np.argmax(y_batch, axis=1)
    correct_predictions += np.sum(predictions == labels)
    total_samples += len(labels)

accuracy = correct_predictions / total_samples
print(f"\nTest Accuracy: {accuracy:.4f}")