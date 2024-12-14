from tensorflow.keras.datasets import cifar10
import numpy as np
from CNN import CNN

# Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}")

# Normalizing the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flattening labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# One-hot encoding for the labels
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]

print("Sample normalized pixel value:", x_train[0, 0, 0, 0])
print("Sample one-hot encoded label:", y_train_one_hot[0])

# Splitting the dataset into mini-batches
batch_size = 64


def create_batches(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size], labels[i : i + batch_size]


# Creating an instance of the CNN class
input_shape = (32, 32, 3)
num_classes = 10
lenet5 = CNN(input_shape, num_classes)

# Hyperparameters
learning_rate = 0.01
epochs = 10


# Loss function: categorical cross-entropy
def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)  # Avoid division by zero
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_batches = create_batches(
        x_train,
        y_train_one_hot,
        batch_size,
        # x_train[0:512, :], y_train_one_hot[0:512, :], batch_size
    )

    total_loss = 0
    batch_count = 0

    for x_batch, y_batch in train_batches:
        print(f"Starting batch {batch_count + 1}")
        lenet5.forward(x_batch)
        # Backward propagation
        loss = lenet5.backprop(x_batch, y_batch, learning_rate=learning_rate)
        print(f"Batch {batch_count + 1} loss: {loss}")
        batch_count += 1
        total_loss += loss

    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

# Testing the trained model
test_batches = create_batches(x_test, y_test_one_hot, batch_size)
# test_batches = create_batches(x_test[:1024, :], y_test_one_hot[:1024, :], batch_size)
correct_predictions = 0
total_samples = 0

for x_batch, y_batch in test_batches:
    output = lenet5.forward(x_batch)
    predictions = np.argmax(output, axis=1)
    labels = np.argmax(y_batch, axis=1)
    correct_predictions += np.sum(predictions == labels)
    total_samples += len(labels)

accuracy = correct_predictions / total_samples
print(f"\nTest Accuracy: {accuracy:.4f}")
