# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring
# pylint: disable=import-error
# pylint: disable=no-name-in-module
from tensorflow.keras.datasets import cifar10  # type: ignore
import numpy as np
from CNN import CNN
import cProfile
import re

# Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}")

# Normalizing the pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flattening labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# One-hot encoding for the labels
num_classes = 10
# y_train_one_hot = np.eye(num_classes)[y_train]
# y_test_one_hot = np.eye(num_classes)[y_test]

# print("Sample normalized pixel value:", x_train[0, 0, 0, 0])
# print("Sample one-hot encoded label:", y_train_one_hot[0])

input_shape = (32, 32, 3)
num_classes = 10
lenet5 = CNN(input_shape, num_classes)

batch_size = 8
learning_rate = 0.1
learning_rate_conv = 0.1


def create_batches(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size], labels[i : i + batch_size]


def train_on_batches(sample_ix, num_batches):
    train_batches = create_batches(
        x_train[sample_ix : sample_ix + (num_batches * batch_size), ...],
        y_train[sample_ix : sample_ix + (num_batches * batch_size), ...],
        batch_size=batch_size,
    )
    total_loss = 0
    for x_batch, y_batch in train_batches:
        lenet5.forward(x_batch)
        loss = lenet5.backprop(
            x_batch,
            y_batch,
            learning_rate=learning_rate,
            learning_rate_conv=learning_rate_conv,
        )
        total_loss += loss
        print(f"Batch loss = {loss}")
    print(f"Cycle loss = {total_loss / num_batches}")


def test_on_batches(sample_ix, num_batches):
    test_batches = create_batches(
        x_test[sample_ix : sample_ix + (num_batches * batch_size), ...],
        y_test[sample_ix : sample_ix + (num_batches * batch_size), ...],
        batch_size=batch_size,
    )
    correct_predictions = 0
    total_samples = 0
    for x_batch, y_batch in test_batches:
        outputs = lenet5.forward(x_batch)
        predictions = np.argmax(outputs, axis=1)
        # labels = np.argmax(y_batch, axis=1)
        correct_predictions += np.sum(predictions == y_batch)
        total_samples += len(y_batch)

    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")
    print("==============================")


def train():
    train_batch_ix = 0
    test_batch_ix = 0
    num_train_batches = 8
    num_test_batches = 2
    # max_batches = 1
    while (
        train_batch_ix * batch_size
        < x_train.shape[0]
        # and train_batch_ix < max_batches
    ):
        print(f"train_batch_ix: {train_batch_ix}")
        train_on_batches(train_batch_ix * batch_size, num_train_batches)
        train_batch_ix += num_train_batches
        test_on_batches(test_batch_ix * batch_size, num_test_batches)
        test_batch_ix += num_test_batches


train()
# cProfile.run("train()", sort="tottime")
