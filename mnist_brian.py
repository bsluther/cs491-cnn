from CNN import CNN
from MNISTDataLoader import MNISTDataLoader
import numpy as np
import cProfile
import re


mnist_loader = MNISTDataLoader()
# x_train: (60,000 x 28 x 28 x 1)
# y_train: (60,000 x 10)
x_train, y_train = mnist_loader.get_train_data()
# x_test: (10,000 x 28 x 28 x 1)
# y_test: (10,000 x 10)
x_test, y_test = mnist_loader.get_test_data()

# Pad MNIST to 32 x 32
padding = 2
x_train = np.pad(
    x_train,
    ((0, 0), (padding, padding), (padding, padding), (0, 0)),
    mode="constant",
    constant_values=0,
)
x_test = np.pad(
    x_test,
    ((0, 0), (padding, padding), (padding, padding), (0, 0)),
    mode="constant",
    constant_values=0,
)

# MNIST is 28 x 28 x 1 but will be padded to 32 x 32 x 1
input_shape = (32, 32, 1)
num_classes = 10
lenet5 = CNN(input_shape, num_classes)


def create_batches(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size], labels[i : i + batch_size]


batch_size = 16
epochs = 1
learning_rate = 0.01
learning_rate_conv = 0.1
train_batches_per_cycle = 16
test_batches_per_cycle = 4


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
            x_batch, y_batch, learning_rate=0.01, learning_rate_conv=0.01
        )
        total_loss += loss
        print(f"Batch loss = {loss}")
    print(f"Cycle loss = {total_loss / num_batches}")


def test_on_batches(sample_ix, num_batches):
    test_batches = create_batches(
        x_train[sample_ix : sample_ix + (num_batches * batch_size), ...],
        y_train[sample_ix : sample_ix + (num_batches * batch_size), ...],
        batch_size=batch_size,
    )
    correct_predictions = 0
    total_samples = 0
    for x_batch, y_batch in test_batches:
        outputs = lenet5.forward(x_batch)
        predictions = np.argmax(outputs, axis=1)
        labels = np.argmax(y_batch, axis=1)
        correct_predictions += np.sum(predictions == labels)
        total_samples += len(labels)

    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")
    print("==============================")


def train():
    train_batch_ix = 0
    test_batch_ix = 0
    num_train_batches = 16
    num_test_batches = 2
    max_batches = 16
    while (
        train_batch_ix * batch_size < x_train.shape[0] and train_batch_ix < max_batches
    ):
        print(f"train_batch_ix: {train_batch_ix}")
        train_on_batches(train_batch_ix * batch_size, num_train_batches)
        train_batch_ix += num_train_batches
        test_on_batches(test_batch_ix * batch_size, num_test_batches)
        test_batch_ix += num_test_batches


train()
# cProfile.run("train()", sort="tottime")
