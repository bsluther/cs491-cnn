from tensorflow.keras.datasets import cifar10
import numpy as np
from CNN import CNN

# loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}")

# normalizing the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# flattenning labels (if required by your implementation)
y_train = y_train.flatten()
y_test = y_test.flatten()

# if one-hot encoding is needed:
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]

print("Sample normalized pixel value:", x_train[0, 0, 0, 0])
print("Sample one-hot encoded label:", y_train_one_hot[0])

# we are taking our batch size as 64
batch_size = 64 

# now we are splitting the dataset into mini-batches for training
def create_batches(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]

# creating an instance of the CNN class, which we defined inside CNN.py file (same directory)
input_shape = (32, 32, 3)
num_classes = 10
lenet5 = CNN(input_shape, num_classes)

# finally, we are performing a forward pass on the first batch of training data
train_batches = create_batches(x_train, y_train_one_hot, batch_size)

# getting a single batch
x_batch, y_batch = next(train_batches)

# performing forward propagation
output = lenet5.forward(x_batch)

print("Output shape:", output.shape)  # (batch_size, num_classes)
