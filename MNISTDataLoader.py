import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class MNISTDataLoader:
    def __init__(self):
        # Load the MNIST dataset (28x28 grayscale images, 10 classes)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # MNIST images are grayscale (one channel), shape = (28, 28)
        # Reshape to (28,28,1) for compatibility with CNN input:
        self.x_train = x_train.reshape((-1, 28, 28, 1))
        self.x_test = x_test.reshape((-1, 28, 28, 1))

        # Normalize pixel values to [0,1]
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        # Convert labels to one-hot encoding
        self.y_train = to_categorical(y_train, num_classes=10)
        self.y_test = to_categorical(y_test, num_classes=10)

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test