import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from CNN import CNN
import ADAM_optimization


# Transforming the size of images
def  transform_imagesize(file_name, batch_size):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    # Loading datasets
    if file_name.lower() == 'cifar10':
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif file_name.lower() == 'mnist':
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Datatset is not Supported")

    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return loader_train, loader_test

def train_model(model, loader_train, loader_test, device):
#     Calling hte ADAM optimizer class and initializing parameters
    ADAM_parameters = [model.conv1_filters, model.conv1_biases, model.conv2_filters, model.conv2_biases, model.fc1_weights, model.fc1_biases, model.fc2_weights, model.fc2_biases, model.output_weights, model.output_biases]
    ADAM_optimizer = ADAM_optimization.ADAM_Optimizer(ADAM_parameters)

    for epoch in range(10):
        for images, labels in loader_train:
            images = images.numpy()
            labels = labels.numpy()
            ADAM_optimizer.zero_grad()
            outputs = model.forward(images)
            loss = model.cross_entropy_loss(outputs, labels)
            ADAM_optimizer.step()
        print(f'Epoch {epoch+1}, Loss:{loss}')

    correct = 0
    total =0
    for images, labels in loader_test:
        images=images.numpy()
        labels = labels.numpy()
        outputs = model.forward(images)
        predicted = np.argmax(outputs.data, 1)
        correct += (predicted == labels).sum()
        total += labels.shape[0]

    print(f"Accuracy on test images: {100 * correct/ total}%")

# Running code
def main():
    batch_size = 64
    cnn = CNN(input_shape=(32,32,3), num_classes=10)

    cifar_train_loader, cifar_test_loader = transform_imagesize('cifar10', batch_size)
    mnist_train_loader, mnist_test_loader = transform_imagesize('mnist', batch_size)

    train_model(cnn, cifar_train_loader, cifar_test_loader)
    train_model(cnn, mnist_train_loader, mnist_test_loader)

if __name__ == "__main__":
    main()

