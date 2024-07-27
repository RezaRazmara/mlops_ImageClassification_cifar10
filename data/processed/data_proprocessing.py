import numpy as np
import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.CIFAR10(
    root=".\\data\\raw",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_dataset = torchvision.datasets.CIFAR10(
    root=".\\data\\raw",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

train_data = train_dataset.data
test_data = test_dataset.data

train_data = train_data / 255.0
test_data = test_data / 255.0

np.save(".\\data\\processed\\train_data.npy", train_data)
np.save(".\\data\\processed\\test_data.npy", test_data)
