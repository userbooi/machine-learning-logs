import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset): # treat like an array

    def __init__(self):
        # data loading
        xy = np.loadtxt("data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        # print(xy)
        self.x = torch.from_numpy(xy[:, 1:]) # features
        self.y = torch.from_numpy(xy[:, [0]]) # labels
        # print(self.x, self.y)
        self.n_samples = xy.shape[0]
        self.n_features = xy.shape[1]

    def __getitem__(self, index):
        # get the item in a dataset
        return self.x[index], self.y[index]

    def __len__(self):
        # get the length of dataset
        return self.n_samples

dataset = WineDataset()

# # # PYTORCH DATALOADER
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# dataiter = iter(dataloader)
# data = next(dataiter)
# features, labels = data
# print(features, labels)

# dummy training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / dataloader.batch_size)
print(total_samples, n_iterations)
print("\n\n")

for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            print(f"epoch: {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs: {features.shape}")
        # forward
        # backward
        # update
# # # PYTORCH DATALOADER

# built-in datasets from pytorch
torchvision.datasets.MNIST()
torchvision.datasets.FashionMNIST()
torchvision.datasets.CIFAR10()
torchvision.datasets.CocoCaptions()
