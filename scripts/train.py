import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
sys.path.append(".\\")
from models.pytorch.cnn import CNN
# load data
train_data = np.load('.\\data\\processed\\train_data.npy')
test_data = np.load('.\\data\\processed\\test_data.npy')

# define the model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for x, y in zip(train_data, test_data):
        x = torch.tensor(x)
        y = torch.tensor(y)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss = {:.4f}'.format(epoch+1, loss.item()))