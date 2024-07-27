import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6, kernel_size=3)
        self.conv2 = nn.Conv2d(6,12, kernel_size=3)
        self.fc1 = nn.Linear(12*12*12, 128)
        self.fc2 = nn.Linear(128, 128)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 12*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)