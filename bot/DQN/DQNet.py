import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNet (nn.Module):
    def __init__(self, state_size, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)
        self.bn = nn.BatchNorm1d(128)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        if len(x.shape) > 1:
            x = F.relu(self.fc1(x))
            x = self.bn(x)
            x = F.relu(self.fc2(x))
            x = self.bn(x)
            x = self.out(x)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.out(x)
        return x





