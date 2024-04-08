import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNet (nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(25, 512)
        self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(512, 512)
        self.out = nn.Linear(512, n_actions)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.out(x)
        return x





