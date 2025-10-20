import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import List, Tuple

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.ReLU(),nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128,10)
    )
  def forward(self, x):
      return self.net(x)