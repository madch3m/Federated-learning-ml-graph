import torch.nn.functional as F
from torch import nn
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
  

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
      super(CIFAR10CNN, self).__init__()

      self.conv1 = nn.Conv2d(3,64, kernel_size=3, padding=1)
      self.batchNorm1 = nn.BatchNorm2d(64)
      self.conv1_2 = nn.Conv2d(64,64, kernel_size=3,padding=1)
      self.batchNorm2 = nn.BatchNorm2d(64)


      self.conv2 = nn.Conv2d(64,128, kernel_size=3, padding=1)
      self.batchNorm2_1 = nn.BatchNorm2d(128)
      self.conv2_2 = nn.Conv2d(128,128, kernel_size=3, padding=1)
      self.batchNorm2_2 = nn.BatchNorm2d(128)

      self.conv3 = nn.Conv2d(128,256, kernel_size=3, padding=1)
      self.batchNorm3_1 = nn.BatchNorm2d(256)
      self.conv3_1 = nn.Conv2d(256,256, kernel_size=3, padding=1)
      self.batchNorm3_3 = nn.BatchNorm2d(256)


      self.pool = nn.MaxPool2d(2,2)
      self.dropout = nn.Dropout(0.5)

      self.fc1 = nn.Linear(256 * 4 * 4, 512)
      self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
          x = F.relu(self.batchNorm1(self.conv1(x)))
          x = F.relu(self.batchNorm2(self.conv1_2(x)))
          x = self.pool(x)

          x = F.relu(self.batchNorm2_1(self.conv2(x)))
          x = F.relu(self.batchNorm2_2(self.conv2_2(x)))
          x = self.pool(x)

          x = F.relu(self.batchNorm3_1(self.conv3(x)))
          x = F.relu(self.batchNorm3_3(self.conv3_1(x)))

          x = self.pool(x)

          x = x.view(x.size(0), -1)
          x = F.relu(self.fc1(x))
          x = self.dropout(x)
          x = self.fc2(x)

          return x


