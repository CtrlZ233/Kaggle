import torch
from torch import nn
class simpleNet(torch.nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(784, 300), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(300, 100), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(100,10), nn.Sigmoid())
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

