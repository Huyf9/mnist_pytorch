import torch.nn as nn
import torch

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        # 28 x 28
        self.fc1 = nn.Linear(784, 392)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(392, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x = torch.flatten(torch.squeeze(x, 0))
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
