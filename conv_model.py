import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 28 x 28
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 1)
        self.relu1 = nn.ReLU()

        self.Conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 1)
        self.relu2 = nn.ReLU()

        self.Conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2, 1)
        self.relu3 = nn.ReLU()

        self.drop = nn.Dropout(0.8)  # 将80%的神经元失活
        self.fc = nn.Linear(10*13*13, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.Conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = self.Conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)

        x = x.view(-1, 10*13*13)  # 将四维矩阵纬度拉成一维  [Batch, Channel, H, W]
        x = self.drop(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
