import torch
import torch.nn as nn

class Conv2Model(nn.Module):
    def __init__(self):
        super(Conv2Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=32 * 64 * 64, out_features=128)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)  # Add dropout layer after fc1
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.dropout2 = nn.Dropout(p=0.5)  # Add dropout layer after fc2

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(-1, 32 * 64 * 64)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        return x