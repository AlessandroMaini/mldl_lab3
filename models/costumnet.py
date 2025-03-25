import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Add more layers...
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        # Add more layers...
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)


        return x