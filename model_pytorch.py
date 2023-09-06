import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 30 * 30, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(128, 256)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(256, 525)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool1(self.batchnorm1(self.relu1(self.conv1(x))))
        x = self.maxpool2(self.batchnorm2(self.relu2(self.conv2(x))))
        x = self.maxpool3(self.batchnorm3(self.relu3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

# Create an instance of the model
model = CNNModel()

# Print the model architecture
print(model)
