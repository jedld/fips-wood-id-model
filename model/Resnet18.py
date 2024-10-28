import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1  # For BasicBlock, expansion is 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer in the block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer in the block
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        # Downsample layer to match dimensions for the residual connection
        self.downsample = downsample
        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # First convolutional layer with BatchNorm and ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolutional layer with BatchNorm
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling to the identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the identity (skip connection)
        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):

    def __init__(self, num_classes=1000):  # Default number of classes is 1000 for ImageNet
        super(ResNet18, self).__init__()
        # Initial convolutional layer with BatchNorm and ReLU
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Input channels = 3 (RGB images)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, blocks=2, stride=2)

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        # Downsample layer if input and output dimensions do not match
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block in the layer
        layers.append(block(in_channels, out_channels, stride, downsample))
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(out_channels * block.expansion, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # Initialize weights using Kaiming He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Max pooling
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Average pooling
        x = self.avgpool(x)
        # Flatten for the fully connected layer
        x = torch.flatten(x, 1)
        # Fully connected layer
        x = self.fc(x)

        return x

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
