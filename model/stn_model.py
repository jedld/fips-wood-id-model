import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(MinimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Adjusted input size for fc1 to match concatenated features
        self.fc1 = nn.Linear(32 + 64 + 128 + 256 + 512, 64)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        outputs = []

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        outputs.append(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        outputs.append(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        outputs.append(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        outputs.append(x)

        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        outputs.append(x)

        # Apply global pooling to each output
        pooled_outputs = [self.global_pool(out) for out in outputs]

        # Flatten and concatenate
        flattened_outputs = [out.view(out.size(0), -1) for out in pooled_outputs]
        x = torch.cat(flattened_outputs, dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)  # Corrected typo here
        x = self.fc3(x)
        return x

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
