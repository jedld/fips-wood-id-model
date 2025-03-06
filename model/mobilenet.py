
import torchvision.models as models
import torch.nn as nn
import torch

class MobileNetV2TransferClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MobileNetV2TransferClassifier, self).__init__()
        base_model = models.mobilenet_v2(pretrained=pretrained)
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, num_classes)
        self.model = base_model
    
    def forward(self, x):
        return self.model(x)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
