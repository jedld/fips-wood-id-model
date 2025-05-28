import torch
import torch.nn as nn
import timm

class EfficientNetTransferClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_name='efficientnet_b0'):
        super(EfficientNetTransferClassifier, self).__init__()
        # Create model with pretrained weights if specified
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location)) 