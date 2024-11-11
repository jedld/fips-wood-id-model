import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# Implements a conditional SAGAN

class Generator(nn.Module):
    def __init__(self, nz=100, num_classes=10, emb_dimen=None):  # nz is the size of the latent vector (noise)
        super(Generator, self).__init__()
        
        self.nz = nz
        self.num_classes = num_classes
        if emb_dimen is None:
            emb_dimen = num_classes
            
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, emb_dimen)

        self.fc = nn.Linear(nz + emb_dimen, 512*4*4)  # Adjusted for 4x4 feature maps

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, 1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # Additional layer
            nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Additional layer
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
  
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Additional layer
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 256x256 -> 512x512
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        label_embeddings = self.label_emb(labels)
        x = torch.cat([noise, label_embeddings], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)  # reshape to 4x4 feature maps
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, 64)

        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, 1024, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            #32x32 -> 16x16
            nn.Conv2d(1024, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            #16x16 -> 8x8
            nn.Conv2d(512, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Additional layer
            nn.Conv2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64, 4, stride=2, padding=1),  # Downsample to 2x2
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final fully connected layer for h(x)
        self.fc = nn.Linear(64, 1)

    def forward(self, x, labels):
        # Pass input through main network
        features = self.main(x)  # Shape: (batch_size, 64, 2, 2)

        # Global sum pooling
        features = torch.sum(features, dim=[2, 3])  # Shape: (batch_size, 64)

        # Calculate h(x)
        out = self.fc(features)  # Shape: (batch_size, 1)

        # Get label embeddings
        label_embeddings = self.label_emb(labels)  # Shape: (batch_size, 64)

        # Projection of label embeddings onto features
        proj = torch.sum(features * label_embeddings, dim=1, keepdim=True)  # Shape: (batch_size, 1)

        # Final output
        return out + proj  # Shape: (batch_size, 1)
    
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
