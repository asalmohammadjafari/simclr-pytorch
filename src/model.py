import torch.nn as nn
from torchvision import models


class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z
