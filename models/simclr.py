import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        embedding = self.backbone(x)
        projection = self.projection(embedding)
        return embedding, projection
