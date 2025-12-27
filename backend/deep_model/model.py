import torch
import torch.nn as nn
import torchvision.models as models

class SpillNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.age_head = nn.Linear(2048, 1)        # minutes (regression)
        self.thickness_head = nn.Linear(2048, 1)  # thickness index
        self.risk_head = nn.Linear(2048, 3)       # low / medium / high

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        age = self.age_head(features)
        thickness = self.thickness_head(features)
        risk_logits = self.risk_head(features)

        return age, thickness, risk_logits
