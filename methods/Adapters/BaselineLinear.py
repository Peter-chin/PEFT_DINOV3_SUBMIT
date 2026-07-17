import torch
import torch.nn as nn

class DINOv3LinearProbe(nn.Module):
    def __init__(self, num_classes, REPO_DIR, weight_path):
        super().__init__()
        # =========================
        self.backbone = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=weight_path)

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        # =========================
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)   # [B, 768]

        out = self.classifier(features)
        return out
