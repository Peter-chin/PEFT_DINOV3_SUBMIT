import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv3LinearProbe(nn.Module):
    def __init__(self, num_classes, REPO_DIR, weight_path, full_finetune=False):
        super().__init__()
        # =========================
        print(f"Loading DINOv3 backbone from {weight_path} with full_finetune={full_finetune}")
        self.backbone = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=weight_path)

        # unfreeze backbone
        print(f"Setting requires_grad={full_finetune} for backbone parameter")
        for param in self.backbone.parameters():
            param.requires_grad = full_finetune
        
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)
        
        self.full_finetune = full_finetune

    def forward(self, x):
        if self.full_finetune:
            features = self.backbone(x)   # [B, 768]
        else:                                           
            with torch.no_grad():
                features = self.backbone(x)   # [B, 768]

        out = self.classifier(features)
        return out

class DINOv3PromptProbe(nn.Module):
    def __init__(self, num_classes, REPO_DIR, weight_path, n_prompt_tokens=4):
        super().__init__()
        # PROMPT-TOKEN EXTENSION: for linear probe, use DINOv3LinearProbe instead of this class.
        print(f"Loading DINOv3 backbone from {weight_path} with n_prompt_tokens={n_prompt_tokens}")
        self.backbone = torch.hub.load(
            REPO_DIR,
            'dinov3_vits16',
            source='local',
            weights=weight_path,
            n_prompt_tokens=n_prompt_tokens,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False
        # PROMPT-TOKEN EXTENSION: linear probe should keep prompt_tokens frozen by removing this line.
        if hasattr(self.backbone, "prompt_tokens"):
            self.backbone.prompt_tokens.requires_grad = True

        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        
        return out

class DINOv3Seg(nn.Module):
    """DINOv3 backbone + lightweight convolutional decoder for segmentation."""

    def __init__(
        self,
        REPO_DIR,
        weight_path,
        num_classes=1,
        n_layers=4,
        decoder_dim=256,
        full_finetune=True,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            REPO_DIR, "dinov3_vits16", source="local", weights=weight_path
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = full_finetune
        
        self.patch_size = getattr(self.backbone, "patch_size", 16)
        self.embed_dim = getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features", 384))
        self.n_layers = n_layers
        self.full_finetune = full_finetune

        in_dim = self.embed_dim * n_layers
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )

        decoder_total = sum(p.numel() for p in self.decoder.parameters())
        decoder_trainable = sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        )

        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")

        print(f"Backbone total params: {backbone_total:,}")
        print(f"Backbone trainable params: {backbone_trainable:,}")

        print(f"Decoder total params: {decoder_total:,}")
        print(f"Decoder trainable params: {decoder_trainable:,}")

    def forward(self, x):
        H, W = x.shape[-2:]

        if self.full_finetune:
            feats = self.backbone.get_intermediate_layers(
                x, n=self.n_layers, reshape=True, norm=True
            )
        else:
            with torch.no_grad():
                feats = self.backbone.get_intermediate_layers(
                    x, n=self.n_layers, reshape=True, norm=True
                )

        feats = torch.cat(feats, dim=1)   # [B, n_layers*C, h, w]

        logits = self.decoder(feats)
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits

if __name__ == "__main__":
    # Example usage
    model = DINOv3Seg(
        REPO_DIR="./models/dinov3",
        weight_path="./models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        num_classes=1,
        n_layers=4,
        decoder_dim=256,
        full_finetune=True,
    )
    print(model)
