import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim, adapter_dim=64, dropout=0.1, scale=1.0):
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.scale = scale

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        correction = self.down(x)
        correction = self.activation(correction)
        correction = self.dropout(correction)
        correction = self.up(correction)
        return x + self.scale * correction


class BlockWithAdapter(nn.Module):
    def __init__(self, block, hidden_dim, adapter_dim=64, dropout=0.1, scale=1.0):
        super().__init__()
        self.block = block
        self.adapter = BottleneckAdapter(hidden_dim, adapter_dim, dropout, scale)

    def forward(self, x, *args, **kwargs):
        out = self.block(x, *args, **kwargs)

        if isinstance(out, tuple):
            hidden = self.adapter(out[0])
            return (hidden, *out[1:])

        if isinstance(out, list):
            out[0] = self.adapter(out[0])
            return out

        return self.adapter(out)


class DINOv3SegAdapters(nn.Module):
    def __init__(
        self,
        REPO_DIR,
        weight_path,
        num_classes=1,
        n_layers=4,
        decoder_dim=256,
        adapter_dim=64,
        adapter_dropout=0.1,
        adapter_scale=1.0,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            REPO_DIR, "dinov3_vits16", source="local", weights=weight_path
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden_dim = getattr(
            self.backbone,
            "num_features",
            getattr(self.backbone, "embed_dim", 384),
        )

        if not hasattr(self.backbone, "blocks"):
            raise AttributeError("Could not find self.backbone.blocks in DINOv3 model.")

        for i, block in enumerate(self.backbone.blocks):
            self.backbone.blocks[i] = BlockWithAdapter(
                block=block,
                hidden_dim=hidden_dim,
                adapter_dim=adapter_dim,
                dropout=adapter_dropout,
                scale=adapter_scale,
            )

        self.vit = self.backbone
        self.patch_size = getattr(self.vit, "patch_size", 16)
        self.embed_dim = hidden_dim
        self.n_layers = n_layers

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

        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100 * trainable / total if total > 0 else 0.0

        print("\n--- Adapter Segmentation Trainable Parameters ---")
        print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")
        print("------------------------------------------------\n")

    def forward(self, x):
        H, W = x.shape[-2:]

        feats = self.vit.get_intermediate_layers(
            x, n=self.n_layers, reshape=True, norm=True
        )
        feats = torch.cat(feats, dim=1)

        logits = self.decoder(feats)
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits
