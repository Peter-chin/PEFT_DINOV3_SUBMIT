import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim=384, adapter_dim=64, dropout=0.1, scale=1.0):
        super().__init__()
        self.scale = scale
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(adapter_dim, hidden_dim)

        # Start close to the frozen backbone.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.dropout(self.act(self.down(x)))) * self.scale


class BlockWithAdapter(nn.Module):
    def __init__(self, block, hidden_dim=384, adapter_dim=64, dropout=0.1, scale=1.0):
        super().__init__()
        self.block = block
        self.adapter = BottleneckAdapter(
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
            dropout=dropout,
            scale=scale,
        )

    def forward(self, x, *args, **kwargs):
        out = self.block(x, *args, **kwargs)

        # Most DINO blocks return a tensor.
        if torch.is_tensor(out):
            return out + self.adapter(out)

        # Be defensive in case a block returns tuple/list.
        if isinstance(out, tuple):
            first = out[0]
            if torch.is_tensor(first):
                first = first + self.adapter(first)
            return (first, *out[1:])

        if isinstance(out, list):
            first = out[0]
            if torch.is_tensor(first):
                first = first + self.adapter(first)
            return [first, *out[1:]]

        return out


class DINOv3MultiClassAdapters(nn.Module):
    def __init__(
        self,
        num_classes,
        repo_dir,
        weight_path,
        adapter_dim=64,
        adapter_dropout=0.1,
        adapter_scale=1.0,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            repo_dir,
            "dinov3_vits16",
            source="local",
            weights=weight_path,
        )

        hidden_dim = getattr(self.backbone, "embed_dim", 384)

        # Freeze original DINOv3 backbone.
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Add adapters into every transformer block.
        self.backbone.blocks = nn.ModuleList(
            [
                BlockWithAdapter(
                    block,
                    hidden_dim=hidden_dim,
                    adapter_dim=adapter_dim,
                    dropout=adapter_dropout,
                    scale=adapter_scale,
                )
                for block in self.backbone.blocks
            ]
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)

        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats:
                feats = feats["x_norm_clstoken"]
            elif "cls_token" in feats:
                feats = feats["cls_token"]
            else:
                feats = next(v for v in feats.values() if torch.is_tensor(v))

        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        return self.classifier(feats)
