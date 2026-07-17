import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Small trainable bottleneck adapter.

    The adapter learns a residual correction for the hidden representation:

        output = x + scale * up(dropout(GELU(down(x))))

    For a ViT-S/16 DINOv3 backbone, x usually has shape [batch, tokens, 384].
    The adapter first projects 384 -> adapter_dim, then back adapter_dim -> 384.
    """

    def __init__(self, hidden_dim, adapter_dim=64, dropout=0.1, scale=1.0):
        super().__init__()

        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.scale = scale

        # Start close to the original backbone behaviour.
        # At initialization the adapter correction is zero, so output ~= x.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        correction = self.down(x)
        correction = self.activation(correction)
        correction = self.dropout(correction)
        correction = self.up(correction)

        return x + self.scale * correction


class BlockWithAdapter(nn.Module):
    """Wrap one existing DINOv3 Transformer block and add an adapter after it.

    We keep the original block unchanged. Only the adapter learns.

    DINOv3 sometimes passes/returns a Tensor, but in forward_features_list it can
    also pass/return a list of tensors. Therefore we handle both cases.
    """

    def __init__(self, block, hidden_dim, adapter_dim=64, dropout=0.1, scale=1.0):
        super().__init__()

        self.block = block
        self.adapter = BottleneckAdapter(
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
            dropout=dropout,
            scale=scale,
        )

    def _apply_adapter(self, value):
        """Apply adapter while preserving the original container structure."""

        if torch.is_tensor(value):
            return self.adapter(value)

        if isinstance(value, list):
            return [self._apply_adapter(v) for v in value]

        if isinstance(value, tuple):
            # Usually tuple output means (hidden_states, extra_info, ...).
            # Adapt the first element and keep the rest unchanged.
            if len(value) == 0:
                return value
            return (self._apply_adapter(value[0]), *value[1:])

        raise TypeError(
            f"Unsupported output type from DINOv3 block: {type(value)}. "
            "Expected Tensor, list of Tensors, or tuple."
        )

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        return self._apply_adapter(out)
    
    
class DINOv3AdapterProbe(nn.Module):
    """DINOv3 with original backbone not trained, plus trainable adapters and classifier.

    Trainable parts:
      - bottleneck adapters inserted after each Transformer block
      - linear classifier head

    Not trained:
      - original DINOv3 backbone weights
    """

    def __init__(
        self,
        num_classes,
        REPO_DIR,
        weight_path,
        adapter_dim=64,
        adapter_dropout=0.1,
        adapter_scale=1.0,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            REPO_DIR,
            "dinov3_vits16",
            source="local",
            weights=weight_path,
        )

        # Do not train the original DINOv3 parameters.
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.num_features

        # DINOv3 ViT models usually expose Transformer blocks as self.backbone.blocks.
        if not hasattr(self.backbone, "blocks"):
            raise AttributeError(
                "Could not find self.backbone.blocks. "
                "Please inspect the DINOv3 model structure before inserting adapters."
            )

        # Insert one adapter after every Transformer block.
        for i, block in enumerate(self.backbone.blocks):
            self.backbone.blocks[i] = BlockWithAdapter(
                block=block,
                hidden_dim=hidden_dim,
                adapter_dim=adapter_dim,
                dropout=adapter_dropout,
                scale=adapter_scale,
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Important: no torch.no_grad() here.
        # Original DINOv3 weights do not train because requires_grad=False,
        # but adapters need gradients because they are inside the backbone forward pass.
        features = self.backbone(x)
        out = self.classifier(features)

        return out


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def print_trainable_parameters(model):
    total, trainable = count_parameters(model)
    percent = 100 * trainable / total if total > 0 else 0.0

    print(f"Trainable parameters: {trainable:,} / {total:,} ({percent:.2f}%)")
