import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError as exc:
    raise ImportError(
        "Could not import transformers. "
        "Please install/update transformers first, ideally transformers>=4.56.0 "
        "for DINOv3 support."
    ) from exc


class BottleneckAdapter(nn.Module):
    """Small trainable bottleneck adapter.

    The adapter learns a residual correction for the hidden representation:

        output = x + scale * up(dropout(GELU(down(x))))

    For DINOv3 ViT-S/16, x usually has shape:
        [batch, tokens, hidden_dim]

    For the HF model facebook/dinov3-vits16-pretrain-lvd1689m,
    hidden_dim is usually 384.
    """

    def __init__(self, hidden_dim: int, adapter_dim: int = 64, dropout: float = 0.1, scale: float = 1.0):
        super().__init__()

        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.scale = scale

        # Start close to the original backbone behaviour.
        # At initialization, correction is zero, so output ~= x.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.down(x)
        correction = self.activation(correction)
        correction = self.dropout(correction)
        correction = self.up(correction)

        return x + self.scale * correction


class BlockWithAdapter(nn.Module):
    """Wrap one existing Hugging Face DINOv3 Transformer layer and add an adapter after it.

    We keep the original Transformer layer unchanged and frozen.

    Conceptually:

        layer(x) becomes adapter(layer(x))

    This is a simple block-level adapter placement.
    It is less invasive than modifying attention/MLP internals directly.
    """

    def __init__(
        self,
        block: nn.Module,
        hidden_dim: int,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()

        self.block = block
        self.adapter = BottleneckAdapter(
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
            dropout=dropout,
            scale=scale,
        )

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)

        # Hugging Face Transformer layers often return a tuple:
        #   (hidden_states, ...)
        # We adapt only hidden_states and keep the rest unchanged.
        if isinstance(out, tuple):
            hidden_states = out[0]
            adapted_hidden_states = self.adapter(hidden_states)
            return (adapted_hidden_states, *out[1:])

        # Some modules may return just the tensor directly.
        if torch.is_tensor(out):
            return self.adapter(out)

        # Safety guard for object-like outputs.
        # This is unlikely for individual encoder layers, but useful for debugging.
        if hasattr(out, "last_hidden_state"):
            out.last_hidden_state = self.adapter(out.last_hidden_state)
            return out

        raise TypeError(
            f"Unsupported output type from wrapped block: {type(out)}. "
            "Expected tuple, tensor, or object with last_hidden_state."
        )


def _get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    """Get nested module by dot path.

    Example:
        _get_module_by_path(model, 'encoder.layer')
    """
    current = root

    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(path)
        current = getattr(current, part)

    return current


def _find_transformer_layers(model: nn.Module) -> Tuple[nn.Module, str]:
    """Find the ModuleList containing the Transformer layers.

    Different Hugging Face vision models may expose layers under slightly
    different paths. We try the common ones.

    Returns:
        layers: nn.ModuleList
        path: string path where the layers were found
    """

    candidate_paths = [
        "encoder.layer",
        "encoder.layers",
        "vit.encoder.layer",
        "vit.encoder.layers",
        "vision_model.encoder.layer",
        "vision_model.encoder.layers",
        "dinov3.encoder.layer",
        "dinov3.encoder.layers",
    ]

    for path in candidate_paths:
        try:
            layers = _get_module_by_path(model, path)
        except AttributeError:
            continue

        if isinstance(layers, nn.ModuleList):
            return layers, path

    raise AttributeError(
        "Could not automatically find the Hugging Face DINOv3 Transformer layers.\n"
        "Please print the model structure with:\n\n"
        "    print(model)\n\n"
        "and check where the encoder layers are stored."
    )


class DINOv3AdapterProbe(nn.Module):
    """Hugging Face DINOv3 with frozen backbone, trainable adapters, and classifier.

    This version uses Hugging Face instead of torch.hub + local .pth weights.

    Trainable parts:
      - bottleneck adapters inserted after each Transformer layer
      - linear classifier head

    Frozen parts:
      - original Hugging Face DINOv3 backbone weights

    Important:
      This class keeps the same argument names as the previous implementation:
        REPO_DIR, weight_path

      But for the Hugging Face version, they are ignored.
      This keeps compatibility with the existing Baseline.py.
    """

    def __init__(
        self,
        num_classes: int,
        REPO_DIR: Optional[str] = None,
        weight_path: Optional[str] = None,
        adapter_dim: int = 64,
        adapter_dropout: float = 0.1,
        adapter_scale: float = 1.0,
        hf_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        feature_type: str = "pooler",
    ):
        super().__init__()

        self.hf_model_name = os.environ.get("DINOV3_HF_MODEL", hf_model_name)
        self.feature_type = feature_type

        # Optional:
        # If the model requires authentication, set HF_TOKEN in your sbatch script:
        #   export HF_TOKEN=your_token_here
        #
        # If the model is public/access already works, token can be None.
        hf_token = os.environ.get("HF_TOKEN", None)

        # Optional:
        # To avoid writing Hugging Face cache into home directory,
        # set HF_HOME in your sbatch script, for example:
        #   export HF_HOME=/vol/miltank/users/rekr/hf_cache
        cache_dir = os.environ.get("HF_HOME", None)

        print(f"Loading Hugging Face DINOv3 model: {self.hf_model_name}")

        self.backbone = AutoModel.from_pretrained(
            self.hf_model_name,
            token=hf_token,
            cache_dir=cache_dir,
        )

        # Freeze the original Hugging Face DINOv3 parameters.
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size

        layers, layers_path = _find_transformer_layers(self.backbone)
        print(f"Found transformer layers at: {layers_path}")
        print(f"Number of transformer layers: {len(layers)}")
        print(f"Hidden dimension: {hidden_dim}")

        # Insert one adapter after every Transformer layer.
        for i, block in enumerate(layers):
            layers[i] = BlockWithAdapter(
                block=block,
                hidden_dim=hidden_dim,
                adapter_dim=adapter_dim,
                dropout=adapter_dropout,
                scale=adapter_scale,
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Important:
        # No torch.no_grad() here.
        #
        # The original DINOv3 weights are frozen with requires_grad=False,
        # but adapters are inside the backbone forward pass and need gradients.
        outputs = self.backbone(pixel_values=x, return_dict=True)

        if self.feature_type == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output

        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            # CLS token = first token.
            features = outputs.last_hidden_state[:, 0, :]

        else:
            raise RuntimeError(
                "Could not extract features from Hugging Face DINOv3 output. "
                "Expected outputs.pooler_output or outputs.last_hidden_state."
            )

        logits = self.classifier(features)

        return logits


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def print_trainable_parameters(model: nn.Module):
    total, trainable = count_parameters(model)
    percent = 100 * trainable / total if total > 0 else 0.0

    print(f"Trainable parameters: {trainable:,} / {total:,} ({percent:.2f}%)")
