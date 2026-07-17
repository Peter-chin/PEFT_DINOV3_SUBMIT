import torch
import torch.nn as nn
from typing import Optional
# segmantation
import torch.nn.functional as F

class DINOv3LinearProbe(nn.Module):
    def __init__(self, num_classes, REPO_DIR, weight_path):
        super().__init__()
        self.backbone = torch.hub.load(
            REPO_DIR, 
            'dinov3_vits16', 
            source='local', 
            weights=weight_path,
            )

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)   # [B, 768]

        out = self.classifier(features)
        return out


class DINOv3PromptProbe(nn.Module):
    def __init__(self, num_classes, REPO_DIR, weight_path, n_prompt_tokens=4):
        super().__init__()
        # PROMPT-TOKEN EXTENSION: for linear probe, use DINOv3LinearProbe instead of this class.
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
        return self.classifier(features)

# class DeepPromptViTWrapper(nn.Module):
#     """
#     Deep Prompt Tuning wrapper for a ViT-style backbone (e.g. DINOv3 ViT).
#     - Does NOT modify backbone class or its parameters.
#     - Inserts learnable prompt tokens before each transformer block (VPT-Deep style).
#     - Typically you will freeze the backbone and train only prompts + downstream head.
#     """

#     # def __init__(
#     #     self,
#     #     backbone: nn.Module,
#     #     num_prompt_tokens: int = 5,
#     #     prompt_dim: Optional[int] = None,
#     #     freeze_backbone: bool = True,
#     #     prompt_dropout: float = 0.0,
#     #     insert_position: str = "after_cls",
#     # ):
#     def __init__(
#         self,
#         REPO_DIR,
#         weight_path,
#         num_prompt_tokens=5,
#         prompt_dim=None,
#         freeze_backbone=True,
#         prompt_dropout=0.0,
#         insert_position="after_cls",
#     ):
#         """
#         Args:
#             backbone: a ViT model instance (e.g. DINOv3 ViT-B/16 loaded via torch.hub).
#             num_prompt_tokens: number of prompt tokens per block.
#             prompt_dim: embedding dimension of prompts; if None, inferred from backbone.
#             freeze_backbone: whether to set backbone parameters requires_grad = False.
#             prompt_dropout: dropout applied on prompt tokens.
#             insert_position: where to insert prompts in the token sequence:
#                 - "after_cls": [CLS] + prompts + patch tokens
#                 - "before_cls": prompts + [CLS] + patch tokens
#                 - "end": [CLS] + patch tokens + prompts
#         """
#         super().__init__()
#         self.backbone = torch.hub.load(
#             REPO_DIR, 
#             'dinov3_vits16', 
#             source='local', 
#             weights=weight_path,
#             )
#         self.num_prompt_tokens = num_prompt_tokens
#         self.insert_position = insert_position

#         # Try to infer embedding dimension from backbone.
#         # Common patterns: backbone.embed_dim or backbone.pos_embed.shape[-1]
#         if prompt_dim is not None:
#             embed_dim = prompt_dim
#         elif hasattr(self.backbone, "embed_dim"):
#             embed_dim = self.backbone.embed_dim
#         elif hasattr(self.backbone, "pos_embed"):
#             embed_dim = self.backbone.pos_embed.shape[-1]
#         else:
#             raise ValueError(
#                 "Cannot infer embed_dim from backbone. "
#                 "Please pass prompt_dim explicitly."
#             )
#         self.embed_dim = embed_dim

#         # Create per-block prompt parameters: one prompt set per transformer block.
#         if not hasattr(self.backbone, "blocks"):
#             raise ValueError("Backbone must have attribute 'blocks' (transformer layers).")
#         num_blocks = len(self.backbone.blocks)

#         # Shape: (num_blocks, num_prompt_tokens, embed_dim), stored as Parameter list.
#         self.deep_prompts = nn.ParameterList(
#             [
#                 nn.Parameter(
#                     torch.zeros(1, num_prompt_tokens, embed_dim)
#                 )  # 1 x P x D, later expand to batch
#                 for _ in range(num_blocks)
#             ]
#         )

#         # Simple initialization: normal with small std; you can change to kaiming/xavier if you like.
#         for p in self.deep_prompts:
#             nn.init.normal_(p, mean=0.0, std=0.02)

#         self.prompt_dropout = nn.Dropout(prompt_dropout)

#         # Optionally freeze backbone parameters
#         if self.backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward that mimics standard ViT forward, but injects deep prompts
#         before each transformer block.

#         Returns:
#             Usually the CLS token feature after the final norm.
#             You can adapt this if you need all tokens.
#         """
#         B = x.shape[0]

#         # ---- Standard ViT input embedding ----
#         # Most ViT implementations (including timm-style used by DINOv3) follow:
#         #   x = patch_embed(img)
#         #   x = concat(cls_token, x)
#         #   x = x + pos_embed
#         #   x = pos_drop(x)
#         #
#         # We do NOT call backbone.forward() because we need access to each block.
#         # Instead, we manually reproduce forward_features-like behavior here.

#         # 1. Patch embedding
#         if hasattr(self.backbone, "patch_embed"):
#             x = self.backbone.patch_embed(x)
#         else:
#             raise ValueError("Backbone must have patch_embed module for ViT-style input.")

#         # 2. Add CLS token
#         if hasattr(self.backbone, "cls_token"):
#             cls_token = self.backbone.cls_token.expand(B, -1, -1)  # (B, 1, D)
#             x = torch.cat((cls_token, x), dim=1)  # (B, 1 + N, D)
#         else:
#             # If backbone has no CLS token, you could treat first token as CLS or add your own.
#             raise ValueError("Backbone must have cls_token for this wrapper implementation.")

#         # 3. Add positional embedding (assume same length as sequence without prompts)
#         if hasattr(self.backbone, "pos_embed"):
#             # pos_embed is typically (1, 1 + N, D)
#             pos_embed = self.backbone.pos_embed
#             if pos_embed.shape[1] != x.shape[1]:
#                 # If shape mismatches (e.g. different image size), interpolate or slice.
#                 # For simplicity, we slice here; you can implement interpolation if needed.
#                 pos_embed = pos_embed[:, : x.shape[1], :]
#             x = x + pos_embed
#         # 4. Dropout on embeddings if available
#         if hasattr(self.backbone, "pos_drop"):
#             x = self.backbone.pos_drop(x)

#         # ---- Transformer blocks with deep prompts ----
#         for idx, block in enumerate(self.backbone.blocks):
#             # Current tokens x: (B, L, D)
#             # Prepare prompts for this block: (B, P, D)
#             prompt = self.deep_prompts[idx]  # (1, P, D)
#             prompt = prompt.expand(B, -1, -1)  # (B, P, D)
#             prompt = self.prompt_dropout(prompt)

#             if self.insert_position == "after_cls":
#                 # [CLS] + prompts + rest
#                 cls = x[:, :1, :]          # (B, 1, D)
#                 rest = x[:, 1:, :]         # (B, L-1, D)
#                 x = torch.cat((cls, prompt, rest), dim=1)
#             elif self.insert_position == "before_cls":
#                 rest = x[:, 1:, :]
#                 cls = x[:, :1, :]
#                 x = torch.cat((prompt, cls, rest), dim=1)
#             elif self.insert_position == "end":
#                 x = torch.cat((x, prompt), dim=1)
#             else:
#                 raise ValueError(f"Unsupported insert_position: {self.insert_position}")

#             # Pass through transformer block
#             x = block(x)

#         # ---- Final norm & CLS pooling ----
#         if hasattr(self.backbone, "norm"):
#             x = self.backbone.norm(x)

#         # By default return CLS token feature
#         cls_out = x[:, 0]  # (B, D)
#         return cls_out
#         # If you need all tokens (e.g. segmentation), simply return x instead.



class DeepPromptViTWrapper(nn.Module):
    def __init__(
        self,
        REPO_DIR,
        weight_path,
        num_prompt_tokens=5,
        prompt_dim=None,
        freeze_backbone=True,
        prompt_dropout=0.0,
        insert_position="after_cls",
    ):
        super().__init__()
        self.backbone = torch.hub.load(
            REPO_DIR,
            'dinov3_vits16',
            source='local',
            weights=weight_path,
        )
        self.num_prompt_tokens = num_prompt_tokens
        self.insert_position = insert_position

        if prompt_dim is not None:
            embed_dim = prompt_dim
        elif hasattr(self.backbone, "embed_dim"):
            embed_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, "pos_embed"):
            embed_dim = self.backbone.pos_embed.shape[-1]
        else:
            raise ValueError(
                "Cannot infer embed_dim from backbone. "
                "Please pass prompt_dim explicitly."
            )
        self.embed_dim = embed_dim

        if not hasattr(self.backbone, "blocks"):
            raise ValueError("Backbone must have attribute 'blocks' (transformer layers).")
        num_blocks = len(self.backbone.blocks)

        self.deep_prompts = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, num_prompt_tokens, embed_dim))
                for _ in range(num_blocks)
            ]
        )

        for p in self.deep_prompts:
            nn.init.normal_(p, mean=0.0, std=0.02)

        self.prompt_dropout = nn.Dropout(prompt_dropout)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = self.backbone.patch_embed(x)

        if x.ndim == 4:
            if x.shape[1] == self.embed_dim:
                x = x.flatten(2).transpose(1, 2)
            elif x.shape[-1] == self.embed_dim:
                x = x.flatten(1, 2)
            else:
                raise ValueError(f"Unexpected patch_embed output shape: {x.shape}")

        elif x.ndim == 3:
            if x.shape[-1] != self.embed_dim:
                raise ValueError(f"Token dim mismatch: expected {self.embed_dim}, got {x.shape[-1]}")
        else:
            raise ValueError(f"Unexpected patch_embed output ndim: {x.ndim}")

        if hasattr(self.backbone, "cls_token"):
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        else:
            raise ValueError("Backbone must have cls_token for this wrapper implementation.")

        if hasattr(self.backbone, "pos_embed"):
            pos_embed = self.backbone.pos_embed
            if pos_embed.shape[1] != x.shape[1]:
                pos_embed = pos_embed[:, : x.shape[1], :]
            x = x + pos_embed

        if hasattr(self.backbone, "pos_drop"):
            x = self.backbone.pos_drop(x)

        for idx, block in enumerate(self.backbone.blocks):
            prompt = self.deep_prompts[idx].expand(B, -1, -1)
            prompt = self.prompt_dropout(prompt)

            if self.insert_position == "after_cls":
                cls = x[:, :1, :]
                rest = x[:, 1:, :]
                x_with_prompt = torch.cat((cls, prompt, rest), dim=1)
                x_with_prompt = block(x_with_prompt)
                cls = x_with_prompt[:, :1, :]
                rest = x_with_prompt[:, 1 + self.num_prompt_tokens :, :]
                x = torch.cat((cls, rest), dim=1)

            elif self.insert_position == "before_cls":
                rest = x[:, 1:, :]
                cls = x[:, :1, :]
                x_with_prompt = torch.cat((prompt, cls, rest), dim=1)
                x_with_prompt = block(x_with_prompt)
                cls = x_with_prompt[:, self.num_prompt_tokens:self.num_prompt_tokens + 1, :]
                rest = x_with_prompt[:, self.num_prompt_tokens + 1 :, :]
                x = torch.cat((cls, rest), dim=1)

            elif self.insert_position == "end":
                x_with_prompt = torch.cat((x, prompt), dim=1)
                x_with_prompt = block(x_with_prompt)
                x = x_with_prompt[:, :-self.num_prompt_tokens, :]

            else:
                raise ValueError(f"Unsupported insert_position: {self.insert_position}")

        if hasattr(self.backbone, "norm"):
            x = self.backbone.norm(x)

        return x[:, 0]

class DeepPromptClassificationHead(nn.Module):
    def __init__(
        self,
        num_classes,
        REPO_DIR,
        weight_path,
        n_prompt_tokens=5,
        prompt_dim=None,
        freeze_backbone=True,
        prompt_dropout=0.0,
        insert_position="after_cls",
    ):
        super().__init__()
        self.prompt_vit = DeepPromptViTWrapper(
            REPO_DIR=REPO_DIR,
            weight_path=weight_path,
            num_prompt_tokens=n_prompt_tokens,
            prompt_dim=prompt_dim,
            freeze_backbone=freeze_backbone,
            prompt_dropout=prompt_dropout,
            insert_position=insert_position,
        )
        self.fc = nn.Linear(self.prompt_vit.embed_dim, num_classes)

    def forward(self, x):
        features = self.prompt_vit(x)
        return self.fc(features)

# class DeepPromptClassificationHead(nn.Module):
#     """
#     Example classification head that uses DeepPromptViTWrapper outputs.
#     """

#     # def __init__(self, prompt_vit: DeepPromptViTWrapper, num_classes: int):
#     def __init__(
#         self,
#         num_classes,
#         REPO_DIR,
#         weight_path,
#         n_prompt_tokens=5,
#         prompt_dim=None,
#         freeze_backbone=True,
#         prompt_dropout=0.0,
#         insert_position="after_cls",
#     ):
#         super().__init__()
#         self.prompt_vit = DeepPromptViTWrapper(
#             REPO_DIR=REPO_DIR,
#             weight_path=weight_path,
#             num_prompt_tokens=n_prompt_tokens,
#             prompt_dim=prompt_dim,
#             freeze_backbone=freeze_backbone,
#             prompt_dropout=prompt_dropout,
#             insert_position=insert_position,
#         )
        
#         # self.prompt_vit = prompt_vit
#         # self.num_classes = num_classes
#         self.fc = nn.Linear(prompt_vit.embed_dim, num_classes)

#     def forward(self, x):
#         features = self.prompt_vit(x)
#         return self.fc(features)

#     # def forward(self, x: torch.Tensor) -> torch.Tensor:
#     #     features = self.prompt_vit(x)  # (B, D)
#     #     logits = self.fc(features)     # (B, C)
#     #     return logits


class DINOv3LinearProbe_Seg(nn.Module):
    def __init__(self, REPO_DIR, weight_path, num_classes=1, n_layers=4, decoder_dim=256):
        super().__init__()

        self.backbone = torch.hub.load(
            REPO_DIR, "dinov3_vits16", source="local", weights=weight_path
        )
        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Underlying (unwrapped) ViT, used to access geometry + intermediate layers.
        self.vit = self.backbone
        self.patch_size = getattr(self.vit, "patch_size", 16)
        self.embed_dim = getattr(self.vit, "embed_dim", 384)
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

    def forward(self, x):
        H, W = x.shape[-2:]

        # Last `n_layers` blocks, reshaped to (B, C, H/patch, W/patch).
        feats = self.vit.get_intermediate_layers(
            x, n=self.n_layers, reshape=True, norm=True
        )
        feats = torch.cat(feats, dim=1)  # (B, n_layers*C, h, w)

        logits = self.decoder(feats)
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits


class DINOv3PromptProbe_Seg(nn.Module):
    def __init__(
        self,
        REPO_DIR,
        weight_path,
        num_classes=1,
        n_prompt_tokens=4,
        n_layers=4,
        decoder_dim=256,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            REPO_DIR,
            "dinov3_vits16",
            source="local",
            weights=weight_path,
            n_prompt_tokens=n_prompt_tokens,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, "prompt_tokens"):
            self.backbone.prompt_tokens.requires_grad = True

        self.vit = self.backbone
        self.patch_size = getattr(self.vit, "patch_size", 16)
        self.embed_dim = getattr(self.vit, "embed_dim", 384)
        self.n_layers = n_layers

        in_dim = self.embed_dim * self.n_layers
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

    def forward(self, x):
        H, W = x.shape[-2:]

        feats = self.vit.get_intermediate_layers(
            x,
            n=self.n_layers,
            reshape=True,
            norm=True,
        )
        feats = torch.cat(feats, dim=1)

        logits = self.decoder(feats)
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        return logits
