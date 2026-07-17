import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    """Combined Dice + binary cross-entropy loss for binary segmentation."""

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        intersection = (probs * targets_f).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets_f.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
