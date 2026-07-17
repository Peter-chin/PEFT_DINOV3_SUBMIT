import numpy as np
import torch

class Mixup():
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x1, y1):
        lam = np.random.beta(self.alpha, self.alpha)
        perm = torch.randperm(x1.size(0), device=x1.device)
        x_mix = lam * x1 + (1.0 - lam) * x1[perm]
        y_a = y1
        y_b = y1[perm]
        return x_mix, y_a, y_b, lam
