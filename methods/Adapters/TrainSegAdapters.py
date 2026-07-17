import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
try:
    import wandb
except ImportError:
    wandb = None
from SegDataset import FracAtlasCocoSegDataset, build_seg_dataset
from SegAdapters import DINOv3SegAdapters
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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


@torch.no_grad()
def seg_metrics(logits, targets, threshold=0.5, eps=1e-7):
    """Return summed intersection / union / dice components for a batch.

    Returning the raw components (rather than per-batch averages) lets the caller
    accumulate exact dataset-level Dice and IoU.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets_f = targets.view(targets.size(0), -1)

    intersection = (preds * targets_f).sum().item()
    pred_sum = preds.sum().item()
    target_sum = targets_f.sum().item()
    union = pred_sum + target_sum - intersection

    correct = (preds == targets_f).sum().item()
    total = targets_f.numel()

    return intersection, union, pred_sum, target_sum, correct, total


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    inter_sum = union_sum = pred_sum_all = target_sum_all = 0.0
    correct_sum = total_sum = 0.0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            inter, union, ps, ts, correct, total = seg_metrics(logits, y)
            inter_sum += inter
            union_sum += union
            pred_sum_all += ps
            target_sum_all += ts
            correct_sum += correct
            total_sum += total

    avg_loss = total_loss / max(len(loader), 1)
    iou = inter_sum / (union_sum + 1e-7)
    dice = (2.0 * inter_sum) / (pred_sum_all + target_sum_all + 1e-7)
    pixel_acc = correct_sum / (total_sum + 1e-7)

    return avg_loss, dice, iou, pixel_acc


def plot_curve(history, keys, ylabel, title, out_path):
    plt.figure()
    for key in keys:
        plt.plot(history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_overlays(model, dataset, indices, device, out_dir, img_size):
    """Save a few qualitative prediction overlays for visual inspection."""
    os.makedirs(out_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            logits = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.float32)

            img = image.permute(1, 2, 0).numpy() * std + mean
            img = np.clip(img, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[1].imshow(mask[0].numpy(), cmap="gray")
            axes[1].set_title("Ground truth")
            axes[2].imshow(img)
            axes[2].imshow(pred, cmap="Reds", alpha=0.5)
            axes[2].set_title("Prediction")
            for ax in axes:
                ax.axis("off")
            plt.savefig(os.path.join(out_dir, f"sample_{i}.png"), bbox_inches="tight")
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Adapter fine-tuning of DINOv3 for binary segmentation "
        "(FracAtlas fracture or BTXRD bone-tumor)"
    )
    parser.add_argument(
        "--datasets-type",
        default="FracAtlas",
        help="Which dataset to use: FracAtlas or BTXRD",
    )
    parser.add_argument(
        "--result-root",
        default="/vol/miltank/users/chebil/dinov3/results_Adapters/FracAtlas_Seg",
        help="Results root directory (can be relative)",
    )
    parser.add_argument(
        "--data-root",
        default="/vol/miltank/users/chebil/dinov3/datasets/FracAtlas",
        help="Dataset root directory (can be relative)",
    )
    parser.add_argument(
        "--coco-json",
        default="Annotations/COCO JSON/COCO_fracture_masks.json",
        help="Path to the COCO JSON file, relative to --data-root (or absolute)",
    )
    parser.add_argument(
        "--img-subdir",
        default="images",
        help="Image directory relative to --data-root (Fractured/ is searched within)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--adapter-dim", type=int, default=64, help="Adapter bottleneck dimension")
    parser.add_argument("--adapter-dropout", type=float, default=0.1, help="Adapter dropout")
    parser.add_argument("--adapter-scale", type=float, default=1.0, help="Adapter residual scale")
    parser.add_argument("--img-size", type=int, default=448, help="Input image size")
    parser.add_argument(
        "--monai",
        type=str2bool,
        default=False,
        help="Apply paired MONAI data augmentation to the training set only.",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Validation fraction"
    )
    parser.add_argument("--test-split", type=float, default=0.15, help="Test fraction")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument(
        "--data-regime",
        default="100%",
        help="Training data regime: 5%%, 10%%, 50%%, or 100%%. Only the training split is subsampled.",
    )
    parser.add_argument(
        "--fold-num",
        type=int,
        default=1,
        help="Fold number for 5-fold cross-validation, from 1 to num_folds.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--model-weight-path",
        default="./models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        help="Path to DINOv3 model weights (can be relative)",
    )
    parser.add_argument(
        "--repo-dir",
        default="./models/dinov3",
        help="Directory for torch.hub to load DINOv3 (can be relative)",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Weights & Biases project name. Omit to disable W&B logging.",
    )
    parser.add_argument(
        "--wandb-run",
        default=None,
        help="W&B run name (optional; auto-generated if not set).",
    )
    args = parser.parse_args()

    result_root = os.path.abspath(os.path.expanduser(args.result_root))
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    repo_root = os.path.abspath(os.path.expanduser(args.repo_dir))
    model_weight_path = os.path.abspath(os.path.expanduser(args.model_weight_path))

    coco_json = args.coco_json
    if not os.path.isabs(coco_json):
        coco_json = os.path.join(data_root, coco_json)
    img_dir = os.path.join(data_root, args.img_subdir)

    os.makedirs(result_root, exist_ok=True)

    print(f"Using dataset type:  {args.datasets_type}")
    print(f"Using result root:   {result_root}")
    print(f"Using data root:     {data_root}")
    if args.datasets_type.strip().lower() == "fracatlas":
        print(f"Using COCO json:     {coco_json}")
        print(f"Using image dir:     {img_dir}")
    print(f"Using repo dir:      {repo_root}")
    print(f"Using model weights: {model_weight_path}")
    print(
        f"Epochs: {args.epochs}, Batch size: {args.batch_size}, "
        f"LR: {args.learning_rate}, Image size: {args.img_size}"
    )
    print(f"Data augmentation: {'enabled with paired MONAI' if args.monai else 'disabled'}")

    # ---------------------------------------------------------- wandb setup
    use_wandb = args.wandb_project is not None
    if use_wandb and wandb is None:
        print("[WARN] wandb is not installed. Disabling W&B logging.")
        use_wandb = False

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("CUDA is not available, but this job requires a GPU.")

    device = torch.device("cuda")

    # ---------------------------------------------------------------- data
    # FracAtlas keeps the explicit (coco_json, img_dir) path so custom
    # --coco-json / --img-subdir flags are honored; BTXRD (and anything else)
    # goes through the factory, which resolves the standard layout.
    if args.datasets_type.strip().lower() == "fracatlas":
        full_dataset = FracAtlasCocoSegDataset(
            coco_json=coco_json, img_dir=img_dir, img_size=args.img_size, augment=False
        )
        train_dataset = FracAtlasCocoSegDataset(
            coco_json=coco_json, img_dir=img_dir, img_size=args.img_size, augment=args.monai
        )
    else:
        full_dataset = build_seg_dataset(
            args.datasets_type, data_root, img_size=args.img_size, augment=False
        )
        train_dataset = build_seg_dataset(
            args.datasets_type, data_root, img_size=args.img_size, augment=args.monai
        )

    if len(train_dataset) != len(full_dataset):
        raise RuntimeError(
            f"Train/eval dataset length mismatch: {len(train_dataset)} vs {len(full_dataset)}"
        )

    n = len(full_dataset)

    if args.num_folds < 2:
        raise ValueError("--num-folds must be at least 2")
    if args.fold_num < 1 or args.fold_num > args.num_folds:
        raise ValueError(f"--fold-num must be between 1 and {args.num_folds}")

    # 5-fold CV:
    # test = current fold
    # val  = next fold
    # train = remaining folds
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, args.num_folds)

    fold_idx = args.fold_num - 1
    val_fold_idx = (fold_idx + 1) % args.num_folds

    test_idx = folds[fold_idx].tolist()
    val_idx = folds[val_fold_idx].tolist()
    train_idx = np.concatenate(
        [folds[i] for i in range(args.num_folds) if i not in (fold_idx, val_fold_idx)]
    ).tolist()

    full_train_size = len(train_idx)
    regime = str(args.data_regime).strip()

    if regime.lower() in ("all", "full", "100", "100%"):
        selected_train_idx = train_idx
        regime_label = "100%"
    else:
        pct_str = regime.replace("%", "")
        pct = float(pct_str) / 100.0
        if pct <= 0.0 or pct > 1.0:
            raise ValueError(f"Invalid data regime: {args.data_regime}")

        n_select = max(1, int(round(full_train_size * pct)))
        regime_rng = np.random.default_rng(args.seed + 12345 + args.fold_num)
        selected_train_idx = regime_rng.choice(
            train_idx, size=n_select, replace=False
        ).tolist()
        regime_label = f"{int(round(pct * 100))}%"

    train_idx = selected_train_idx

    print(f"Fold: {args.fold_num}/{args.num_folds}")
    print(f"Data regime: {regime_label}")
    print(
        f"Split -> train: {len(train_idx)} / {full_train_size}, "
        f"val: {len(val_idx)}, test: {len(test_idx)}"
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------------------- model
    model = DINOv3SegAdapters(repo_root, model_weight_path, num_classes=1).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "train_pixacc": [],
        "val_pixacc": [],
    }

    best_val_dice = -1.0
    best_path = os.path.join(result_root, "best_model.pth")

    # ------------------------------------------------------------ training
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice, tr_iou, tr_pix = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        va_loss, va_dice, va_iou, va_pix = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(va_iou)
        history["train_pixacc"].append(tr_pix)
        history["val_pixacc"].append(va_pix)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss {tr_loss:.4f} Dice {tr_dice:.4f} IoU {tr_iou:.4f} | "
            f"Val Loss {va_loss:.4f} Dice {va_dice:.4f} IoU {va_iou:.4f} "
            f"PixAcc {va_pix:.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": tr_loss,
                    "train/dice": tr_dice,
                    "train/iou": tr_iou,
                    "train/pixel_acc": tr_pix,
                    "val/loss": va_loss,
                    "val/dice": va_dice,
                    "val/iou": va_iou,
                    "val/pixel_acc": va_pix,
                },
                step=epoch,
            )

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best (val Dice {best_val_dice:.4f}) saved to {best_path}")

    # ------------------------------------------------------------- curves
    plot_curve(
        history, ["train_loss", "val_loss"], "Loss", "Loss",
        os.path.join(result_root, "curve_loss.png"),
    )
    plot_curve(
        history, ["train_dice", "val_dice"], "Dice", "Dice",
        os.path.join(result_root, "curve_dice.png"),
    )
    plot_curve(
        history, ["train_iou", "val_iou"], "IoU", "IoU",
        os.path.join(result_root, "curve_iou.png"),
    )
    plot_curve(
        history, ["train_pixacc", "val_pixacc"], "Pixel Acc", "Pixel Accuracy",
        os.path.join(result_root, "curve_pixacc.png"),
    )

    # -------------------------------------------------------------- test
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"\nLoaded best model (val Dice {best_val_dice:.4f}) for testing.")

    te_loss, te_dice, te_iou, te_pix = run_epoch(
        model, test_loader, criterion, device, optimizer=None
    )
    print(
        f"\n===== TEST =====\n"
        f"Loss {te_loss:.4f} | Dice {te_dice:.4f} | IoU {te_iou:.4f} | "
        f"PixAcc {te_pix:.4f}"
    )

    metrics = {
        "best_val_dice": best_val_dice,
        "test_loss": te_loss,
        "test_dice": te_dice,
        "test_iou": te_iou,
        "test_pixacc": te_pix,
        "history": history,
        "args": vars(args),
    }
    with open(os.path.join(result_root, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Qualitative overlays from the test split.
    overlay_dir = os.path.join(result_root, "overlays")
    save_overlays(
        model,
        full_dataset,
        test_idx[: min(8, len(test_idx))],
        device,
        overlay_dir,
        args.img_size,
    )

    if use_wandb:
        wandb.log(
            {
                "test/loss": te_loss,
                "test/dice": te_dice,
                "test/iou": te_iou,
                "test/pixel_acc": te_pix,
                "test/best_val_dice": best_val_dice,
            }
        )
        # Upload saved overlay images so they appear in the W&B media panel.
        overlay_images = sorted(
            os.path.join(overlay_dir, f)
            for f in os.listdir(overlay_dir)
            if f.endswith(".png")
        )
        if overlay_images:
            wandb.log(
                {
                    "test/overlays": [
                        wandb.Image(p, caption=os.path.basename(p))
                        for p in overlay_images
                    ]
                }
            )
        wandb.finish()

    print(f"\nAll outputs written to {result_root}")


if __name__ == "__main__":
    main()
