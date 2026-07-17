import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# segmentation dataset
from SegDataset import FracAtlasCocoSegDataset, build_seg_dataset
# segmentation task
from BaselineLinear import DINOv3LinearProbe_Seg
# split dataset
from sklearn.model_selection import train_test_split, KFold
from DataRegime import SegDataRegime

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

def parse_float_list(s):
    return [float(v.strip()) for v in str(s).split(",") if v.strip()]


def grid_search(args, lr_list):
    base_result_root = os.path.abspath(os.path.expanduser(args.result_root))
    grid_root = os.path.join(base_result_root, "grid_search")
    os.makedirs(grid_root, exist_ok=True)

    summary_rows = []
    best_lr = None
    best_score = float("-inf")

    for lr in lr_list:
        lr_tag = f"lr_{lr:g}".replace(".", "p")
        lr_result_root = os.path.join(grid_root, lr_tag)

        print("\n" + "=" * 80)
        print(f"GRID SEARCH | learning_rate = {lr}")
        print("=" * 80)

        results = run_training(args, current_lr=lr, result_root_override=lr_result_root)

        fold_metrics = []
        for regime, rows in results.items():
            fold_metrics.extend(rows)

        row = {
            "lr": float(lr),
            "mean_best_val_dice": float(np.mean([r["best_val_dice"] for r in fold_metrics])) if fold_metrics else None,
            "mean_test_loss": float(np.mean([r["test_loss"] for r in fold_metrics])) if fold_metrics else None,
            "mean_test_dice": float(np.mean([r["test_dice"] for r in fold_metrics])) if fold_metrics else None,
            "mean_test_iou": float(np.mean([r["test_iou"] for r in fold_metrics])) if fold_metrics else None,
            "mean_test_pixacc": float(np.mean([r["test_pixacc"] for r in fold_metrics])) if fold_metrics else None,
        }
        summary_rows.append(row)

        if row["mean_best_val_dice"] is not None and row["mean_best_val_dice"] > best_score:
            best_score = row["mean_best_val_dice"]
            best_lr = lr

    summary_csv_path = os.path.join(grid_root, "grid_search_summary.csv")
    summary_json_path = os.path.join(grid_root, "grid_search_summary.json")

    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv_path, index=False)
    summary_df.to_json(summary_json_path, orient="records", indent=2)

    print("\n===== Grid Search Summary =====")
    print(summary_df)

    if best_lr is not None:
        print(f"Best learning rate: {best_lr} (mean_best_val_dice={best_score:.4f})")

    print(f"Saved grid-search CSV summary to: {summary_csv_path}")
    print(f"Saved grid-search JSON summary to: {summary_json_path}")

def run_training(args, current_lr, result_root_override=None):
    result_root = os.path.abspath(os.path.expanduser(
        result_root_override if result_root_override is not None else args.result_root
    ))
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
        f"LR: {current_lr}, Image size: {args.img_size}"
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Abort training instead of falling back to CPU.")
    device = torch.device("cuda")

    if args.datasets_type.strip().lower() == "fracatlas":
        full_train_dataset = FracAtlasCocoSegDataset(
            coco_json, img_dir, img_size=args.img_size, augment=True, cache=True
        )
        full_eval_dataset = FracAtlasCocoSegDataset(
            coco_json, img_dir, img_size=args.img_size, augment=False, cache=True
        )
    else:
        full_train_dataset = build_seg_dataset(
            args.datasets_type, data_root, img_size=args.img_size, augment=True, cache=True
        )
        full_eval_dataset = build_seg_dataset(
            args.datasets_type, data_root, img_size=args.img_size, augment=False, cache=True
        )

    n = len(full_eval_dataset)
    indices = np.arange(n)

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.1, random_state=args.seed, shuffle=True
    )
    test_idx = np.array(test_idx)

    test_loader = DataLoader(
        Subset(full_eval_dataset, test_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    def parse_regime_list(s):
        s = str(s).strip()
        if s.lower() == "all":
            return ["5%", "10%", "50%", "100%"]
        return [x.strip() for x in s.split(",") if x.strip()]

    regime_seed_map = {
        "5%": 5,
        "10%": 10,
        "50%": 50,
        "100%": 100,
    }

    # regime_manager = SegDataRegime(train_val_idx, args.data_regime, seed=args.seed)
    # all_results = {}
    all_results = {}
    regimes = parse_regime_list(args.data_regime)

    for regime in regimes:
        regime_seed = regime_seed_map.get(regime, args.seed)
        regime_manager = SegDataRegime(train_val_idx, regime, seed=regime_seed)
        regime_items = list(regime_manager.get_data())
        if len(regime_items) == 0:
            print(f"\n===== Running data regime: {regime} but no data returned =====")
            continue

        # for X_tv, _, regime in regime_manager.get_data():
        for X_tv, _, regime_name in regime_items:
            # print(f"\n===== Running data regime: {regime} with {len(X_tv)} samples =====")
            print(f"\n===== Running data regime: {regime_name} with {len(X_tv)} samples =====")
            print(f"Using regime seed: {regime_seed}")
            regime_indices = np.array(X_tv)

            # regime_result_root = os.path.join(result_root, regime)
            regime_result_root = os.path.join(result_root, regime_name)
            os.makedirs(regime_result_root, exist_ok=True)

            fold_metrics = []
            fold_histories = []
            # kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
            kf = KFold(n_splits=5, shuffle=True, random_state=regime_seed)

            for fold, (tr_rel, va_rel) in enumerate(kf.split(regime_indices), start=1):
                print(f"\n--- Regime {regime_name} | Fold {fold}/5 ---")

                train_idx = regime_indices[tr_rel]
                val_idx = regime_indices[va_rel]

                train_loader = DataLoader(
                    Subset(full_train_dataset, train_idx.tolist()),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                val_loader = DataLoader(
                    Subset(full_eval_dataset, val_idx.tolist()),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )

                model = DINOv3LinearProbe_Seg(repo_root, model_weight_path, num_classes=1).to(device)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total params: {total_params:,}")
                print(f"Trainable params: {trainable_params:,}")

                criterion = DiceBCELoss(bce_weight=0.5)
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_lr,
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
                fold_dir = os.path.join(regime_result_root, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                best_path = os.path.join(fold_dir, "best_model.pth")

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

                    if va_dice > best_val_dice:
                        best_val_dice = va_dice
                        torch.save(model.state_dict(), best_path)
                        print(f"  -> new best (val Dice {best_val_dice:.4f}) saved to {best_path}")

                plot_curve(
                    history, ["train_loss", "val_loss"], "Loss", "Loss",
                    os.path.join(fold_dir, "curve_loss.png"),
                )
                plot_curve(
                    history, ["train_dice", "val_dice"], "Dice", "Dice",
                    os.path.join(fold_dir, "curve_dice.png"),
                )
                plot_curve(
                    history, ["train_iou", "val_iou"], "IoU", "IoU",
                    os.path.join(fold_dir, "curve_iou.png"),
                )
                plot_curve(
                    history, ["train_pixacc", "val_pixacc"], "Pixel Acc", "Pixel Accuracy",
                    os.path.join(fold_dir, "curve_pixacc.png"),
                )

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

                overlay_dir = os.path.join(fold_dir, "overlays")
                save_overlays(
                    model,
                    full_eval_dataset,
                    test_idx[: min(8, len(test_idx))],
                    device,
                    overlay_dir,
                    args.img_size,
                )

                fold_result = {
                    "fold": fold,
                    "best_val_dice": best_val_dice,
                    "test_loss": te_loss,
                    "test_dice": te_dice,
                    "test_iou": te_iou,
                    "test_pixacc": te_pix,
                    "history": history,
                    "regime_seed": regime_seed,
                }
                fold_metrics.append(fold_result)
                fold_histories.append({k: v[:] for k, v in history.items()})

                with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
                    json.dump(fold_result, f, indent=2)

            save_combined_training_curves(regime_result_root, fold_histories)

            all_results[regime_name] = fold_metrics
            with open(os.path.join(regime_result_root, "summary.json"), "w") as f:
                json.dump(fold_metrics, f, indent=2)

    with open(os.path.join(result_root, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll outputs written to {result_root}")
    return all_results

def save_combined_training_curves(save_dir, histories):
    """Plot all folds' val curves with mean and shaded std range for common metrics."""
    os.makedirs(save_dir, exist_ok=True)
    n_folds = len(histories)
    if n_folds == 0:
        return

    epochs = len(histories[0]["train_loss"])
    x = np.arange(1, epochs + 1)

    def stack_metric(key):
        arr = np.array([h[key] for h in histories])
        return arr

    metrics_to_plot = [
        ("val_loss", "Validation Loss"),
        ("val_dice", "Validation Dice"),
        ("val_iou", "Validation IoU"),
        ("val_pixacc", "Validation Pixel Accuracy"),
    ]

    for key, label in metrics_to_plot:
        try:
            data = stack_metric(key)
        except Exception:
            continue

        plt.figure(figsize=(8, 4))
        # plot each fold lightly
        for i in range(n_folds):
            plt.plot(x, data[i], color='gray', alpha=0.3)

        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        plt.plot(x, mean, color='C0', linewidth=2, label='Mean')
        plt.fill_between(x, mean - std, mean + std, color='C0', alpha=0.2)

        plt.title(f"{label} Across Folds")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/combined_{key}.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Linear Probing of DINOv3 for binary segmentation "
        "(FracAtlas fracture or BTXRD bone-tumor)"
    )
    parser.add_argument("--datasets-type", default="BTXRD", help="Which dataset to use: FracAtlas or BTXRD",)
    parser.add_argument("--result-root", default="/vol/miltank/users/lechi/dinov3/results_LoRA/FracAtlas_Seg", help="Results root directory (can be relative)",)
    parser.add_argument("--data-root",default="/vol/miltank/users/lechi/dinov3/datasets/BTXRD",help="Dataset root directory (can be relative)",)
    parser.add_argument("--coco-json",default="Annotations/COCO JSON/COCO_fracture_masks.json",help="Path to the COCO JSON file, relative to --data-root (or absolute)",)
    parser.add_argument("--img-subdir",default="images",help="Image directory relative to --data-root (Fractured/ is searched within)",)
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", default="1e-4", help='Learning rate; single value like "1e-4" or comma-separated like "1e-4,3e-4,1e-3"',)
    parser.add_argument("--img-size", type=int, default=448, help="Input image size")
    parser.add_argument('--data-regime', default='100%', help='Data regime: 5%, 10%, 50%, or 100% (can also be "all" or a comma-separated list like "5%,10%")')
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--model-weight-path",default="./models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",help="Path to DINOv3 model weights (can be relative)",)
    parser.add_argument("--repo-dir",default="./models/dinov3",help="Directory for torch.hub to load DINOv3 (can be relative)",)
    args = parser.parse_args()

    lr_list = parse_float_list(args.learning_rate)
    if len(lr_list) == 0:
        raise ValueError("No valid learning rate provided.")

    if len(lr_list) > 1:
        grid_search(args, lr_list)
    else:
        run_training(args, current_lr=lr_list[0])


if __name__ == "__main__":
    main()
