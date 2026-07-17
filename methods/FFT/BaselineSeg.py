import argparse
import json
import os
import sys
import re
import subprocess
from SegLoss import DiceBCELoss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from SegDatasets import FracAtlasCocoSegDataset, build_seg_dataset
from BaselineLinear import DINOv3Seg
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from DataRegime import SegDataRegime
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import monai.transforms as mt
from torchvision import models, transforms

# TODO: Change LoRA model into dinov3 + linear probe, and FFT
# TODO: Add DataRegime
# TODO: Add grid search for lr
# TODO: delete wandb logging and save image to results path

def grid_search(lr_list:list):
    
    # TODO: Maybe need to be modified
    """Run the training script repeatedly for each learning rate and report the best one.

    The function re-invokes this script as a subprocess so the existing single-run
    training path can be reused without duplicating the full training loop.
    """
    script_path = os.path.abspath(__file__)
    base_argv = sys.argv[1:]
    best_lr = None
    best_score = float('-inf')
    summary_rows = []

    def get_cli_arg(argv, flag, default=None):
        # Read an optional command-line value from the current invocation.
        for i in range(len(argv) - 1):
            if argv[i] == flag:
                return argv[i + 1]
        return default

    def replace_cli_arg(argv, flag, value):
        # Replace a CLI argument if it already exists, otherwise append it.
        updated = []
        replaced = False
        i = 0
        while i < len(argv):
            if argv[i] == flag and i + 1 < len(argv):
                updated.extend([flag, str(value)])
                replaced = True
                i += 2
            else:
                updated.append(argv[i])
                i += 1
        if not replaced:
            updated.extend([flag, str(value)])
        return updated

    def extract_last_metric(output_text, metric_name):
        # Grab the last reported metric value from the subprocess output.
        pattern = rf"{re.escape(metric_name)}:\s*([0-9.]+)"
        matches = re.findall(pattern, output_text)
        if not matches:
            return None
        return float(matches[-1])

    # Grid search is driven by the learning-rate list passed from `main`.
    print(f"Grid search over learning rates: {lr_list}", flush=True)

    # Keep grid-search results under the current result root so runs stay grouped.
    base_result_root = get_cli_arg(base_argv, '--result-root', '/vol/miltank/users/gayan/results/Baseline')
    grid_result_root = os.path.join(os.path.abspath(os.path.expanduser(base_result_root)), 'grid_search')

    for lr in lr_list:
        # Re-run this script with one learning rate at a time.
        lr_tag = str(lr).replace('.', 'p')
        run_argv = replace_cli_arg(base_argv, '--learning-rate', lr)
        run_argv = replace_cli_arg(run_argv, '--result-root', os.path.join(grid_result_root, f'lr_{lr_tag}'))
        cmd = [sys.executable, '-u', script_path] + run_argv

        print(f"\n===== Grid search run for lr={lr} =====", flush=True)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        output_chunks = []
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                if process.poll() is not None:
                    break
                continue
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            output_chunks.append(chunk)

        return_code = process.wait()
        output_text = b"".join(output_chunks).decode(errors='replace')
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        val_dice = extract_last_metric(output_text, 'val_dice')
        val_loss = extract_last_metric(output_text, 'val_loss')
        val_iou = extract_last_metric(output_text, 'val_iou')
        test_dice = extract_last_metric(output_text, 'test_dice')
        test_iou = extract_last_metric(output_text, 'test_iou')
        # TODO: Check Code
        summary_rows.append({
            'lr': lr,
            'val_dice': val_dice,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'test_dice': test_dice,
            'test_iou': test_iou,
        })

        # Track the best learning rate using validation Dice.
        if val_dice is not None and val_dice > best_score:
            best_score = val_dice
            best_lr = lr

    # Print a compact summary so the best lr is easy to spot in logs.
    print("\n===== Grid Search Summary =====", flush=True)
    for row in summary_rows:
        print(
            f"lr={row['lr']} | val_dice={row['val_dice']} | val_loss={row['val_loss']} | val_iou={row['val_iou']} | "
            f"test_dice={row['test_dice']} | test_iou={row['test_iou']}",
            flush=True,
        )

    if best_lr is not None:
        print(f"Best learning rate: {best_lr} (val_dice={best_score})", flush=True)
    else:
        print("Best learning rate could not be determined from the run output.", flush=True)

    return best_lr


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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

def build_seg_transforms(args):
    train_transform = None
    eval_transform = None

    if args.monai:
        train_transform = transforms.Compose([
            mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),  # Horizontal flip
            mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),  # Vertical flip
            mt.RandRotated(keys=["image", "mask"], prob=0.5, range_x=0.17, keep_size=True),  # ~10 degrees
            mt.RandZoomd(keys=["image", "mask"], prob=0.5, min_zoom=0.9, max_zoom=1.1, keep_size=True),
            mt.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.8, 1.2)),
            mt.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        ])

    return train_transform, eval_transform


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of DINOv3 for binary segmentation "
        "(FracAtlas fracture or BTXRD bone-tumor)"
    )
    parser.add_argument(
        "--datasets-type",
        default="FracAtlas",
        help="Which dataset to use: FracAtlas or BTXRD",
    )
    parser.add_argument(
        "--result-root",
        default="/vol/miltank/users/gayan/dinov3/results_LoRA/FracAtlas_Seg",
        help="Results root directory (can be relative)",
    )
    parser.add_argument(
        "--data-root",
        default="/vol/miltank/users/gayan/dinov3/datasets/FracAtlas",
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
        "--learning-rate", default=1e-4, help="Learning rate"
    )
    parser.add_argument("--img-size", type=int, default=448, help="Input image size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
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
        "--full-finetune",
        type=str2bool,
        default=False,
        help="Whether to fine-tune the entire DINOv3 backbone (default: False)",
    )
    parser.add_argument(
        "--data-regime",
        default="all",
        help=(
            "Data regime to use: 'all', '5%', '10%', '50%', '100%', or a comma-separated "
            "list of regimes (e.g., '5%,10%')."
        ),
    )
    parser.add_argument(
        "--monai",
        type=str2bool,
        default=False,
        help="Whether to use MONAI transforms (default: False)"
    )
    parser.add_argument(
        "--preload-to-ram",
        type=str2bool,
        default=False,
        help="Whether to preload the dataset into RAM (default: False)"
    )

    args = parser.parse_args()

    learning_rate_list = [v.strip() for v in str(args.learning_rate).split(',') if v.strip()]
    if len(learning_rate_list) == 0:
        learning_rate_list = ['1e-3']

    if len(learning_rate_list) > 2:
        grid_search(learning_rate_list)
        return
    if len(learning_rate_list) > 1:
        print(f"Multiple learning rates provided but grid search only triggers when there are more than two. Using the first value: {learning_rate_list[0]}")

    args.learning_rate = float(learning_rate_list[0])
    
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

    device = torch.device("cuda")

    
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_tfr, tfr = build_seg_transforms(args)
    
    available_regimes = ['5%','10%','50%','100%']
    if args.data_regime == 'all':
        regimes_list = available_regimes
    elif ',' in args.data_regime:
        regimes_list = [r.strip() for r in args.data_regime.split(',') if r.strip()]
    else:
        regimes_list = [args.data_regime]

    if args.datasets_type.strip().lower() == "fracatlas":
        full_dataset_eval = FracAtlasCocoSegDataset(
            coco_json=coco_json, img_dir=img_dir, img_size=args.img_size, transform=tfr, preload_to_ram=args.preload_to_ram
        )
        full_dataset_train = FracAtlasCocoSegDataset(
            coco_json=coco_json, img_dir=img_dir, img_size=args.img_size, transform=train_tfr, preload_to_ram=args.preload_to_ram
        )
    else:
        full_dataset_eval = build_seg_dataset(
            args.datasets_type, data_root, img_size=args.img_size, transform=tfr, preload_to_ram=args.preload_to_ram
        )
        full_dataset_train = build_seg_dataset(
            args.datasets_type, data_root, img_size=args.img_size, transform=train_tfr, preload_to_ram=args.preload_to_ram
        )

    
    # TODO: inplement Five-fold and DataRegime
    indices = np.arange(len(full_dataset_eval))
    train_val_indices, test_indices = train_test_split(
        indices, test_size=0.1, 
        random_state=args.seed, 
        )    
    
    # full_train_eval_dataset = Subset(full_dataset_eval, train_val_indices)
    test_dataset = Subset(full_dataset_eval, test_indices)

    regime_manager = SegDataRegime(train_val_indices, args.data_regime)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    for regime_indices, regime in regime_manager.get_data():
        print(f"\n===== Running data regime: {regime} with {len(regime_indices)} samples =====")

        fold_histories = []
        fold_results = []
        result_root = os.path.join(args.result_root, regime)
        
        if not os.path.exists(result_root):
            os.makedirs(result_root, exist_ok=True)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(regime_indices), start=1):
            
            print(f"Fold {fold}: train {len(train_idx)}, val {len(val_idx)}")
            
            train_indices = regime_indices[train_idx]
            val_indices = regime_indices[val_idx]
            train_dataset = Subset(full_dataset_train, train_indices)
            val_dataset = Subset(full_dataset_eval, val_indices)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
            
            model = DINOv3Seg(repo_root, model_weight_path, num_classes=1).to(device)
            
            criterion = DiceBCELoss(bce_weight=0.5)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
            )
            
            fold_history = {
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
            best_epoch = -1
            best_path = os.path.join(result_root, "best_model.pth")
            
            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_dice, tr_iou, tr_pix = run_epoch(
                    model, train_loader, criterion, device, optimizer
                )
                va_loss, va_dice, va_iou, va_pix = run_epoch(
                    model, val_loader, criterion, device, optimizer=None
                )

                fold_history["train_loss"].append(tr_loss)
                fold_history["val_loss"].append(va_loss)
                fold_history["train_dice"].append(tr_dice)
                fold_history["val_dice"].append(va_dice)
                fold_history["train_iou"].append(tr_iou)
                fold_history["val_iou"].append(va_iou)
                fold_history["train_pixacc"].append(tr_pix)
                fold_history["val_pixacc"].append(va_pix)

                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss {tr_loss:.4f} Dice {tr_dice:.4f} IoU {tr_iou:.4f} | "
                    f"Val Loss {va_loss:.4f} Dice {va_dice:.4f} IoU {va_iou:.4f} "
                    f"PixAcc {va_pix:.4f}"
                )
                if va_dice > best_val_dice:
                    best_epoch = epoch - 1
                    best_val_dice = va_dice
                    torch.save(model.state_dict(), best_path)
                    # print(f"  -> new best (val Dice {best_val_dice:.4f}) saved to {best_path}")
            
            fold_result_root = os.path.join(result_root, f"fold_{fold}")
            os.makedirs(fold_result_root, exist_ok=True)
            
            plot_curve(
                fold_history,
                keys=["train_loss", "val_loss"],
                ylabel="Loss",
                title=f"Fold {fold} Loss Curves",
                out_path=os.path.join(fold_result_root, f"fold_{fold}_loss_curve.png"),
            )
            plot_curve(
                fold_history,
                keys=["train_dice", "val_dice"],
                ylabel="Dice",
                title=f"Fold {fold} Dice Curves",
                out_path=os.path.join(fold_result_root, f"fold_{fold}_dice_curve.png"),
            )
            plot_curve(
                fold_history,
                keys=["train_iou", "val_iou"],
                ylabel="IoU",
                title=f"Fold {fold} IoU Curves",
                out_path=os.path.join(fold_result_root, f"fold_{fold}_iou_curve.png"),
            )
    # -------------------------------------------------------------- test
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device))
                # print(f"\nLoaded best model (val Dice {best_val_dice:.4f}) for testing.")

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
                "test_pixacc": te_pix
            }
            
            with open(os.path.join(result_root, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            # Qualitative overlays from the test split.
            overlay_dir = os.path.join(result_root, "overlays")
            save_overlays(
                model,
                test_dataset,
                list(range(min(8, len(test_dataset)))),
                device,
                overlay_dir,
                args.img_size,
            )
            
            print(f"\nAll outputs in fold {fold} written to {result_root}")
        
            def safe_get(lst, idx, default=0.0):
                try:
                    return lst[idx]
                except Exception:
                    return default
            
            fold_results.append({
                "fold": fold,
                "train_loss": safe_get(fold_history["train_loss"], best_epoch, 0.0),
                "val_loss": safe_get(fold_history["val_loss"], best_epoch, 0.0),
                "test_loss": te_loss,
                "train_dice": safe_get(fold_history["train_dice"], best_epoch, 0.0),
                "val_dice": safe_get(fold_history["val_dice"], best_epoch, 0.0),
                "test_dice": te_dice,
                "train_iou": safe_get(fold_history["train_iou"], best_epoch, 0.0),
                "val_iou": safe_get(fold_history["val_iou"], best_epoch, 0.0),
                "test_iou": te_iou,
                "train_pixacc": safe_get(fold_history["train_pixacc"], best_epoch, 0.0),
                "val_pixacc": safe_get(fold_history["val_pixacc"], best_epoch, 0.0),
                "test_pixacc": te_pix,
                "best_epoch": best_epoch,
                "best_val_dice": best_val_dice,
            })

            fold_histories.append({
                "train_loss": fold_history["train_loss"],
                "val_loss": fold_history["val_loss"],
                "train_dice": fold_history["train_dice"],
                "val_dice": fold_history["val_dice"],
                "train_iou": fold_history["train_iou"],
                "val_iou": fold_history["val_iou"],
                "train_pixacc": fold_history["train_pixacc"],
                "val_pixacc": fold_history["val_pixacc"],
            })
        completed_folds = len(fold_results)
        print(f"\nCompleted regime {regime}: {completed_folds} folds finished.")

        save_combined_training_curves(result_root, fold_histories)
                
        print(f"\nPer-fold results for regime {regime}:")
        keys = ["train_loss", "val_loss", "train_dice", "val_dice", "train_iou", "val_iou", "train_pixacc", "val_pixacc", "test_loss", "test_dice", "test_iou", "test_pixacc"]
        for r in fold_results:
            print(f"Fold {r['fold']}: " + "| ".join([f"{k}: {r.get(k,0):.4f}" for k in keys]))
            
        avg = {}
        for k in keys:
            avg[k] = np.mean([r.get(k, 0.0) for r in fold_results])

        print(f"\n{completed_folds}-fold averages for regime {regime}:")
        print(", ".join([f"{k}: {avg[k]:.4f}" for k in keys]))
if __name__ == "__main__":
    main()
