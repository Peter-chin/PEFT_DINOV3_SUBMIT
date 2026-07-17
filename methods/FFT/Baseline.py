import os
import re
import subprocess
import sys
import matplotlib.pyplot as plt
from BaselineLinear import DINOv3LinearProbe, DINOv3PromptProbe
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset
from Datasets import ExcelDataset, FracAtlasDataset, preload_dataset_to_ram, InMemoryDataset
from DataRegime import DataRegime
import numpy as np
import copy
import monai.transforms as mt
from DataAugmentation import Mixup


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params: {total - trainable:,}")

def build_subset(dataset, indices):
    """Build a torch.utils.data.Subset from any dataset and index array."""
    return Subset(dataset, list(indices))

def grid_search(lr_list:list):
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

        val_auc = extract_last_metric(output_text, 'val_auc')
        val_acc = extract_last_metric(output_text, 'val_acc')
        test_auc = extract_last_metric(output_text, 'test_auc')
        test_acc = extract_last_metric(output_text, 'test_acc')

        summary_rows.append({
            'lr': lr,
            'val_auc': val_auc,
            'val_acc': val_acc,
            'test_auc': test_auc,
            'test_acc': test_acc,
        })

        # Track the best learning rate using validation AUC.
        if val_auc is not None and val_auc > best_score:
            best_score = val_auc
            best_lr = lr

    # Print a compact summary so the best lr is easy to spot in logs.
    print("\n===== Grid Search Summary =====", flush=True)
    for row in summary_rows:
        print(
            f"lr={row['lr']} | val_auc={row['val_auc']} | val_acc={row['val_acc']} | "
            f"test_auc={row['test_auc']} | test_acc={row['test_acc']}",
            flush=True,
        )

    if best_lr is not None:
        print(f"Best learning rate: {best_lr} (val_auc={best_score})", flush=True)
    else:
        print("Best learning rate could not be determined from the run output.", flush=True)

    return best_lr

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('yes', 'true', 't', 'y', '1'):
        return True
    if s in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []
    all_preds = []

    mixup = Mixup(alpha=mixup_alpha) if mixup_alpha > 0 else None

    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        y_a = y_b = lam = None
        if mixup is not None or mixup_alpha > 0:
            x, y_a, y_b, lam = mixup(x, y)
        out = model(x)
        if mixup is not None or mixup_alpha > 0:
            loss_a = criterion(out, y_a)
            loss_b = criterion(out, y_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:  
            loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        check_model_grad = False 
        # for name, param in model.named_parameters():
        #     if "backbone" in name and param.grad is not None:
        #         check_model_grad = True
        #         break
        # if check_model_grad:
        #     print(f"Gradients are flowing to the backbone parameters.")
        optimizer.step()

        total_loss += loss.item()

        probs = torch.softmax(out, dim=1)[:, 1]
        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    acc = correct / total if total > 0 else 0.0
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    # compute average precision and f1 and balanced accuracy with guards
    try:
        ap = average_precision_score(all_labels, all_probs)
    except Exception:
        ap = 0.0

    try:
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    except Exception:
        f1 = 0.0

    try:
        bal = balanced_accuracy_score(all_labels, all_preds)
    except Exception:
        bal = 0.0

    return total_loss / len(loader), acc, auc, ap, f1, bal, all_labels, all_preds

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()

            probs = torch.softmax(out, dim=1)[:, 1]

            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

    acc = correct / total if total > 0 else 0.0
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    try:
        ap = average_precision_score(all_labels, all_probs)
    except Exception:
        ap = 0.0

    try:
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    except Exception:
        f1 = 0.0

    try:
        bal = balanced_accuracy_score(all_labels, all_preds)
    except Exception:
        bal = 0.0

    return total_loss / len(loader), acc, auc, ap, f1, bal, all_labels, all_preds, all_probs


def save_training_curve(save_dir, train_losses, val_losses, train_accs, val_accs, train_aucs, val_aucs, train_aps=None, val_aps=None, train_f1s=None, val_f1s=None, train_bals=None, val_bals=None):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.title("Training Curve (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curve_loss.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accs, marker='o', label='Train Acc')
    plt.plot(epochs, val_accs, marker='o', label='Val Accuracy')
    plt.plot(epochs, train_aucs, marker='o', label='Train AUC')
    plt.plot(epochs, val_aucs, marker='o', label='Val AUC')
    plt.title("Validation Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curve_val_metrics.png")
    plt.close()

    # optional plots for AP, F1, Balanced Accuracy
    if train_aps is not None and val_aps is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_aps, marker='o', label='Train AP')
        plt.plot(epochs, val_aps, marker='o', label='Val AP')
        plt.title("Average Precision")
        plt.xlabel("Epoch")
        plt.ylabel("AP")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curve_ap.png")
        plt.close()

    if train_f1s is not None and val_f1s is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_f1s, marker='o', label='Train F1')
        plt.plot(epochs, val_f1s, marker='o', label='Val F1')
        plt.title("F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curve_f1.png")
        plt.close()

    if train_bals is not None and val_bals is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_bals, marker='o', label='Train Balanced Acc')
        plt.plot(epochs, val_bals, marker='o', label='Val Balanced Acc')
        plt.title("Balanced Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Balanced Acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curve_balanced_acc.png")
        plt.close()


def save_confusion_matrix_plot(save_dir, y_true, y_pred, fold, title_suffix="Validation"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize='true') # normalize='true' to get percentages instead of counts

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"{title_suffix} Confusion Matrix (Fold {fold})")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    plt.xticks(range(cm.shape[1]))
    plt.yticks(range(cm.shape[0]))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{title_suffix.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

def save_results(save_dir, fold_results):
    os.makedirs(save_dir, exist_ok=True)

    accs = [r["val_acc"] for r in fold_results]
    aucs = [r["val_auc"] for r in fold_results]
    aps = [r.get("val_ap", 0.0) for r in fold_results]
    f1s = [r.get("val_f1", 0.0) for r in fold_results]
    bals = [r.get("val_bal", 0.0) for r in fold_results]
    test_accs = [r.get("test_acc", 0.0) for r in fold_results]
    test_aucs = [r.get("test_auc", 0.0) for r in fold_results]
    test_aps = [r.get("test_ap", 0.0) for r in fold_results]
    test_f1s = [r.get("test_f1", 0.0) for r in fold_results]
    test_bals = [r.get("test_bal", 0.0) for r in fold_results]
    val_losses = [r.get("val_loss", 0.0) for r in fold_results]
    train_losses = [r.get("train_loss", 0.0) for r in fold_results]
    train_accs = [r.get("train_acc", 0.0) for r in fold_results]

    plt.figure(figsize=(4, 3))
    plt.plot(accs, marker='o')
    plt.title("5-Fold Validation Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.savefig(f"{save_dir}/acc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(aucs, marker='o')
    plt.title("5-Fold Validation AUC")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.savefig(f"{save_dir}/auc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(aps, marker='o')
    plt.title("5-Fold Validation AP")
    plt.xlabel("Fold")
    plt.ylabel("AP")
    plt.savefig(f"{save_dir}/ap.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(f1s, marker='o')
    plt.title("5-Fold Validation F1")
    plt.xlabel("Fold")
    plt.ylabel("F1")
    plt.savefig(f"{save_dir}/f1.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(bals, marker='o')
    plt.title("5-Fold Validation Balanced Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Balanced Acc")
    plt.savefig(f"{save_dir}/balanced_acc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_accs, marker='o')
    plt.title("5-Fold Test Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Test Accuracy")
    plt.savefig(f"{save_dir}/test_acc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_aucs, marker='o')
    plt.title("5-Fold Test AUC")
    plt.xlabel("Fold")
    plt.ylabel("Test AUC")
    plt.savefig(f"{save_dir}/test_auc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_aps, marker='o')
    plt.title("5-Fold Test AP")
    plt.xlabel("Fold")
    plt.ylabel("Test AP")
    plt.savefig(f"{save_dir}/test_ap.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_f1s, marker='o')
    plt.title("5-Fold Test F1")
    plt.xlabel("Fold")
    plt.ylabel("Test F1")
    plt.savefig(f"{save_dir}/test_f1.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_bals, marker='o')
    plt.title("5-Fold Test Balanced Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Test Balanced Acc")
    plt.savefig(f"{save_dir}/test_balanced_acc.png")
    plt.close()


def save_combined_training_curves(save_dir, histories):
    """Plot all folds' val curves with mean and shaded std range for common metrics."""
    os.makedirs(save_dir, exist_ok=True)
    n_folds = len(histories)
    if n_folds == 0:
        return

    epochs = len(histories[0]["train_losses"])
    x = np.arange(1, epochs + 1)

    def stack_metric(key):
        arr = np.array([h[key] for h in histories])
        return arr

    metrics_to_plot = [
        ("val_losses", "Loss", True),
        ("val_accs", "Accuracy", False),
        ("val_aucs", "AUC", False),
        ("val_aps", "AP", False),
        ("val_f1s", "F1", False),
        ("val_bals", "Balanced Acc", False),
    ]

    for key, label, invert in metrics_to_plot:
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
    parser = argparse.ArgumentParser(description='Baseline training with configurable paths')
    parser.add_argument('--result-root', default='/vol/miltank/users/gayan/results/Baseline', help='Results root directory (can be relative)')
    parser.add_argument('--datasets-type', default='BTXRD', help='Which dataset to use: BTXRD or FracAtlas')
    parser.add_argument('--balanced', type=str2bool, default=True, help='Boolean flag. Accepts true/false, 1/0, yes/no.')
    parser.add_argument('--data-root', default='/vol/miltank/users/gayan/datasets/BTXRD/BTXRD', help='Dataset root directory containing dataset.xlsx and images (can be relative)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--learning-rate', default='1e-3', help='Learning rate for optimizer. Accepts a single value or a comma-separated list.')
    parser.add_argument('--full-finetune', type=str2bool, default=False, help='Enable full fine-tuning of backbone (default: linear probe only)')
    parser.add_argument('--model-weight-path', default='./models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', help='Path to DINOv3 model weights (can be relative)')
    parser.add_argument('--repo-dir', default='./models/dinov3', help='Directory for torch.hub to load DINOv3 (can be relative)')
    parser.add_argument('--data-regime', default='100%', help='Data regime: 5%, 10%, 50%, or 100% (can also be "all" or a comma-separated list like "5%,10%")')
    parser.add_argument('--n-prompt-tokens', default='1,4,8,16', help='Prompt token sweep, e.g. "1,4,8,16" or a single value like "4", if using 0 then it wont pass n_prompt_tokens to the model and will just use the default DINOv3 forward without prompt tokens.')
    parser.add_argument('--preload-ram', type=str2bool, default=False, help='Preload the full dataset into RAM to avoid repeated disk I/O during training.')
    parser.add_argument('--monai', type=str2bool, default=False, help='Whether to apply data augmentation during training.')
    parser.add_argument('--mixup-alpha', type=float, default=1.0, help='Alpha parameter for Mixup data augmentation.')
    args = parser.parse_args()

    # resolve relative/tilde paths to absolute
    base_result_root = os.path.abspath(os.path.expanduser(args.result_root))
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    repo_root = os.path.abspath(os.path.expanduser(args.repo_dir))
    model_weight_path = os.path.abspath(os.path.expanduser(args.model_weight_path))
    learning_rate_list = [v.strip() for v in str(args.learning_rate).split(',') if v.strip()]
    if len(learning_rate_list) == 0:
        learning_rate_list = ['1e-3']

    print(f"Using result root: {base_result_root}")
    print(f"Using data root:   {data_root}")
    print(f"Using repo dir:    {repo_root}")
    print(f"Using model weights: {model_weight_path}")
    print(f"Dataset type: {args.datasets_type}")
    print(f"Data regime: {args.data_regime}")
    # print(f"Data augmentation: {'enabled' if args.monai else 'disabled'}")
    print(f"Data augmentation: {'enabled with MONAI' if args.monai else 'disabled'}")
    print(f"Mixup alpha: {args.mixup_alpha}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    print(f"Fine-tuning mode: {'Full fine-tuning' if args.full_finetune else 'Linear probe only'}")
    print("Prompt tuning: enabled for independent DINOv3 prompt tokens")

    if len(learning_rate_list) > 2:
        grid_search(learning_rate_list)
        return
    if len(learning_rate_list) > 1:
        print(f"Multiple learning rates provided but grid search only triggers when there are more than two. Using the first value: {learning_rate_list[0]}")

    args.learning_rate = float(learning_rate_list[0])

    if ',' in str(args.n_prompt_tokens):
        prompt_token_list = [int(v.strip()) for v in str(args.n_prompt_tokens).split(',') if v.strip()]
    else:
        prompt_token_list = [int(args.n_prompt_tokens)]

    if prompt_token_list != [0]:
        print("Full Fine-tuning mode is incompatible with prompt token tuning. Disabling full fine-tuning.")
        args.full_finetune = False

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # fold results/histories will be created per data-regime below

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.datasets_type == "BTXRD":
        df_path = os.path.join(data_root, 'dataset.xlsx')
        df = pd.read_excel(df_path)
        print(f"Loaded BTXRD dataset with {len(df)} samples")
    elif args.datasets_type == "FracAtlas":
        df_path = os.path.join(data_root, 'dataset.csv')
        df = pd.read_csv(df_path)
        print(f"Loaded FracAtlas dataset with {len(df)} samples")
    else:
        print(f"Error: Unsupported dataset type {args.datasets_type}")
        return
    
    # Support running multiple data regimes in one invocation.
    # `--data-regime` accepts: single value like '100%', comma-separated list like '5%,10%', or 'all'.
    available_regimes = ['5%','10%','50%','100%']
    if args.data_regime == 'all':
        regimes_list = available_regimes
    elif ',' in args.data_regime:
        regimes_list = [r.strip() for r in args.data_regime.split(',') if r.strip()]
    else:
        regimes_list = [args.data_regime]

    # We'll loop over regimes_list below; keep the original full train+val indices as a source.
            
    tfr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]), 
    ])
    
    
    train_tfr = tfr
    if args.monai:
        # train_tfr = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(15),
        #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225]),
        # ])
        train_tfr = mt.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # MONAI augmentations
                mt.RandFlip(prob=0.5, spatial_axis=1),  # Horizontal flip
                mt.RandFlip(prob=0.5, spatial_axis=0),  # Vertical flip
                mt.RandRotate(range_x=0.17, prob=0.5, keep_size=True),  # ~10 degrees
                mt.RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.1, keep_size=True),
                mt.RandAdjustContrast(prob=0.5, gamma=(0.8, 1.2)),
                mt.RandGaussianNoise(prob=0.2, mean=0.0, std=0.01),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    img_dir = os.path.join(data_root, 'images')

    if args.preload_ram:
        print("Preloading dataset into RAM...")

    X = df.index.values
    y = df['tumor'].values if args.datasets_type == "BTXRD" else df['fractured'].values

    cached_dataset = None
    if args.preload_ram:
        cached_dataset, df = preload_dataset_to_ram(df, img_dir, args.datasets_type)
        X = np.arange(len(df))
        y = df['tumor'].values if args.datasets_type == "BTXRD" else df['fractured'].values

    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    test_df = df.iloc[X_test].reset_index(drop=True)

    # Create test dataset and loader (test split is fixed above with random_state=42)
    if args.preload_ram:
        # test_dataset = build_subset(cached_dataset, X_test)
        test_dataset = InMemoryDataset(
            [cached_dataset.images[i] for i in X_test],
            [cached_dataset.labels[i] for i in X_test],
            transform=tfr
        )
    else:
        if args.datasets_type == "BTXRD":
            test_dataset = ExcelDataset(test_df, img_dir=img_dir, transform=tfr)
        elif args.datasets_type == "FracAtlas":
            test_dataset = FracAtlasDataset(test_df, img_dir=img_dir, transform=tfr)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    regime_manager = DataRegime(X_train_val, y_train_val, args.data_regime)

    for n_prompt_tokens in prompt_token_list:
        print(f"\n\n=== Running with n_prompt_tokens = {n_prompt_tokens} ===")
    # Per-regime loop: perform subsampling (if any) then run 5-fold for that regime
        for X_tv, y_tv, regime in regime_manager.get_data():
            print(f"\n===== Running data regime: {regime} with {len(X_tv)} samples =====")
            
            # result_root = os.path.join(base_result_root, regime)
            result_root = os.path.join(base_result_root, f"prompt_{n_prompt_tokens}", regime)

            # reset per-regime fold accumulators
            fold_results = []
            fold_histories = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_tv, y_tv)):
                fold_num = fold + 1
                fold_save_dir = f"{result_root}/fold_{fold_num}"
                epochs = args.epochs
                batch_size = args.batch_size
                lr = float(args.learning_rate)
                
                print(f"\n========== Fold {fold_num} ==========")

                train_df = df.iloc[X_tv[train_idx]].reset_index(drop=True)
                val_df = df.iloc[X_tv[val_idx]].reset_index(drop=True)
                train_indices = X_tv[train_idx]
                val_indices = X_tv[val_idx]

                if n_prompt_tokens == 1 or n_prompt_tokens == 0:
                    print("Using linear probe (no prompt tokens)")
                    model = DINOv3LinearProbe(2, repo_root, model_weight_path, full_finetune=args.full_finetune).to(device)
                else:
                    print(f"Using DINOv3 with {n_prompt_tokens} prompt tokens")
                    model = DINOv3PromptProbe(2, repo_root, model_weight_path, n_prompt_tokens=n_prompt_tokens).to(device)
                # model = DINOv3LinearProbe(2, repo_root, model_weight_path, full_finetune=args.full_finetune).to(device)
                if args.datasets_type == "BTXRD":
                    if args.preload_ram:
                        # train_dataset = build_subset()
                        train_dataset = InMemoryDataset(
                            [cached_dataset.images[i] for i in train_indices],
                            [cached_dataset.labels[i] for i in train_indices],
                            transform=train_tfr
                        )
                        val_dataset = InMemoryDataset(
                            [cached_dataset.images[i] for i in val_indices],
                            [cached_dataset.labels[i] for i in val_indices],
                            transform=tfr
                        )
                    else:
                        train_dataset = ExcelDataset(train_df, img_dir=img_dir, transform=train_tfr)
                        val_dataset = ExcelDataset(val_df, img_dir=img_dir, transform=tfr)
                    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
                elif args.datasets_type == "FracAtlas":
                    if args.preload_ram:
                        train_dataset = InMemoryDataset(
                            [cached_dataset.images[i] for i in train_indices],
                            [cached_dataset.labels[i] for i in train_indices],
                            transform=train_tfr
                        )
                        val_dataset = InMemoryDataset(
                            [cached_dataset.images[i] for i in val_indices],
                            [cached_dataset.labels[i] for i in val_indices],
                            transform=tfr
                        )
                    else:
                        train_dataset = FracAtlasDataset(train_df, img_dir=img_dir, transform=train_tfr)
                        val_dataset = FracAtlasDataset(val_df, img_dir=img_dir, transform=tfr)
                    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
                else:
                    print(f"Error: Unsupported dataset type {args.datasets_type}")
                    return

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                # Set optimizer based on fine-tuning mode
                if args.full_finetune:
                    # Use differential learning rates for backbone and classifier
                    backbone_params = model.backbone.parameters()
                    classifier_params = model.classifier.parameters()
                    
                    optimizer = torch.optim.Adam([
                        {'params': backbone_params, 'lr': lr},  # Backbone lr
                        {'params': classifier_params, 'lr': lr}        # Classifier lr
                    ])
                else:
                    # Linear probe: only train the classifier
                    for param in model.backbone.parameters():
                        param.requires_grad = False
                    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)

                if args.balanced:
                    criterion = nn.CrossEntropyLoss()
                else:
                    class_counts = train_df["fractured"].value_counts().sort_index()
                    class_weights = len(train_df) / (2 * torch.tensor(class_counts.values, dtype=torch.float32))
                    print("class_counts:", class_counts.to_dict())
                    print("class_weights:", class_weights.tolist())
                    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

                # optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)

                train_losses, val_losses = [], []
                train_accs, val_accs = [], []
                train_aucs, val_aucs = [], []
                train_aps, val_aps = [], []
                train_f1s, val_f1s = [], []
                train_bals, val_bals = [], []

                
                # store per-epoch validation predictions so we can pick best epoch later
                val_labels_per_epoch = []
                val_preds_per_epoch = []
                val_probs_per_epoch = []

                last_val_labels = []
                last_val_preds = []

                best_val_auc = float('-inf')
                best_epoch_idx = -1
                best_state = None

                count_params(model)
                
                for epoch in range(epochs):
                    train_loss, train_acc, train_auc, train_ap, train_f1, train_bal, _, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, mixup_alpha=args.mixup_alpha)
                    val_loss, val_acc, val_auc, val_ap, val_f1, val_bal, val_labels, val_preds, val_probs = evaluate(model, val_loader, criterion, device)

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                    train_accs.append(train_acc)
                    val_accs.append(val_acc)

                    train_aucs.append(train_auc)
                    val_aucs.append(val_auc)

                    train_aps.append(train_ap)
                    val_aps.append(val_ap)

                    train_f1s.append(train_f1)
                    val_f1s.append(val_f1)

                    train_bals.append(train_bal)
                    val_bals.append(val_bal)

                    # record per-epoch validation outputs
                    val_labels_per_epoch.append(val_labels)
                    val_preds_per_epoch.append(val_preds)
                    val_probs_per_epoch.append(val_probs)

                    last_val_labels = val_labels
                    last_val_preds = val_preds

                    # checkpoint best model by validation AUC
                    try:
                        cur_val_auc = float(val_auc)
                    except Exception:
                        cur_val_auc = float('-inf')
                    if cur_val_auc > best_val_auc:
                        best_val_auc = cur_val_auc
                        best_epoch_idx = epoch
                        best_state = copy.deepcopy(model.state_dict())

                    print(f"Epoch {epoch+1}: "
                        f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                        f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
                        f"Train AP {train_ap:.4f} | Val AP {val_ap:.4f} | "
                        f"Train F1 {train_f1:.4f} | Val F1 {val_f1:.4f} | "
                        f"Train Bal {train_bal:.4f} | Val Bal {val_bal:.4f} | "
                        f"Val AUC {val_auc:.4f}")

                # save per-fold curves
                save_training_curve(fold_save_dir, train_losses, val_losses, train_accs, val_accs, train_aucs, val_aucs, train_aps, val_aps, train_f1s, val_f1s, train_bals, val_bals)

                # determine best epoch index (fallback to recorded best_epoch_idx)
                if best_epoch_idx >= 0:
                    best_idx = best_epoch_idx
                else:
                    try:
                        best_idx = int(np.nanargmax(np.array(val_aucs)))
                    except Exception:
                        best_idx = len(val_aucs) - 1 if len(val_aucs) > 0 else 0

                # use best epoch's validation preds for confusion matrix
                try:
                    best_val_labels = val_labels_per_epoch[best_idx]
                    best_val_preds = val_preds_per_epoch[best_idx]
                except Exception:
                    best_val_labels = last_val_labels
                    best_val_preds = last_val_preds

                save_confusion_matrix_plot(fold_save_dir, best_val_labels, best_val_preds, fold_num, "Validation")

                # If we saved a best checkpoint, load it and evaluate test on the best model
                if best_state is not None:
                    model.load_state_dict(best_state)

                test_loss, test_acc, test_auc, test_ap, test_f1, test_bal, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion, device)

                print(f"Test Results for Fold {fold_num} (best-epoch model): "
                    f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f} | Test AUC {test_auc:.4f} | "
                    f"Test AP {test_ap:.4f} | Test F1 {test_f1:.4f} | Test Bal {test_bal:.4f}")

                # Save test confusion matrix (for best model)
                save_confusion_matrix_plot(fold_save_dir, test_labels, test_preds, fold_num, "Test")

            # record final metrics for this fold
                # record best-epoch metrics for this fold (use best_idx)
                def safe_get(lst, idx, default=0.0):
                    try:
                        return lst[idx]
                    except Exception:
                        return default

                fold_results.append({
                    "fold": fold_num,
                    "train_acc": safe_get(train_accs, best_idx, 0.0),
                    "val_acc": safe_get(val_accs, best_idx, 0.0),
                    "test_acc": test_acc,
                    "train_loss": safe_get(train_losses, best_idx, 0.0),
                    "val_loss": safe_get(val_losses, best_idx, 0.0),
                    "test_loss": test_loss,
                    "train_auc": safe_get(train_aucs, best_idx, 0.0),
                    "val_auc": safe_get(val_aucs, best_idx, 0.0),
                    "test_auc": test_auc,
                    "train_ap": safe_get(train_aps, best_idx, 0.0),
                    "val_ap": safe_get(val_aps, best_idx, 0.0),
                    "test_ap": test_ap,
                    "train_f1": safe_get(train_f1s, best_idx, 0.0),
                    "val_f1": safe_get(val_f1s, best_idx, 0.0),
                    "test_f1": test_f1,
                    "train_bal": safe_get(train_bals, best_idx, 0.0),
                    "val_bal": safe_get(val_bals, best_idx, 0.0),
                    "test_bal": test_bal
                })

                fold_histories.append({
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accs": train_accs,
                    "val_accs": val_accs,
                    "train_aucs": train_aucs,
                    "val_aucs": val_aucs,
                    "train_aps": train_aps,
                    "val_aps": val_aps,
                    "train_f1s": train_f1s,
                    "val_f1s": val_f1s,
                    "train_bals": train_bals,
                    "val_bals": val_bals
                })

            completed_folds = len(fold_results)
            print(f"\nCompleted regime {regime} for {n_prompt_tokens} prompt tokens: {completed_folds} folds finished.")

            # Generate 5-fold results only AFTER all folds are complete
            save_results(result_root, fold_results)
            save_combined_training_curves(result_root, fold_histories)

            # print per-fold metrics and averages for this regime
            print(f"\nPer-fold results for regime {regime}:")
            keys = ["train_acc","val_acc","test_acc","train_loss","val_loss","test_loss","train_ap","val_ap","test_ap","train_f1","val_f1","test_f1","train_bal","val_bal","test_bal","train_auc","val_auc","test_auc"]
            for r in fold_results:
                print(f"Fold {r['fold']}: " + "| ".join([f"{k}: {r.get(k,0):.4f}" for k in keys]))

            # averages
            avg = {}
            for k in keys:
                avg[k] = np.mean([r.get(k, 0.0) for r in fold_results])

            print(f"\n{completed_folds}-fold averages for regime {regime}:")
            print(", ".join([f"{k}: {avg[k]:.4f}" for k in keys]))


if __name__ == "__main__":
    main()
