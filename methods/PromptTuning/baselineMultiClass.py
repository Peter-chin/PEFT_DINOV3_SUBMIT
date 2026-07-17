import os
import sys
import re
import subprocess
import copy
import matplotlib.pyplot as plt
from BaselineLinear import DINOv3LinearProbe
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from Datasets import preload_dataset_to_ram_multiclass, multi_class_preprocess, MULTI_CLASS_LABELS
from DataRegime import DataRegime
import numpy as np
import monai.transforms as mt


IDX_TO_LABEL = [label for label, _ in sorted(MULTI_CLASS_LABELS.items(), key=lambda x: x[1])]

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

    # Grid search is driven by the learning-rate list passed from `main`.
    print(f"Grid search over learning rates: {lr_list}")

    # Keep grid-search results under the current result root so runs stay grouped.
    base_result_root = get_cli_arg(base_argv, '--result-root', '/vol/miltank/users/lechi/results/Baseline')
    data_regime = get_cli_arg(base_argv, '--data-regime', '100%')
    if data_regime == 'all' or ',' in str(data_regime):
        raise ValueError("grid_search currently supports only a single data regime, e.g. '100%' or '50%'.")
    grid_result_root = os.path.join(os.path.abspath(os.path.expanduser(base_result_root)), 'grid_search')
    
    for lr in lr_list:
        # Re-run this script with one learning rate at a time.
        lr_tag = str(lr).replace('.', 'p')
        run_argv = replace_cli_arg(base_argv, '--learning-rate', lr)
        run_argv = replace_cli_arg(run_argv, '--result-root', os.path.join(grid_result_root, f'lr_{lr_tag}'))
        cmd = [sys.executable, '-u', script_path] + run_argv

        print(f"\n===== Grid search run for lr={lr} =====")
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        # Stream child output live so long training steps don't look like a hang.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
        )

        output_chunks = []
        if proc.stdout is not None:
            while True:
                chunk = proc.stdout.read(1)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                output_chunks.append(chunk)

        return_code = proc.wait()
        if return_code != 0:
            print(f"Grid-search run for lr={lr} exited with code {return_code}")

        summary_csv = os.path.join(grid_result_root, f'lr_{lr_tag}', data_regime, 'lr_summary.csv')
        if os.path.exists(summary_csv):
            summary_df = pd.read_csv(summary_csv)
            mean_val_f1 = float(summary_df.loc[0, 'mean_best_val_f1'])
            mean_val_acc = float(summary_df.loc[0, 'mean_best_val_acc'])
            mean_val_loss = float(summary_df.loc[0, 'mean_best_val_loss'])
            mean_test_f1 = float(summary_df.loc[0, 'mean_test_f1_at_best_val_f1'])
            mean_test_acc = float(summary_df.loc[0, 'mean_test_acc_at_best_val_f1'])
        else:
            mean_val_f1 = None
            mean_val_acc = None
            mean_val_loss = None
            mean_test_f1 = None
            mean_test_acc = None
        
        summary_rows.append({
            'lr': lr,
            'mean_best_val_f1': mean_val_f1,
            'mean_best_val_acc': mean_val_acc,
            'mean_best_val_loss': mean_val_loss,
            'mean_test_f1_at_best_val_f1': mean_test_f1,
            'mean_test_acc_at_best_val_f1': mean_test_acc,
        })

        if mean_val_f1 is not None and mean_val_f1 > best_score:
            best_score = mean_val_f1
            best_lr = lr

    # Print a compact summary so the best lr is easy to spot in logs.
    print("\n===== Grid Search Summary =====")
    for row in summary_rows:
        print(
            f"lr={row['lr']} | "
            f"mean_best_val_f1={row['mean_best_val_f1']} | "
            f"mean_best_val_acc={row['mean_best_val_acc']} | "
            f"mean_best_val_loss={row['mean_best_val_loss']} | "
            f"mean_test_f1_at_best_val_f1={row['mean_test_f1_at_best_val_f1']} | "
            f"mean_test_acc_at_best_val_f1={row['mean_test_acc_at_best_val_f1']}"
        )

    if best_lr is not None:
        print(f"Best learning rate: {best_lr} (mean_best_val_f1={best_score})")
    else:
        print("Best learning rate could not be determined from the run output.")
    
    print("\n===== Best Fold/Epoch per Learning Rate =====")
    for row in summary_rows:
        lr = row['lr']
        lr_tag = str(lr).replace('.', 'p')
        fold_csv = os.path.join(
            grid_result_root,
            f'lr_{lr_tag}',
            data_regime,
            'fold_best_epoch_summary.csv'
        )

        if os.path.exists(fold_csv):
            fold_df = pd.read_csv(fold_csv)
            if len(fold_df) > 0 and 'best_val_f1' in fold_df.columns:
                best_idx = int(fold_df['best_val_f1'].astype(float).idxmax())
                best_row = fold_df.loc[best_idx]
                print(
                    f"lr={lr} | "
                    f"best_fold_by_val_f1={int(best_row['fold'])} | "
                    f"best_epoch_in_that_fold={int(best_row['best_epoch_by_val_f1'])} | "
                    f"best_val_f1={float(best_row['best_val_f1']):.4f}"
                )
            else:
                print(f"lr={lr} | fold_best_epoch_summary.csv is empty or missing best_val_f1 column")
        else:
            print(f"lr={lr} | fold_best_epoch_summary.csv not found: {fold_csv}")
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


def should_run_grid_search(learning_rate_list):
    return len(learning_rate_list) > 1

def train_one_epoch(model, loader, optimizer, criterion, device, desc=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []
    all_preds = []

    for x, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=10):
        x = x.to(device)
        y = y.to(device).long()

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.softmax(out, dim=1)
        preds = torch.argmax(out, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        all_probs.append(probs.detach().cpu())
        all_labels.extend(y.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    acc = correct / total if total > 0 else 0.0

    try:
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    except Exception:
        f1 = 0.0

    try:
        bal = balanced_accuracy_score(all_labels, all_preds)
    except Exception:
        bal = 0.0

    return total_loss / len(loader), acc, f1, bal, all_labels, all_preds, all_probs

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()

            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_probs.append(probs.detach().cpu())
            all_labels.extend(y.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    acc = correct / total if total > 0 else 0.0

    try:
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    except Exception:
        f1 = 0.0

    try:
        bal = balanced_accuracy_score(all_labels, all_preds)
    except Exception:
        bal = 0.0

    return total_loss / len(loader), acc, f1, bal, all_labels, all_preds, all_probs

def save_training_curve(save_dir, train_losses, val_losses, train_accs, val_accs, train_f1s=None, val_f1s=None, train_bals=None, val_bals=None):
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
    plt.title("Validation Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curve_val_metrics.png")
    plt.close()

    # optional plots for AP, F1, Balanced Accuracy
    

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

def save_multiclass_confusion_matrix(save_dir, y_true, y_pred, fold, label_names, title_suffix="Validation", normalize=False):
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(np.float32)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row_sum, 1), where=row_sum != 0)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"{title_suffix} Confusion Matrix (Fold {fold})")
    plt.colorbar()
    plt.xticks(range(len(label_names)), label_names, rotation=45, ha='right')
    plt.yticks(range(len(label_names)), label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{title_suffix.lower().replace(' ', '_')}_confusion_matrix.png", dpi=200)
    plt.close()

def save_results(save_dir, fold_results):
    os.makedirs(save_dir, exist_ok=True)

    accs = [r["val_acc"] for r in fold_results]
    f1s = [r.get("val_f1", 0.0) for r in fold_results]
    bals = [r.get("val_bal", 0.0) for r in fold_results]
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


def save_epoch_metrics_csv(save_dir, epoch_metrics):
    os.makedirs(save_dir, exist_ok=True)
    if not epoch_metrics:
        return None

    df = pd.DataFrame(epoch_metrics)
    csv_path = os.path.join(save_dir, "epoch_val_metrics.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def save_best_epoch_summary_csv(save_dir, fold_results):
    os.makedirs(save_dir, exist_ok=True)
    if not fold_results:
        return None, None

    per_fold_df = pd.DataFrame([
        {
            "fold": r.get("fold", idx + 1),
            "best_epoch_by_val_f1": r.get("best_epoch_by_val_f1", 1),
            "best_val_f1": r.get("best_val_f1", 0.0),
            "best_epoch_by_val_acc": r.get("best_epoch_by_val_acc", 1),
            "best_val_acc": r.get("best_val_acc", 0.0),
            "best_epoch_by_val_loss": r.get("best_epoch_by_val_loss", 1),
            "best_val_loss": r.get("best_val_loss", 0.0),
            "test_acc_at_best_val_f1": r.get("test_acc_at_best_val_f1", 0.0),
            "test_loss_at_best_val_f1": r.get("test_loss_at_best_val_f1", 0.0),
            "test_f1_at_best_val_f1": r.get("test_f1_at_best_val_f1", 0.0),
            "test_bal_at_best_val_f1": r.get("test_bal_at_best_val_f1", 0.0),
        }
        for idx, r in enumerate(fold_results)
    ])

    fold_csv_path = os.path.join(save_dir, "fold_best_epoch_summary.csv")
    per_fold_df.to_csv(fold_csv_path, index=False)

    summary_row = {
        "num_folds": len(per_fold_df),
        "mean_best_val_f1": float(per_fold_df["best_val_f1"].mean()),
        "mean_best_val_acc": float(per_fold_df["best_val_acc"].mean()),
        "mean_best_val_loss": float(per_fold_df["best_val_loss"].mean()),
        "mean_test_acc_at_best_val_f1": float(per_fold_df["test_acc_at_best_val_f1"].mean()),
        "mean_test_loss_at_best_val_f1": float(per_fold_df["test_loss_at_best_val_f1"].mean()),
        "mean_test_f1_at_best_val_f1": float(per_fold_df["test_f1_at_best_val_f1"].mean()),
        "mean_test_bal_at_best_val_f1": float(per_fold_df["test_bal_at_best_val_f1"].mean()),
    }
    summary_df = pd.DataFrame([summary_row])
    summary_csv_path = os.path.join(save_dir, "lr_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    return fold_csv_path, summary_csv_path


def main():
    parser = argparse.ArgumentParser(description='Baseline training with configurable paths')
    parser.add_argument('--result-root', default='/vol/miltank/users/gayan/results/Baseline', help='Results root directory (can be relative)')
    parser.add_argument('--datasets-type', default='BTXRD', help='Which dataset to use: BTXRD or FracAtlas')
    parser.add_argument('--balanced', type=str2bool, default=True, help='Boolean flag. Accepts true/false, 1/0, yes/no.')
    parser.add_argument('--data-root', default='/vol/miltank/users/gayan/datasets/BTXRD/BTXRD', help='Dataset root directory containing dataset.xlsx and images (can be relative)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--learning-rate', default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--model-weight-path', default='./models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', help='Path to DINOv3 model weights (can be relative)')
    parser.add_argument('--repo-dir', default='./models/dinov3', help='Directory for torch.hub to load DINOv3 (can be relative)')
    parser.add_argument('--data-regime', default='100%', help='Data regime: 5%, 10%, 50%, or 100% (can also be "all" or a comma-separated list like "5%,10%")')
    parser.add_argument('--cache-dataset', type=str2bool, default=True, help='Cache decoded images in memory so each sample is loaded once per run.')
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
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")

    if should_run_grid_search(learning_rate_list):
        print(f"Running grid search over {len(learning_rate_list)} learning rates: {learning_rate_list}")
        grid_search(learning_rate_list)
        return

    args.learning_rate = float(learning_rate_list[0])

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


    processed_labels = multi_class_preprocess(df, datasets_type=args.datasets_type)
    valid_mask = processed_labels >= 0

    df = df.loc[valid_mask].reset_index(drop=True)
    y = processed_labels[valid_mask]
    X = df.index.values
    num_classes = len(IDX_TO_LABEL)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    test_df = df.iloc[X_test].reset_index(drop=True)
    # print(test_df[:5])
    print(f"training with {num_classes} classes")
    
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
    # tfr = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),

    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]), 
    # ])

    # implement Monai
    train_tfr = mt.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        mt.RandFlip(prob=0.5, spatial_axis=1),  # Horizontal flip
        mt.RandFlip(prob=0.5, spatial_axis=0),  # Vertical flip
        mt.RandRotate(range_x=0.17, prob=0.5, keep_size=True),  # ~10 degrees
        mt.RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.1, keep_size=True),
        mt.RandAdjustContrast(prob=0.5, gamma=(0.8, 1.2)),
        mt.RandGaussianNoise(prob=0.2, mean=0.0, std=0.01),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tfr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_dir = os.path.join(data_root, 'images')

    # Create test dataset and loader (test split is fixed above with random_state=42)
    if args.datasets_type == "BTXRD":
        test_dataset, test_df = preload_dataset_to_ram_multiclass(
            test_df,
            img_dir=img_dir,
            datasets_type=args.datasets_type,
            transform=eval_tfr
        )
    elif args.datasets_type == "FracAtlas":
        raise ValueError(f"Unsupported dataset type: {args.datasets_type}")
       
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    regime_manager = DataRegime(X_train_val, y_train_val, args.data_regime)

    # Per-regime loop: perform subsampling (if any) then run 5-fold for that regime
    for X_tv, y_tv, regime in regime_manager.get_data():
        print(f"\n===== Running data regime: {regime} with {len(X_tv)} samples =====")

        regime_df_input = df.iloc[X_tv].reset_index(drop=True)
        if args.datasets_type == "BTXRD":
            regime_train_dataset, regime_train_df = preload_dataset_to_ram_multiclass(
                regime_df_input,
                img_dir=img_dir,
                datasets_type=args.datasets_type,
                transform=train_tfr
            )

            regime_val_dataset, regime_val_df = preload_dataset_to_ram_multiclass(
                regime_df_input,
                img_dir=img_dir,
                datasets_type=args.datasets_type,
                transform=eval_tfr
            )
        else:
            print(f"Error: Unsupported dataset type {args.datasets_type}")
            return

        regime_df = regime_train_df.reset_index(drop=True)
        regime_y = multi_class_preprocess(regime_df, datasets_type=args.datasets_type)
        valid_regime_mask = regime_y >= 0
        regime_df = regime_df.loc[valid_regime_mask].reset_index(drop=True)
        regime_y = regime_y[valid_regime_mask]

        print(f"number of classes in regime {regime}: {num_classes}")
        
        # per-regime result folder
        result_root = os.path.join(base_result_root, regime)

        # reset per-regime fold accumulators
        fold_results = []
        fold_histories = []

        # regime_indices = np.arange(len(regime_dataset))
        regime_indices = np.arange(len(regime_train_dataset))

        for fold, (train_idx, val_idx) in enumerate(kf.split(regime_indices, regime_y)):
            fold_num = fold + 1
            fold_save_dir = f"{result_root}/fold_{fold_num}"
            epochs = args.epochs
            batch_size = args.batch_size
            lr = args.learning_rate
            
            print(f"\n========== Fold {fold_num} ==========")

            train_df = regime_df.iloc[train_idx].reset_index(drop=True)
            val_df = regime_df.iloc[val_idx].reset_index(drop=True)

            # train_dataset = Subset(regime_dataset, train_idx)
            # val_dataset = Subset(regime_dataset, val_idx)
            train_dataset = Subset(regime_train_dataset, train_idx)
            val_dataset = Subset(regime_val_dataset, val_idx)
            print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=False,
            )

            model = DINOv3LinearProbe(num_classes, repo_root, model_weight_path).to(device)
            # 印出 parameter 數量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")

            if args.balanced:
                criterion = nn.CrossEntropyLoss()
            else:
                train_labels_for_weights = multi_class_preprocess(train_df, datasets_type=args.datasets_type)
                train_labels_for_weights = train_labels_for_weights[train_labels_for_weights >= 0]

                class_counts = np.bincount(train_labels_for_weights, minlength=num_classes).astype(np.float32)
                class_weights = len(train_labels_for_weights) / np.clip(num_classes * class_counts, 1.0, None)
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

                print("class_counts:", class_counts.tolist())
                print("class_weights:", class_weights.tolist())

                criterion = nn.CrossEntropyLoss(weight=class_weights)

            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
            train_losses, val_losses = [], []
            train_accs, val_accs = [], []
            train_f1s, val_f1s = [], []
            train_bals, val_bals = [], []

            
            # store per-epoch validation predictions so we can pick best epoch later
            val_labels_per_epoch = []
            val_preds_per_epoch = []
            val_probs_per_epoch = []

            last_val_labels = []
            last_val_preds = []

            epoch_val_metrics = []
            best_val_acc = float('-inf')
            best_val_f1 = float('-inf')
            best_val_loss = float('inf')
            best_epoch_by_val_acc = -1
            best_epoch_by_val_f1 = -1
            best_epoch_by_val_loss = -1
            best_state_by_acc = None
            best_state_by_f1 = None
            best_state_by_loss = None

            for epoch in range(epochs):
                train_loss, train_acc, train_f1, train_bal, _, _, _ = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    desc=f"Epoch {epoch+1}/{epochs}"
                )
                val_loss, val_acc, val_f1, val_bal, val_labels, val_preds, val_probs = evaluate(model, val_loader, criterion, device)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                train_accs.append(train_acc)
                val_accs.append(val_acc)

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

                epoch_val_metrics.append({
                    "epoch": epoch + 1,
                    "val_f1": float(val_f1),
                    "val_acc": float(val_acc),
                    "val_loss": float(val_loss),
                })

                try:
                    cur_val_acc = float(val_acc)
                    cur_val_loss = float(val_loss)
                    cur_val_f1 = float(val_f1)
                except Exception:
                    cur_val_f1 = float('-inf')
                    cur_val_acc = float('-inf')
                    cur_val_loss = float('inf')

                if cur_val_f1 > best_val_f1:
                    best_val_f1 = cur_val_f1
                    best_epoch_by_val_f1 = epoch
                    best_state_by_f1 = copy.deepcopy(model.state_dict())

                if cur_val_acc > best_val_acc:
                    best_val_acc = cur_val_acc
                    best_epoch_by_val_acc = epoch
                    best_state_by_acc = copy.deepcopy(model.state_dict())

                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_epoch_by_val_loss = epoch
                    best_state_by_loss = copy.deepcopy(model.state_dict())

                tqdm.write(
                    f"Epoch {epoch+1}: "
                    f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                    f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
                    f"Train F1 {train_f1:.4f} | Val F1 {val_f1:.4f} | "
                    f"Train Bal {train_bal:.4f} | Val Bal {val_bal:.4f} | "
                )

            save_epoch_metrics_csv(fold_save_dir, epoch_val_metrics)

            # save per-fold curves
            save_training_curve(fold_save_dir, train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, train_bals, val_bals)

            if best_epoch_by_val_f1 >= 0:
                best_idx_by_val_f1 = best_epoch_by_val_f1
            else:
                try:
                    best_idx_by_val_f1 = int(np.nanargmax(np.array(val_f1s)))
                except Exception:
                    best_idx_by_val_f1 = len(val_f1s) - 1 if len(val_f1s) > 0 else 0

            if best_epoch_by_val_acc >= 0:
                best_idx_by_val_acc = best_epoch_by_val_acc
            else:
                try:
                    best_idx_by_val_acc = int(np.nanargmax(np.array(val_accs)))
                except Exception:
                    best_idx_by_val_acc = len(val_accs) - 1 if len(val_accs) > 0 else 0

            if best_epoch_by_val_loss >= 0:
                best_idx_by_val_loss = best_epoch_by_val_loss
            else:
                try:
                    best_idx_by_val_loss = int(np.nanargmin(np.array(val_losses)))
                except Exception:
                    best_idx_by_val_loss = len(val_losses) - 1 if len(val_losses) > 0 else 0

            try:
                best_val_labels = val_labels_per_epoch[best_idx_by_val_f1]
                best_val_preds = val_preds_per_epoch[best_idx_by_val_f1]
            except Exception:
                best_val_labels = last_val_labels
                best_val_preds = last_val_preds

            # save_confusion_matrix_plot(fold_save_dir, best_val_labels, best_val_preds, fold_num, "Validation")
            save_multiclass_confusion_matrix(
                fold_save_dir,
                best_val_labels,
                best_val_preds,
                fold_num,
                IDX_TO_LABEL,
                title_suffix="Validation",
                normalize=False
            )

            if best_state_by_f1 is not None:
                model.load_state_dict(best_state_by_f1)

            test_loss, test_acc, test_f1, test_bal, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion, device)

            print(f"Test Results for Fold {fold_num} (best-epoch model): "
                f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f} | Test F1 {test_f1:.4f} | Test Bal {test_bal:.4f}")

           
            save_multiclass_confusion_matrix(
                fold_save_dir,
                test_labels,
                test_preds,
                fold_num,
                IDX_TO_LABEL,
                title_suffix="Test",
                normalize=True
            )


            # record final metrics for this fold (use best_idx)
            def safe_get(lst, idx, default=0.0):
               try:
                   return lst[idx]
               except Exception:
                   return default

            best_fold_val_f1 = safe_get(val_f1s, best_idx_by_val_f1, 0.0)
            best_fold_val_acc = safe_get(val_accs, best_idx_by_val_acc, 0.0)
            best_fold_val_loss = safe_get(val_losses, best_idx_by_val_loss, 0.0)
            best_fold_val_bal = safe_get(val_bals, best_idx_by_val_f1, 0.0)

            fold_results.append({
                "fold": fold_num,

                "best_epoch_by_val_f1": best_idx_by_val_f1 + 1,
                "best_val_f1": safe_get(val_f1s, best_idx_by_val_f1, 0.0),

                "best_epoch_by_val_acc": best_idx_by_val_acc + 1,
                "best_val_acc": safe_get(val_accs, best_idx_by_val_acc, 0.0),

                "best_epoch_by_val_loss": best_idx_by_val_loss + 1,
                "best_val_loss": safe_get(val_losses, best_idx_by_val_loss, 0.0),

                "train_acc": safe_get(train_accs, best_idx_by_val_f1, 0.0),
                "val_acc": safe_get(val_accs, best_idx_by_val_f1, 0.0),
                "test_acc": test_acc,

                "train_loss": safe_get(train_losses, best_idx_by_val_f1, 0.0),
                "val_loss": safe_get(val_losses, best_idx_by_val_f1, 0.0),
                "test_loss": test_loss,

                "train_f1": safe_get(train_f1s, best_idx_by_val_f1, 0.0),
                "val_f1": safe_get(val_f1s, best_idx_by_val_f1, 0.0),
                "test_f1": test_f1,

                "train_bal": safe_get(train_bals, best_idx_by_val_f1, 0.0),
                "val_bal": safe_get(val_bals, best_idx_by_val_f1, 0.0),
                "test_bal": test_bal,

                "test_acc_at_best_val_f1": test_acc,
                "test_loss_at_best_val_f1": test_loss,
                "test_f1_at_best_val_f1": test_f1,
                "test_bal_at_best_val_f1": test_bal,
            })

            fold_histories.append({
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
                "train_f1s": train_f1s,
                "val_f1s": val_f1s,
                "train_bals": train_bals,
                "val_bals": val_bals
            })

        completed_folds = len(fold_results)
        # print(f"\nCompleted regime {regime} for {n_prompt_tokens} prompt tokens: {completed_folds} folds finished.")

        # Generate 5-fold results only AFTER all folds are complete
        save_results(result_root, fold_results)
        save_combined_training_curves(result_root, fold_histories)
        fold_csv_path, summary_csv_path = save_best_epoch_summary_csv(result_root, fold_results)
        if fold_csv_path is not None:
            print(f"Saved per-fold best metrics to: {fold_csv_path}")
        if summary_csv_path is not None:
            print(f"Saved summary metrics to: {summary_csv_path}")
        if fold_results:
            avg_best_val_f1 = float(np.mean([r.get("best_val_f1", 0.0) for r in fold_results]))
            avg_best_val_acc = float(np.mean([r.get("best_val_acc", 0.0) for r in fold_results]))
            avg_best_val_loss = float(np.mean([r.get("best_val_loss", 0.0) for r in fold_results]))
            avg_test_acc = float(np.mean([r.get("test_acc_at_best_val_f1", 0.0) for r in fold_results]))
            print(
                f"Mean best val f1: {avg_best_val_f1:.4f} | "
                f"Mean best val acc: {avg_best_val_acc:.4f} | "
                f"Mean best val loss: {avg_best_val_loss:.4f} | "
                f"Mean test acc at best val f1: {avg_test_acc:.4f}"
            )
        # print per-fold metrics and averages for this regime
        print(f"\nPer-fold results for regime {regime}:")
        keys = [
            "train_acc","val_acc","test_acc",
            "train_loss","val_loss","test_loss",
            "train_f1","val_f1","test_f1",
            "train_bal","val_bal","test_bal"
        ]

        for r in fold_results:
            print(
                f"Fold {r['fold']}: "
                f"best_epoch_by_val_f1={r.get('best_epoch_by_val_f1', 0)} | "
                f"best_val_f1={r.get('best_val_f1', 0.0):.4f} | "
                f"best_epoch_by_val_acc={r.get('best_epoch_by_val_acc', 0)} | "
                f"best_val_acc={r.get('best_val_acc', 0.0):.4f} | "
                f"best_epoch_by_val_loss={r.get('best_epoch_by_val_loss', 0)} | "
                f"best_val_loss={r.get('best_val_loss', 0.0):.4f} | "
                + " | ".join([f"{k}: {r.get(k, 0.0):.4f}" for k in keys])
            )
        # averages
        avg = {}
        for k in keys:
            avg[k] = np.mean([r.get(k, 0.0) for r in fold_results])

        avg_best_epoch_by_val_f1 = np.mean([r.get("best_epoch_by_val_f1", 0) for r in fold_results])
        avg_best_val_f1 = np.mean([r.get("best_val_f1", 0.0) for r in fold_results])
        avg_best_epoch_by_val_acc = np.mean([r.get("best_epoch_by_val_acc", 0) for r in fold_results])
        avg_best_val_acc = np.mean([r.get("best_val_acc", 0.0) for r in fold_results])
        avg_best_epoch_by_val_loss = np.mean([r.get("best_epoch_by_val_loss", 0) for r in fold_results])
        avg_best_val_loss = np.mean([r.get("best_val_loss", 0.0) for r in fold_results])

        print(f"\n{completed_folds}-fold averages for regime {regime}:")
        print(
            f"best_epoch_by_val_f1: {avg_best_epoch_by_val_f1:.2f}, "
            f"best_val_f1: {avg_best_val_f1:.4f}, "
            f"best_epoch_by_val_acc: {avg_best_epoch_by_val_acc:.2f}, "
            f"best_val_acc: {avg_best_val_acc:.4f}, "
            f"best_epoch_by_val_loss: {avg_best_epoch_by_val_loss:.2f}, "
            f"best_val_loss: {avg_best_val_loss:.4f}, "
            + ", ".join([f"{k}: {avg[k]:.4f}" for k in keys])
        )

if __name__ == "__main__":
    main()
