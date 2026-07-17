import os
import matplotlib.pyplot as plt
from BaselineLinear import DINOv3LinearProbe
from BaselineAdapters import DINOv3AdapterProbe, print_trainable_parameters



import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import monai.transforms as mt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from Datasets import ExcelDataset, FracAtlasDataset
from DataRegime import DataRegime
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v

    s = str(v).strip().lower()

    if s in ("yes", "true", "t", "y", "1"):
        return True

    if s in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []
    all_preds = []

    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
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


def save_training_curve(
    save_dir,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    train_aucs,
    val_aucs,
    train_aps=None,
    val_aps=None,
    train_f1s=None,
    val_f1s=None,
    train_bals=None,
    val_bals=None,
):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Val Loss")
    plt.title("Training Curve (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curve_loss.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accs, marker="o", label="Train Acc")
    plt.plot(epochs, val_accs, marker="o", label="Val Accuracy")
    plt.plot(epochs, train_aucs, marker="o", label="Train AUC")
    plt.plot(epochs, val_aucs, marker="o", label="Val AUC")
    plt.title("Validation Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curve_val_metrics.png")
    plt.close()

    if train_aps is not None and val_aps is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_aps, marker="o", label="Train AP")
        plt.plot(epochs, val_aps, marker="o", label="Val AP")
        plt.title("Average Precision")
        plt.xlabel("Epoch")
        plt.ylabel("AP")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curve_ap.png")
        plt.close()

    if train_f1s is not None and val_f1s is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_f1s, marker="o", label="Train F1")
        plt.plot(epochs, val_f1s, marker="o", label="Val F1")
        plt.title("F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curve_f1.png")
        plt.close()

    if train_bals is not None and val_bals is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_bals, marker="o", label="Train Balanced Acc")
        plt.plot(epochs, val_bals, marker="o", label="Val Balanced Acc")
        plt.title("Balanced Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Balanced Acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curve_balanced_acc.png")
        plt.close()


def save_confusion_matrix_plot(save_dir, y_true, y_pred, fold, title_suffix="Validation"):
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{title_suffix} Confusion Matrix (Fold {fold})")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

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

    plt.figure(figsize=(4, 3))
    plt.plot(accs, marker="o")
    plt.title("5-Fold Validation Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.savefig(f"{save_dir}/acc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(aucs, marker="o")
    plt.title("5-Fold Validation AUC")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.savefig(f"{save_dir}/auc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(aps, marker="o")
    plt.title("5-Fold Validation AP")
    plt.xlabel("Fold")
    plt.ylabel("AP")
    plt.savefig(f"{save_dir}/ap.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(f1s, marker="o")
    plt.title("5-Fold Validation F1")
    plt.xlabel("Fold")
    plt.ylabel("F1")
    plt.savefig(f"{save_dir}/f1.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(bals, marker="o")
    plt.title("5-Fold Validation Balanced Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Balanced Acc")
    plt.savefig(f"{save_dir}/balanced_acc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_accs, marker="o")
    plt.title("5-Fold Test Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Test Accuracy")
    plt.savefig(f"{save_dir}/test_acc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_aucs, marker="o")
    plt.title("5-Fold Test AUC")
    plt.xlabel("Fold")
    plt.ylabel("Test AUC")
    plt.savefig(f"{save_dir}/test_auc.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_aps, marker="o")
    plt.title("5-Fold Test AP")
    plt.xlabel("Fold")
    plt.ylabel("Test AP")
    plt.savefig(f"{save_dir}/test_ap.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_f1s, marker="o")
    plt.title("5-Fold Test F1")
    plt.xlabel("Fold")
    plt.ylabel("Test F1")
    plt.savefig(f"{save_dir}/test_f1.png")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(test_bals, marker="o")
    plt.title("5-Fold Test Balanced Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Test Balanced Acc")
    plt.savefig(f"{save_dir}/test_balanced_acc.png")
    plt.close()


def save_combined_training_curves(save_dir, histories):
    """Plot all folds' validation curves with mean and shaded std range."""
    os.makedirs(save_dir, exist_ok=True)

    n_folds = len(histories)

    if n_folds == 0:
        return

    epochs = len(histories[0]["train_losses"])
    x = np.arange(1, epochs + 1)

    def stack_metric(key):
        return np.array([h[key] for h in histories])

    metrics_to_plot = [
        ("val_losses", "Loss"),
        ("val_accs", "Accuracy"),
        ("val_aucs", "AUC"),
        ("val_aps", "AP"),
        ("val_f1s", "F1"),
        ("val_bals", "Balanced Acc"),
    ]

    for key, label in metrics_to_plot:
        try:
            data = stack_metric(key)
        except Exception:
            continue

        plt.figure(figsize=(8, 4))

        for i in range(n_folds):
            plt.plot(x, data[i], color="gray", alpha=0.3)

        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        plt.plot(x, mean, color="C0", linewidth=2, label="Mean")
        plt.fill_between(x, mean - std, mean + std, color="C0", alpha=0.2)

        plt.title(f"{label} Across Folds")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/combined_{key}.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline training with configurable paths")

    parser.add_argument(
        "--result-root",
        default="/vol/miltank/users/gayan/results/Baseline",
        help="Results root directory",
    )

    parser.add_argument(
        "--datasets-type",
        default="BTXRD",
        help="Which dataset to use: BTXRD or FracAtlas",
    )

    parser.add_argument(
        "--balanced",
        type=str2bool,
        default=True,
        help="Boolean flag. Accepts true/false, 1/0, yes/no.",
    )

    parser.add_argument(
        "--data-root",
        default="/vol/miltank/users/gayan/datasets/BTXRD/BTXRD",
        help="Dataset root directory containing dataset file and images",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and validation",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer",
    )

    parser.add_argument(
        "--model-weight-path",
        default="./models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        help="Path to DINOv3 model weights",
    )

    parser.add_argument(
        "--repo-dir",
        default="./models/dinov3",
        help="Directory for torch.hub to load DINOv3",
    )

    parser.add_argument(
        "--data-regime",
        default="100%",
        help='Data regime: 5%%, 10%%, 50%%, 100%%, or "all"',
    )

    parser.add_argument(
        "--monai",
        type=str2bool,
        default=False,
        help="Apply MONAI data augmentation to the training set only.",
    )


    parser.add_argument(
        "--model-type",
        default="linear_probe",
        choices=["linear_probe", "adapters"],
        help="Model type to train: linear_probe or adapters",
    )

    parser.add_argument(
        "--adapter-dim",
        type=int,
        default=64,
        help="Bottleneck dimension for adapter models",
    )

    parser.add_argument(
        "--adapter-dropout",
        type=float,
        default=0.1,
        help="Dropout used inside adapter modules",
    )

    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=1.0,
        help="Scale factor for the adapter residual correction",
    )

    args = parser.parse_args()

    base_result_root = os.path.abspath(os.path.expanduser(args.result_root))
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    repo_root = os.path.abspath(os.path.expanduser(args.repo_dir))
    model_weight_path = os.path.abspath(os.path.expanduser(args.model_weight_path))

    print(f"Using result root: {base_result_root}")
    print(f"Using data root: {data_root}")
    print(f"Using repo dir: {repo_root}")
    print(f"Using model weights: {model_weight_path}")
    print(f"Dataset type: {args.datasets_type}")
    print(f"Data regime: {args.data_regime}")
    print(f"Data augmentation: {'enabled with MONAI' if args.monai else 'disabled'}")
    print(f"Model type: {args.model_type}")

    if args.model_type == "adapters":
        print(
            f"Adapter dim: {args.adapter_dim}, "
            f"dropout: {args.adapter_dropout}, "
            f"scale: {args.adapter_scale}"
        )

    print(
        f"Epochs: {args.epochs}, "
        f"Batch size: {args.batch_size}, "
        f"Learning rate: {args.learning_rate}"
    )

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.datasets_type == "BTXRD":
        df_path = os.path.join(data_root, "dataset.xlsx")
        df = pd.read_excel(df_path)
        label_column = "tumor"
        print(f"Loaded BTXRD dataset with {len(df)} samples")

    elif args.datasets_type == "FracAtlas":
        df_path = os.path.join(data_root, "dataset.csv")
        df = pd.read_csv(df_path)
        label_column = "fractured"
        print(f"Loaded FracAtlas dataset with {len(df)} samples")

    else:
        print(f"Error: Unsupported dataset type {args.datasets_type}")
        return

    X = df.index.values
    y = df[label_column].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        stratify=y,
        random_state=42,
    )

    test_df = df.iloc[X_test].reset_index(drop=True)

    tfr = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_tfr = tfr
    if args.monai:
        train_tfr = mt.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),

                # MONAI augmentations
                mt.RandFlip(prob=0.5, spatial_axis=1),  # horizontal flip
                mt.RandFlip(prob=0.5, spatial_axis=0),  # vertical flip
                mt.RandRotate(range_x=0.17, prob=0.5, keep_size=True),  # ~10 degrees
                mt.RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.1, keep_size=True),
                mt.RandAdjustContrast(prob=0.5, gamma=(0.8, 1.2)),
                mt.RandGaussianNoise(prob=0.2, mean=0.0, std=0.01),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    img_dir = os.path.join(data_root, "images")

    if args.datasets_type == "BTXRD":
        test_dataset = ExcelDataset(test_df, img_dir=img_dir, transform=tfr)

    elif args.datasets_type == "FracAtlas":
        test_dataset = FracAtlasDataset(test_df, img_dir=img_dir, transform=tfr)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    regime_manager = DataRegime(X_train_val, y_train_val, args.data_regime)

    for X_tv, y_tv, regime in regime_manager.get_data():
        print(f"\n===== Running data regime: {regime} with {len(X_tv)} samples =====")

        result_root = os.path.join(base_result_root, regime)

        fold_results = []
        fold_histories = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_tv, y_tv)):
            fold_num = fold + 1
            fold_save_dir = f"{result_root}/fold_{fold_num}"

            epochs = args.epochs
            batch_size = args.batch_size
            lr = args.learning_rate

            print(f"\n========== Fold {fold_num} ==========")

            train_df = df.iloc[X_tv[train_idx]].reset_index(drop=True)
            val_df = df.iloc[X_tv[val_idx]].reset_index(drop=True)

            if args.datasets_type == "BTXRD":
                train_dataset = ExcelDataset(train_df, img_dir=img_dir, transform=train_tfr)
                val_dataset = ExcelDataset(val_df, img_dir=img_dir, transform=tfr)

            elif args.datasets_type == "FracAtlas":
                train_dataset = FracAtlasDataset(train_df, img_dir=img_dir, transform=train_tfr)
                val_dataset = FracAtlasDataset(val_df, img_dir=img_dir, transform=tfr)

            else:
                print(f"Error: Unsupported dataset type {args.datasets_type}")
                return

            print(
                f"Train dataset size: {len(train_dataset)}, "
                f"Val dataset size: {len(val_dataset)}"
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
            )

            if args.model_type == "linear_probe":
                model = DINOv3LinearProbe(
                    2,
                    repo_root,
                    model_weight_path,
                ).to(device)

            elif args.model_type == "adapters":
                model = DINOv3AdapterProbe(
                    num_classes=2,
                    REPO_DIR=repo_root,
                    weight_path=model_weight_path,
                    adapter_dim=args.adapter_dim,
                    adapter_dropout=args.adapter_dropout,
                    adapter_scale=args.adapter_scale,
                ).to(device)

            else:
                raise ValueError(f"Unsupported model type: {args.model_type}")

            print_trainable_parameters(model)

            if args.balanced:
                criterion = nn.CrossEntropyLoss()

            else:
                class_counts = train_df[label_column].value_counts().sort_index()
                class_weights = len(train_df) / (
                    2 * torch.tensor(class_counts.values, dtype=torch.float32)
                )

                print("class_counts:", class_counts.to_dict())
                print("class_weights:", class_weights.tolist())

                criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
            )

            train_losses, val_losses = [], []
            train_accs, val_accs = [], []
            train_aucs, val_aucs = [], []
            train_aps, val_aps = [], []
            train_f1s, val_f1s = [], []
            train_bals, val_bals = [], []

            last_val_labels = []
            last_val_preds = []

            for epoch in range(epochs):
                (
                    train_loss,
                    train_acc,
                    train_auc,
                    train_ap,
                    train_f1,
                    train_bal,
                    _,
                    _,
                ) = train_one_epoch(model, train_loader, optimizer, criterion, device)

                (
                    val_loss,
                    val_acc,
                    val_auc,
                    val_ap,
                    val_f1,
                    val_bal,
                    val_labels,
                    val_preds,
                    val_probs,
                ) = evaluate(model, val_loader, criterion, device)

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

                last_val_labels = val_labels
                last_val_preds = val_preds

                print(
                    f"Epoch {epoch + 1}: "
                    f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                    f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
                    f"Train AP {train_ap:.4f} | Val AP {val_ap:.4f} | "
                    f"Train F1 {train_f1:.4f} | Val F1 {val_f1:.4f} | "
                    f"Train Bal {train_bal:.4f} | Val Bal {val_bal:.4f} | "
                    f"Val AUC {val_auc:.4f}"
                )

            save_training_curve(
                fold_save_dir,
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                train_aucs,
                val_aucs,
                train_aps,
                val_aps,
                train_f1s,
                val_f1s,
                train_bals,
                val_bals,
            )

            save_confusion_matrix_plot(
                fold_save_dir,
                last_val_labels,
                last_val_preds,
                fold_num,
                "Validation",
            )

            (
                test_loss,
                test_acc,
                test_auc,
                test_ap,
                test_f1,
                test_bal,
                test_labels,
                test_preds,
                test_probs,
            ) = evaluate(model, test_loader, criterion, device)

            print(
                f"Test Results for Fold {fold_num}: "
                f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f} | "
                f"Test AUC {test_auc:.4f} | Test AP {test_ap:.4f} | "
                f"Test F1 {test_f1:.4f} | Test Bal {test_bal:.4f}"
            )

            save_confusion_matrix_plot(
                fold_save_dir,
                test_labels,
                test_preds,
                fold_num,
                "Test",
            )

            fold_results.append(
                {
                    "fold": fold_num,
                    "train_acc": train_accs[-1],
                    "val_acc": val_accs[-1],
                    "test_acc": test_acc,
                    "train_loss": train_losses[-1],
                    "val_loss": val_losses[-1],
                    "test_loss": test_loss,
                    "train_auc": train_aucs[-1],
                    "val_auc": val_aucs[-1],
                    "test_auc": test_auc,
                    "train_ap": train_aps[-1],
                    "val_ap": val_aps[-1],
                    "test_ap": test_ap,
                    "train_f1": train_f1s[-1],
                    "val_f1": val_f1s[-1],
                    "test_f1": test_f1,
                    "train_bal": train_bals[-1],
                    "val_bal": val_bals[-1],
                    "test_bal": test_bal,
                }
            )

            fold_histories.append(
                {
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
                    "val_bals": val_bals,
                }
            )

        completed_folds = len(fold_results)
        print(f"\nCompleted regime {regime}: {completed_folds} folds finished.")

        save_results(result_root, fold_results)
        save_combined_training_curves(result_root, fold_histories)

        print(f"\nPer-fold results for regime {regime}:")

        keys = [
            "train_acc",
            "val_acc",
            "test_acc",
            "train_loss",
            "val_loss",
            "test_loss",
            "train_ap",
            "val_ap",
            "test_ap",
            "train_f1",
            "val_f1",
            "test_f1",
            "train_bal",
            "val_bal",
            "test_bal",
        ]

        for r in fold_results:
            print(
                f"Fold {r['fold']}: "
                + " | ".join([f"{k}: {r.get(k, 0):.4f}" for k in keys])
            )

        avg = {}

        for k in keys:
            avg[k] = np.mean([r.get(k, 0.0) for r in fold_results])

        print(f"\n{completed_folds}-fold averages for regime {regime}:")
        print(", ".join([f"{k}: {avg[k]:.4f}" for k in keys]))


if __name__ == "__main__":
    main()
