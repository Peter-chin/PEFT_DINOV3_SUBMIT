import os
import sys
import re
import subprocess
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
import monai.transforms as mt

from Baseline import (
    str2bool,
    train_one_epoch,
    evaluate,
    save_training_curve,
    save_confusion_matrix_plot,
    save_results,
    save_combined_training_curves,
    save_epoch_metrics_csv,
    save_best_epoch_summary_csv,
)
from BaselineLinear import DINOv3PromptProbe
from BaselineLinear import DeepPromptClassificationHead
from Datasets import ExcelDataset, FracAtlasDataset
from DataRegime import DataRegime
from typing import Optional


def grid_search(lr_list: list):
    """Run the prompt-tuning script repeatedly for each learning rate and report the best one."""
    script_path = os.path.abspath(__file__)
    base_argv = sys.argv[1:]
    best_lr = None
    best_score = float('-inf')
    summary_rows = []

    def get_cli_arg(argv, flag, default=None):
        for i in range(len(argv) - 1):
            if argv[i] == flag:
                return argv[i + 1]
        return default

    def replace_cli_arg(argv, flag, value):
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

    print(f"Grid search over learning rates: {lr_list}")

    base_result_root = get_cli_arg(base_argv, '--result-root', '/vol/miltank/users/gayan/results/Baseline_with_prompt')
    data_regime = get_cli_arg(base_argv, '--data-regime', '100%')
    if data_regime == 'all' or ',' in str(data_regime):
        raise ValueError("grid_search currently supports only a single data regime, e.g. '100%' or '50%'.")

    grid_result_root = os.path.join(os.path.abspath(os.path.expanduser(base_result_root)), 'grid_search')

    for lr in lr_list:
        lr_tag = str(lr).replace('.', 'p')
        run_argv = replace_cli_arg(base_argv, '--learning-rate', lr)
        run_argv = replace_cli_arg(run_argv, '--result-root', os.path.join(grid_result_root, f'lr_{lr_tag}'))
        cmd = [sys.executable, '-u', script_path] + run_argv

        print(f"\n===== Grid search run for lr={lr} =====")
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

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
            mean_val_auc = float(summary_df.loc[0, 'mean_best_val_auc'])
            mean_val_acc = float(summary_df.loc[0, 'mean_best_val_acc'])
            mean_val_loss = float(summary_df.loc[0, 'mean_best_val_loss'])
            mean_test_auc = float(summary_df.loc[0, 'mean_test_auc_at_best_val_auc'])
            mean_test_acc = float(summary_df.loc[0, 'mean_test_acc_at_best_val_auc'])
        else:
            mean_val_auc = None
            mean_val_acc = None
            mean_val_loss = None
            mean_test_auc = None
            mean_test_acc = None

        summary_rows.append({
            'lr': lr,
            'mean_best_val_auc': mean_val_auc,
            'mean_best_val_acc': mean_val_acc,
            'mean_best_val_loss': mean_val_loss,
            'mean_test_auc_at_best_val_auc': mean_test_auc,
            'mean_test_acc_at_best_val_auc': mean_test_acc,
        })

        if mean_val_auc is not None and mean_val_auc > best_score:
            best_score = mean_val_auc
            best_lr = lr

    print("\n===== Grid Search Summary =====")
    for row in summary_rows:
        print(
            f"lr={row['lr']} | "
            f"mean_best_val_auc={row['mean_best_val_auc']} | "
            f"mean_best_val_acc={row['mean_best_val_acc']} | "
            f"mean_best_val_loss={row['mean_best_val_loss']} | "
            f"mean_test_auc_at_best_val_auc={row['mean_test_auc_at_best_val_auc']} | "
            f"mean_test_acc_at_best_val_auc={row['mean_test_acc_at_best_val_auc']}"
        )

    if best_lr is not None:
        print(f"Best learning rate: {best_lr} (mean_best_val_auc={best_score})")
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
            if len(fold_df) > 0 and 'best_val_auc' in fold_df.columns:
                best_idx = int(fold_df['best_val_auc'].astype(float).idxmax())
                best_row = fold_df.loc[best_idx]
                print(
                    f"lr={lr} | "
                    f"best_fold_by_val_auc={int(best_row['fold'])} | "
                    f"best_epoch_in_that_fold={int(best_row['best_epoch_by_val_auc'])} | "
                    f"best_val_auc={float(best_row['best_val_auc']):.4f}"
                )
            else:
                print(f"lr={lr} | fold_best_epoch_summary.csv is empty or missing best_val_auc column")
        else:
            print(f"lr={lr} | fold_best_epoch_summary.csv not found: {fold_csv}")

    return best_lr

def should_run_grid_search(learning_rate_list, data_regime):
    return len(learning_rate_list) > 1 and str(data_regime).strip() == '100%'

def main():
    parser = argparse.ArgumentParser(description='Baseline training with shallow prompt tuning')
    parser.add_argument('--result-root', default='/vol/miltank/users/gayan/results/Baseline_with_prompt', help='Results root directory (can be relative)')
    parser.add_argument('--datasets-type', default='BTXRD', help='Which dataset to use: BTXRD or FracAtlas')
    parser.add_argument('--balanced', type=str2bool, default=True, help='Boolean flag. Accepts true/false, 1/0, yes/no.')
    parser.add_argument('--data-root', default='/vol/miltank/users/gayan/datasets/BTXRD/BTXRD', help='Dataset root directory containing dataset.xlsx and images (can be relative)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--learning-rate', default='1e-3', help='Learning rate for optimizer, or a comma-separated sweep such as "1e-4,3e-4,1e-3"')
    parser.add_argument('--model-weight-path', default='./models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', help='Path to DINOv3 model weights (can be relative)')
    parser.add_argument('--repo-dir', default='./models/dinov3', help='Directory for torch.hub to load DINOv3 (can be relative)')
    # PROMPT-TOKEN EXTENSION: change this to a single value like 4 if you do not want a sweep.
    parser.add_argument('--data-regime', default='100%', help='Data regime: 5%, 10%, 50%, or 100% (can also be "all" or a comma-separated list like "5%,10%")')
    parser.add_argument('--n-prompt-tokens', default='1,4,8,16', help='Prompt token sweep, e.g. "1,4,8,16" or a single value like "4"')
    parser.add_argument('--cache-dataset', type=str2bool, default=True, help='Cache decoded images in memory so each sample is loaded once per run.')
    parser.add_argument('--use-mixup', type=str2bool, default=False, help='Apply MixUp augmentation to training batches.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Beta distribution alpha for MixUp augmentation.')
    parser.add_argument('--monai', type=str2bool, default=False)
    args = parser.parse_args()

    base_result_root = os.path.abspath(os.path.expanduser(args.result_root))
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    repo_root = os.path.abspath(os.path.expanduser(args.repo_dir))
    model_weight_path = os.path.abspath(os.path.expanduser(args.model_weight_path))

    print(f"Using result root: {base_result_root}")
    print(f"Using data root:   {data_root}")
    print(f"Using repo dir:    {repo_root}")
    print(f"Using model weights: {model_weight_path}")
    print(f"Dataset type: {args.datasets_type}")
    print(f"Data regime: {args.data_regime}")
    print(f"Use MixUp: {args.use_mixup}, MixUp alpha: {args.mixup_alpha}")
    learning_rate_list = [v.strip() for v in str(args.learning_rate).split(',') if v.strip()]
    if len(learning_rate_list) == 0:
        learning_rate_list = ['1e-3']

    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    print("Prompt tuning: enabled for independent DINOv3 prompt tokens")

    if should_run_grid_search(learning_rate_list, args.data_regime):
        print(f"Running grid search over {len(learning_rate_list)} learning rates: {learning_rate_list}")
        grid_search(learning_rate_list)
        return

    if len(learning_rate_list) > 1 and str(args.data_regime).strip() != '100%':
        print("Error: multiple learning rates are only allowed when --data-regime 100% is used for grid search.")
        print("Please run grid search on 100% first, then rerun other regimes with a single fixed learning rate.")
        return

    args.learning_rate = float(learning_rate_list[0])

    if ',' in str(args.n_prompt_tokens):
        prompt_token_list = [int(v.strip()) for v in str(args.n_prompt_tokens).split(',') if v.strip()]
    else:
        prompt_token_list = [int(args.n_prompt_tokens)]

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

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

    X = df.index.values
    y = df['tumor'].values if args.datasets_type == "BTXRD" else df['fractured'].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        stratify=y,
        random_state=42,
    )
    test_df = df.iloc[X_test].reset_index(drop=True)

    available_regimes = ['5%', '10%', '50%', '100%']
    if args.data_regime == 'all':
        regimes_list = available_regimes
    elif ',' in args.data_regime:
        regimes_list = [r.strip() for r in args.data_regime.split(',') if r.strip()]
    else:
        regimes_list = [args.data_regime]

    # tfr = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    eval_tfr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_tfr = eval_tfr
    if args.monai:
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

    

    img_dir = os.path.join(data_root, 'images')

    if args.datasets_type == "BTXRD":
        test_dataset = ExcelDataset(test_df, img_dir=img_dir, transform=eval_tfr, cache_images=args.cache_dataset)
    else:
        test_dataset = FracAtlasDataset(test_df, img_dir=img_dir, transform=eval_tfr, cache_images=args.cache_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    regime_manager = DataRegime(X_train_val, y_train_val, args.data_regime)

    for n_prompt_tokens in prompt_token_list:
        # PROMPT-TOKEN EXTENSION: remove the prompt_{n_prompt_tokens} folder if you want the old single-run layout.
        print(f"\n===== Running prompt-token setting: {n_prompt_tokens} =====")

        for X_tv, y_tv, regime in regime_manager.get_data():
            print(f"\n===== Running data regime: {regime} with {len(X_tv)} samples =====")
            regime_df_input = df.iloc[X_tv].reset_index(drop=True)
            if args.datasets_type == "BTXRD":
                # regime_dataset = ExcelDataset(
                #     df.iloc[X_tv].reset_index(drop=True),
                #     img_dir=img_dir,
                #     transform=tfr,
                #     cache_images=args.cache_dataset,
                # )
                regime_train_dataset = ExcelDataset(
                    regime_df_input,
                    img_dir=img_dir,
                    transform=train_tfr,
                    cache_images=args.cache_dataset,
                )
                regime_val_dataset = ExcelDataset(
                    regime_df_input,
                    img_dir=img_dir,
                    transform=eval_tfr,
                    cache_images=args.cache_dataset,
                )
            else:
                # regime_dataset = FracAtlasDataset(
                #     df.iloc[X_tv].reset_index(drop=True),
                #     img_dir=img_dir,
                #     transform=tfr,
                #     cache_images=args.cache_dataset,
                # )
                regime_train_dataset = FracAtlasDataset(
                    regime_df_input,
                    img_dir=img_dir,
                    transform=train_tfr,
                    cache_images=args.cache_dataset,
                )
                regime_val_dataset = FracAtlasDataset(
                    regime_df_input,
                    img_dir=img_dir,
                    transform=eval_tfr,
                    cache_images=args.cache_dataset,
                )

            # regime_df = regime_dataset.df.reset_index(drop=True)
            regime_df = regime_df_input.reset_index(drop=True)
            label_col = "tumor" if args.datasets_type == "BTXRD" else "fractured"
            regime_y = regime_df[label_col].values

            result_root = os.path.join(base_result_root, f"prompt_{n_prompt_tokens}", regime)
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
                    num_workers=2,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=False,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=False,
                )

                # deep prompt
                model = DeepPromptClassificationHead(
                    2,
                    repo_root,
                    model_weight_path,
                    n_prompt_tokens=n_prompt_tokens,
                ).to(device)

                # 原版prompt tuning
                # model = DINOv3PromptProbe(2, repo_root, model_weight_path, n_prompt_tokens=n_prompt_tokens).to(device)
                
                
                # 印出 parameter 數量
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"\nTotal parameters: {total_params}")
                print(f"Trainable parameters: {trainable_params}")

                if args.balanced:
                    criterion = nn.CrossEntropyLoss()
                else:
                    class_counts = train_df[label_col].value_counts().sort_index()
                    class_weights = len(train_df) / (2 * torch.tensor(class_counts.values, dtype=torch.float32))
                    print("class_counts:", class_counts.to_dict())
                    print("class_weights:", class_weights.tolist())
                    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

                optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

                train_losses, val_losses = [], []
                train_accs, val_accs = [], []
                train_aucs, val_aucs = [], []
                train_aps, val_aps = [], []
                train_f1s, val_f1s = [], []
                train_bals, val_bals = [], []

                
                val_labels_per_epoch = []
                val_preds_per_epoch = []
                val_probs_per_epoch = []

                last_val_labels = []
                last_val_preds = []

                epoch_val_metrics = []
                best_val_acc = float('-inf')
                best_val_auc = float('-inf')
                best_val_loss = float('inf')
                best_epoch_by_val_auc = -1
                best_epoch_by_val_acc = -1
                best_epoch_by_val_loss = -1
                best_state_by_auc = None
                best_state_by_acc = None
                best_state_by_loss = None

                for epoch in range(epochs):
                    train_loss, train_acc, train_auc, train_ap, train_f1, train_bal, _, _ = train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        criterion,
                        device,
                        use_mixup=args.use_mixup,
                        mixup_alpha=args.mixup_alpha,
                    )
                    val_loss, val_acc, val_auc, val_ap, val_f1, val_bal, val_labels, val_preds, val_probs = evaluate(
                        model, val_loader, criterion, device
                    )

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

                    val_labels_per_epoch.append(val_labels)
                    val_preds_per_epoch.append(val_preds)
                    val_probs_per_epoch.append(val_probs)

                    last_val_labels = val_labels
                    last_val_preds = val_preds

                    epoch_val_metrics.append({
                        "epoch": epoch + 1,
                        "val_auc": float(val_auc),
                        "val_acc": float(val_acc),
                        "val_loss": float(val_loss),
                    })

                    try:
                        cur_val_auc = float(val_auc)
                        cur_val_acc = float(val_acc)
                        cur_val_loss = float(val_loss)
                    except Exception:
                        cur_val_auc = float('-inf')
                        cur_val_acc = float('-inf')
                        cur_val_loss = float('inf')

                    if cur_val_auc > best_val_auc:
                        best_val_auc = cur_val_auc
                        best_epoch_by_val_auc = epoch
                        best_state_by_auc = copy.deepcopy(model.state_dict())

                    if cur_val_acc > best_val_acc:
                        best_val_acc = cur_val_acc
                        best_epoch_by_val_acc = epoch
                        best_state_by_acc = copy.deepcopy(model.state_dict())

                    if cur_val_loss < best_val_loss:
                        best_val_loss = cur_val_loss
                        best_epoch_by_val_loss = epoch
                        best_state_by_loss = copy.deepcopy(model.state_dict())

                    print(
                        f"Epoch {epoch+1}: "
                        f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                        f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
                        f"Train AP {train_ap:.4f} | Val AP {val_ap:.4f} | "
                        f"Train F1 {train_f1:.4f} | Val F1 {val_f1:.4f} | "
                        f"Train Bal {train_bal:.4f} | Val Bal {val_bal:.4f} | "
                        f"Val AUC {val_auc:.4f}"
                    )

                save_epoch_metrics_csv(fold_save_dir, epoch_val_metrics)

                if best_epoch_by_val_auc >= 0:
                    best_idx_by_val_auc = best_epoch_by_val_auc
                else:
                    try:
                        best_idx_by_val_auc = int(np.nanargmax(np.array(val_aucs)))
                    except Exception:
                        best_idx_by_val_auc = len(val_aucs) - 1 if len(val_aucs) > 0 else 0

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
                    best_val_labels = val_labels_per_epoch[best_idx_by_val_auc]
                    best_val_preds = val_preds_per_epoch[best_idx_by_val_auc]
                except Exception:
                    best_val_labels = last_val_labels
                    best_val_preds = last_val_preds

                if best_state_by_auc is not None:
                    model.load_state_dict(best_state_by_auc)

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

                save_confusion_matrix_plot(fold_save_dir, best_val_labels, best_val_preds, fold_num, "Validation")
                test_loss, test_acc, test_auc, test_ap, test_f1, test_bal, test_labels, test_preds, test_probs = evaluate(
                    model, test_loader, criterion, device
                )

                print(
                    f"Test Results for Fold {fold_num}: "
                    f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f} | Test AUC {test_auc:.4f} | "
                    f"Test AP {test_ap:.4f} | Test F1 {test_f1:.4f} | Test Bal {test_bal:.4f}"
                )

                save_confusion_matrix_plot(fold_save_dir, test_labels, test_preds, fold_num, "Test")

                def safe_get(lst, idx, default=0.0):
                    try:
                        return lst[idx]
                    except Exception:
                        return default

                fold_results.append({
                    "fold": fold_num,
                    "best_epoch_by_val_auc": best_idx_by_val_auc + 1,
                    "best_val_auc": safe_get(val_aucs, best_idx_by_val_auc, 0.0),
                    "best_epoch_by_val_acc": best_idx_by_val_acc + 1,
                    "best_val_acc": safe_get(val_accs, best_idx_by_val_acc, 0.0),
                    "best_epoch_by_val_loss": best_idx_by_val_loss + 1,
                    "best_val_loss": safe_get(val_losses, best_idx_by_val_loss, 0.0),

                    "train_acc": safe_get(train_accs, best_idx_by_val_auc, 0.0),
                    "val_acc": safe_get(val_accs, best_idx_by_val_auc, 0.0),
                    "test_acc": test_acc,

                    "train_loss": safe_get(train_losses, best_idx_by_val_auc, 0.0),
                    "val_loss": safe_get(val_losses, best_idx_by_val_auc, 0.0),
                    "test_loss": test_loss,

                    "train_auc": safe_get(train_aucs, best_idx_by_val_auc, 0.0),
                    "val_auc": safe_get(val_aucs, best_idx_by_val_auc, 0.0),
                    "test_auc": test_auc,

                    "train_ap": safe_get(train_aps, best_idx_by_val_auc, 0.0),
                    "val_ap": safe_get(val_aps, best_idx_by_val_auc, 0.0),
                    "test_ap": test_ap,

                    "train_f1": safe_get(train_f1s, best_idx_by_val_auc, 0.0),
                    "val_f1": safe_get(val_f1s, best_idx_by_val_auc, 0.0),
                    "test_f1": test_f1,

                    "train_bal": safe_get(train_bals, best_idx_by_val_auc, 0.0),
                    "val_bal": safe_get(val_bals, best_idx_by_val_auc, 0.0),
                    "test_bal": test_bal,

                    "test_auc_at_best_val_auc": test_auc,
                    "test_acc_at_best_val_auc": test_acc,
                    "test_loss_at_best_val_auc": test_loss,
                    "test_ap_at_best_val_auc": test_ap,
                    "test_f1_at_best_val_auc": test_f1,
                    "test_bal_at_best_val_auc": test_bal,
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
                    "val_bals": val_bals,
                })

            completed_folds = len(fold_results)
            print(f"\nCompleted prompt-token setting {n_prompt_tokens}, regime {regime}: {completed_folds} folds finished.")

            save_results(result_root, fold_results)
            save_combined_training_curves(result_root, fold_histories)
            fold_csv_path, summary_csv_path = save_best_epoch_summary_csv(result_root, fold_results)
            if fold_csv_path is not None:
                print(f"Saved per-fold best metrics to: {fold_csv_path}")
            if summary_csv_path is not None:
                print(f"Saved summary metrics to: {summary_csv_path}")

            if fold_results:
                avg_best_val_auc = float(np.mean([r.get("best_val_auc", 0.0) for r in fold_results]))
                avg_best_val_acc = float(np.mean([r.get("best_val_acc", 0.0) for r in fold_results]))
                avg_best_val_loss = float(np.mean([r.get("best_val_loss", 0.0) for r in fold_results]))
                avg_test_auc = float(np.mean([r.get("test_auc_at_best_val_auc", 0.0) for r in fold_results]))
                avg_test_acc = float(np.mean([r.get("test_acc_at_best_val_auc", 0.0) for r in fold_results]))
                print(f"Mean best val auc: {avg_best_val_auc:.4f} | Mean best val acc: {avg_best_val_acc:.4f} | Mean best val loss: {avg_best_val_loss:.4f}")
                print(f"Mean test auc at best val auc: {avg_test_auc:.4f} | Mean test acc at best val auc: {avg_test_acc:.4f}")

            print(f"\nPer-fold results for prompt-token setting {n_prompt_tokens}, regime {regime}:")
            keys = [
                "train_acc", "val_acc", "test_acc",
                "train_loss", "val_loss", "test_loss",
                "train_ap", "val_ap", "test_ap",
                "train_f1", "val_f1", "test_f1",
                "train_bal", "val_bal", "test_bal"
            ]

            for r in fold_results:
                print(
                    f"Fold {r['fold']}: "
                    f"best_epoch_by_val_auc={r.get('best_epoch_by_val_auc', 0)} | "
                    f"best_val_auc={r.get('best_val_auc', 0.0):.4f} | "
                    f"best_epoch_by_val_acc={r.get('best_epoch_by_val_acc', 0)} | "
                    f"best_val_acc={r.get('best_val_acc', 0.0):.4f} | "
                    f"best_epoch_by_val_loss={r.get('best_epoch_by_val_loss', 0)} | "
                    f"best_val_loss={r.get('best_val_loss', 0.0):.4f} | "
                    + " | ".join([f"{k}: {r.get(k, 0.0):.4f}" for k in keys])
                )

            avg = {}
            for k in keys:
                avg[k] = np.mean([r.get(k, 0.0) for r in fold_results])

            avg_best_epoch_by_val_auc = np.mean([r.get("best_epoch_by_val_auc", 0) for r in fold_results])
            avg_best_val_auc = np.mean([r.get("best_val_auc", 0.0) for r in fold_results])
            avg_best_epoch_by_val_acc = np.mean([r.get("best_epoch_by_val_acc", 0) for r in fold_results])
            avg_best_val_acc = np.mean([r.get("best_val_acc", 0.0) for r in fold_results])
            avg_best_epoch_by_val_loss = np.mean([r.get("best_epoch_by_val_loss", 0) for r in fold_results])
            avg_best_val_loss = np.mean([r.get("best_val_loss", 0.0) for r in fold_results])

            print(f"\n{completed_folds}-fold averages for prompt-token setting {n_prompt_tokens}, regime {regime}:")
            print(
                f"best_epoch_by_val_auc: {avg_best_epoch_by_val_auc:.2f}, "
                f"best_val_auc: {avg_best_val_auc:.4f}, "
                f"best_epoch_by_val_acc: {avg_best_epoch_by_val_acc:.2f}, "
                f"best_val_acc: {avg_best_val_acc:.4f}, "
                f"best_epoch_by_val_loss: {avg_best_epoch_by_val_loss:.2f}, "
                f"best_val_loss: {avg_best_val_loss:.4f}, "
                + ", ".join([f"{k}: {avg[k]:.4f}" for k in keys])
            )
            
if __name__ == "__main__":
    main()
