"""ASU speech imagery EEG training pipeline with stratified k-fold cross-validation."""
from __future__ import annotations

# ── User configuration ────────────────────────────────────────────────────────
DATA_PATH    = "/path/to/asu-dataset"
RESULTS_ROOT = "./results"
# ─────────────────────────────────────────────────────────────────────────────

import gc
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from core_loaders import ASULoader
from core_dataset import preprocess, stratified_kfold_loaders
from core_model import SincTran
from core_loss import CELoss
from core_train import train_model
from core_utils import format_time, save_metadata, save_model_summary, set_all_seeds


def main_asu(
    task:       str,
    debug_mode: bool = False,
    subjects:   Optional[List[str]] = None,
    num_epochs: int = 100,
    n_splits:   int = 5,
    early_stop: bool = False,
) -> None:
    """
    Args:
        task: ASU task identifier, one of ``"n1"``, ``"n2"``, ``"n3"``, ``"n4"``.
        debug_mode: If True, train only the first subject for 1 epoch.
        subjects: Restrict training to these subject IDs; defaults to all available.
        num_epochs: Number of training epochs per fold. Default: 100.
        n_splits: Number of CV folds. Default: 5.
        early_stop: Enable early stopping. Default: False.
    """
    DATA_CFG = {
        "batch_size":   64,
        "random_state": 37,
        "sfreq":        256,
        "aug_dict": {
            "freq_shift": {"probability": 0.5, "max_delta_freq": 2.0},
            "gaussian":   {"probability": 0.5, "std": 0.01},
            "ch_dropout": {"probability": 0.2, "p_drop": 0.2},
        },
        "normalize": True,
        "whiten":    True,
    }

    MODEL_CFG = {
        "eeg_bands": {
            (0.5,  4): 129,
            (4,    8):  65,
            (8,   13):  33,
            (13,  30):  17,
            (30, 100):   9,
        },
        "n_filters":        8,
        "depth_multiplier": 8,
        "t_kern":          15,
        "pool1":           10,
        "pool2":            9,
        "drop_cnn":      0.01,
        "num_heads":        4,
        "ff_ratio":       2.0,
        "drop_trans":    0.10,
        "num_layers":       4,
        "use_cls":       True,
        "trans_act":   "gelu",
        "norm_first":    True,
        "embedding_dim":  128,
    }

    LOSS_CFG = {
        "label_smoothing": 0.0,
    }

    TRAIN_CFG = {
        "optimizer_class":  torch.optim.AdamW,
        "optimizer_kwargs": {
            "lr":           1e-3,
            "weight_decay": 0.0,
        },
        "scheduler_class":  torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {
            "factor":    0.5,
            "patience":  5,
            "min_lr":    1e-6,
            "threshold": 0.0,
        },
        "monitor_metric": "f1",
        "early_stopping": {
            "enabled":   early_stop,
            "patience":  10,
            "min_delta": 0.0,
        },
        "use_amp":   False,
        "grad_clip": None,
    }

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root   = os.path.join(RESULTS_ROOT, "models")
    metrics_root = os.path.join(RESULTS_ROOT, "metrics")
    meta_root    = os.path.join(RESULTS_ROOT, "metadata")

    for path in (model_root, metrics_root, meta_root):
        os.makedirs(path, exist_ok=True)

    set_all_seeds(DATA_CFG["random_state"])

    loader       = ASULoader(root=DATA_PATH, limit_chan=True)
    subject_list = loader.list_subjects(task)

    if subjects:
        subject_list = [s for s in subject_list if s in subjects]
    if not subject_list:
        print(f"No matching subjects for task {task!r} — skipping.")
        return

    if debug_mode:
        subject_list = subject_list[:1]
        num_epochs   = 1
        print("Debug mode: first subject, 1 epoch.")

    # Load first subject to resolve shapes for model summary; reused in loop.
    X0, y0    = loader.load(subject_name=subject_list[0], dataset=task)
    n_chans   = X0.shape[1]
    n_times   = X0.shape[2]
    n_classes = len(np.unique(y0))

    save_model_summary(
        SincTran(
            n_chans=n_chans, n_times=n_times, n_outputs=n_classes,
            sfreq=DATA_CFG["sfreq"], **MODEL_CFG,
        ),
        input_shape = (1, n_chans, n_times),
        save_path   = os.path.join(meta_root, "model_summary.txt"),
    )

    training_times: Dict[str, str] = {}

    for subject_id in subject_list:
        print(f"\nSubject {subject_id}")

        X, y      = (X0, y0) if subject_id == subject_list[0] else loader.load(subject_name=subject_id, dataset=task)
        n_classes = len(np.unique(y))
        print(f"Classes: {n_classes} | Channels: {n_chans} | Length: {n_times} | sfreq: {DATA_CFG['sfreq']} Hz")

        X = preprocess(X, fs=DATA_CFG["sfreq"], bandpass_hz=(8, 70), notch_hz=60)

        start = time.time()

        for fold, train_dl, _, test_dl, cls_wts in stratified_kfold_loaders(
            X, y,
            n_splits     = n_splits,
            batch_size   = DATA_CFG["batch_size"],
            num_workers  = 0,
            aug_dict     = DATA_CFG["aug_dict"],
            sfreq        = DATA_CFG["sfreq"],
            normalize    = DATA_CFG["normalize"],
            whiten       = DATA_CFG["whiten"],
            random_state = DATA_CFG["random_state"],
        ):
            print(f"  Fold {fold + 1}/{n_splits}")

            model = SincTran(
                n_chans   = n_chans,
                n_times   = n_times,
                n_outputs = n_classes,
                sfreq     = DATA_CFG["sfreq"],
                **MODEL_CFG,
            ).to(device)

            if torch.cuda.device_count() > 1:
                print(f"  Using {torch.cuda.device_count()} GPUs")
                model = torch.nn.DataParallel(model)

            criterion = CELoss(**LOSS_CFG, class_weights=cls_wts.to(device)).to(device)

            train_cfg = {
                **TRAIN_CFG,
                "device":       device,
                "subject_name": str(subject_id),
                "cm_tag":       f"{subject_id}_fold{fold + 1}",
                "model_path":   os.path.join(model_root, f"weights_{subject_id}_fold{fold + 1}.pth"),
                "metrics_root": os.path.join(metrics_root, f"fold_{fold + 1}"),
            }

            train_model(
                model        = model,
                criterion    = criterion,
                train_loader = train_dl,
                val_loader   = None,
                test_loader  = test_dl,
                num_epochs   = num_epochs,
                config       = train_cfg,
            )

            del model, criterion
            torch.cuda.empty_cache()
            gc.collect()

        elapsed = time.time() - start
        training_times[str(subject_id)] = format_time(elapsed)
        print(f"Subject {subject_id} completed in {training_times[str(subject_id)]}")

        del train_dl, test_dl
        torch.cuda.empty_cache()
        gc.collect()

    save_metadata(
        configs={
            "data": {
                **DATA_CFG,
                "n_chans":   n_chans,
                "n_times":   n_times,
                "n_classes": n_classes,
                "n_splits":  n_splits,
            },
            "model":    MODEL_CFG,
            "loss":     LOSS_CFG,
            "training": {**TRAIN_CFG, "num_epochs": num_epochs},
        },
        metadata_root  = meta_root,
        task           = task,
        training_times = training_times,
    )

    print(f"\nTraining complete — {len(training_times)} subjects across {n_splits} folds.")