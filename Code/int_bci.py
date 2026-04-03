"""BCI Competition 2020 Track 3 — SincTran interpretation pipeline."""
from __future__ import annotations

# ── User configuration ────────────────────────────────────────────────────────
DATA_PATH    = "/path/to/bci-competition-2020-track-3"
WEIGHTS_ROOT = "/path/to/sinctran/weights/bci"
RESULTS_ROOT = "./results/BCI_interpret"
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core_model import SincTran
from core_dataset import stratified_kfold_loaders
from core_loaders import BCILoader

from int_module import (
    AttentionRolloutVisualizer,
    SpatialGradCAMVisualizer,
    visualize_band_attention,
    visualize_cls_umap,
)



_ALL_SUBJECTS: List[int] = list(range(1, 16))

CLASS_NAMES = ["hello", "help me", "stop", "thank you", "yes"]

MODEL_CFG: Dict = {
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
    "pool1":            8,
    "pool2":            7,
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

DATA_CFG: Dict = {
    "batch_size":   64,
    "random_state": 37,
    "aug_dict":     None,
    "normalize":    True,
    "whiten":       True,
    "pin_memory":   False,
}


def _load_model(
    weight_path: str,
    n_chans:     int,
    n_times:     int,
    n_classes:   int,
    sfreq:       float,
    device:      torch.device,
) -> SincTran:
    """
    Args:
        weight_path: Path to the ``.pth`` checkpoint.
        n_chans: Number of EEG channels.
        n_times: Number of time samples per trial.
        n_classes: Number of output classes.
        sfreq: Sampling frequency in Hz.
        device: Target device.

    Returns:
        SincTran in eval mode.
    """
    model = SincTran(
        n_chans=n_chans, n_times=n_times, n_outputs=n_classes,
        sfreq=sfreq, **MODEL_CFG,
    ).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main_bci_interpret(
    subject_id: Optional[int],
    fold:       Optional[int],
    debug_mode: bool = False,
    n_splits:   int  = 10,
) -> None:
    """
    Run all interpretation analyses for one or all BCI subjects/folds.

    Exactly one of ``subject_id`` or ``fold`` may be ``None``:

    - ``fold=None``: run all folds for the given subject; fold-average heatmaps
      are written after the last fold.
    - ``subject_id=None``: run all subjects for the given fold; fold-average
      heatmaps aggregate across subjects.

    Args:
        subject_id: Subject ID (1–15), or ``None`` to iterate all subjects.
        fold: Fold index (1-indexed), or ``None`` to iterate all folds.
        debug_mode: If ``True``, restrict each loader to a single batch.
        n_splits: Total CV folds; must match the training run.

    Raises:
        ValueError: If both ``subject_id`` and ``fold`` are ``None``.
    """
    if subject_id is None and fold is None:
        raise ValueError("At least one of subject_id or fold must be specified.")

    subjects = _ALL_SUBJECTS if subject_id is None else [subject_id]
    folds    = list(range(1, n_splits + 1)) if fold is None else [fold]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader      = BCILoader(root=DATA_PATH, csv_labels=os.path.join(DATA_PATH, "Track3_clean.csv"))
    rollout_viz = AttentionRolloutVisualizer(class_names=CLASS_NAMES)
    gradcam_viz = SpatialGradCAMVisualizer(class_names=CLASS_NAMES)

    for sid in subjects:
        X_tr, y_tr   = loader.load(sid, "trn")
        X_val, y_val = loader.load(sid, "val")
        X_tst, y_tst = loader.load(sid, "tst")
        sfreq     = loader.sfreq
        n_chans   = X_tr.shape[1]
        n_times   = X_tr.shape[2]
        n_classes = len(np.unique(y_tr))
        X_super   = np.concatenate([X_tr, X_val])
        y_super   = np.concatenate([y_tr, y_val])

        for fld in folds:
            tag     = f"sub{sid:02d}_fold{fld}"
            out_dir = Path(RESULTS_ROOT) / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            for f, _, _, test_dl, _ in stratified_kfold_loaders(
                X_super, y_super,
                n_splits     = n_splits,
                batch_size   = DATA_CFG["batch_size"],
                num_workers  = 0,
                aug_dict     = DATA_CFG["aug_dict"],
                sfreq        = sfreq,
                normalize    = DATA_CFG["normalize"],
                whiten       = DATA_CFG["whiten"],
                random_state = DATA_CFG["random_state"],
                pin_memory   = DATA_CFG["pin_memory"],
                X_test_fixed = X_tst,
                y_test_fixed = y_tst,
            ):
                if f != fld - 1:
                    continue

                if debug_mode:
                    test_dl = DataLoader(
                        Subset(test_dl.dataset, range(min(64, len(test_dl.dataset)))),
                        batch_size=DATA_CFG["batch_size"],
                    )

                model = _load_model(
                    os.path.join(WEIGHTS_ROOT, f"weights_{sid:02d}_fold{fld}.pth"),
                    n_chans, n_times, n_classes, sfreq, device,
                )

                visualize_band_attention(model, test_dl, out_dir, class_names=CLASS_NAMES, subject_tag=tag)
                visualize_cls_umap(model, test_dl, out_dir, class_names=CLASS_NAMES, subject_tag=tag)
                rollout_viz.add_subject(model, test_dl, out_dir, subject_tag=tag)
                gradcam_viz.add_subject(model, test_dl, out_dir, subject_tag=tag)
                break

            print(f"  Subject {sid:02d}  fold {fld} → {out_dir}")

        if fold is None:
            avg_dir = Path(RESULTS_ROOT) / f"sub{sid:02d}_avg"
            avg_dir.mkdir(parents=True, exist_ok=True)
            rollout_viz.save_fold_average(avg_dir, fold_tag=f"sub{sid:02d}")
            gradcam_viz.save_fold_average(avg_dir, fold_tag=f"sub{sid:02d}")

    if subject_id is None:
        avg_dir = Path(RESULTS_ROOT) / f"fold{fold}_avg"
        avg_dir.mkdir(parents=True, exist_ok=True)
        rollout_viz.save_fold_average(avg_dir, fold_tag=f"fold{fold}", average_label="subject average")
        gradcam_viz.save_fold_average(avg_dir, fold_tag=f"fold{fold}", average_label="subject average")

    print("Interpretation complete.")