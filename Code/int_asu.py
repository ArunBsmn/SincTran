"""ASU speech imagery EEG — SincTran interpretation pipeline."""
from __future__ import annotations

# ── User configuration ────────────────────────────────────────────────────────
DATA_PATH    = "/path/to/asu-rawmat"
WEIGHTS_ROOT = "/path/to/sinctran/weights"
RESULTS_ROOT = "./results/ASU_interpret"
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core_model import SincTran
from core_dataset import preprocess, stratified_kfold_loaders
from core_loaders import ASULoader

from int_module import (
    AttentionRolloutVisualizer,
    SpatialGradCAMVisualizer,
    visualize_band_attention,
    visualize_cls_umap,
)


TASK_CLASS_NAMES: Dict[str, List[str]] = {
    "n1": ["cooperate", "independent"],
    "n2": ["in", "out", "up"],
    "n3": ["/a/", "/i/", "/u/"],
    "n4": ["in", "cooperate"],
}

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

DATA_CFG: Dict = {
    "batch_size":   64,
    "random_state": 37,
    "sfreq":        256,
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
    device:      torch.device,
) -> SincTran:
    """
    Args:
        weight_path: Path to the ``.pth`` checkpoint.
        n_chans: Number of EEG channels.
        n_times: Number of time samples per trial.
        n_classes: Number of output classes.
        device: Target device.

    Returns:
        SincTran in eval mode.
    """
    model = SincTran(
        n_chans=n_chans, n_times=n_times, n_outputs=n_classes,
        sfreq=DATA_CFG["sfreq"], **MODEL_CFG,
    ).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main_asu_interpret(
    task:       str,
    subject_id: Optional[str],
    fold:       Optional[int],
    debug_mode: bool = False,
    n_splits:   int  = 10,
) -> None:
    """
    Run all interpretation analyses for one or all ASU subjects/folds.

    Exactly one of ``subject_id`` or ``fold`` may be ``None``:

    - ``fold=None``: run all folds for the given subject; fold-average heatmaps
      are written after the last fold.
    - ``subject_id=None``: run all subjects for the given fold; heatmaps
      aggregate across subjects.

    Args:
        task: ASU task identifier, one of ``"n1"``, ``"n2"``, ``"n3"``, ``"n4"``.
        subject_id: Subject identifier string, or ``None`` to iterate all subjects.
        fold: Fold index (1-indexed), or ``None`` to iterate all folds.
        debug_mode: If ``True``, restrict each loader to a single batch.
        n_splits: Total CV folds; must match the training run.

    Raises:
        ValueError: If both ``subject_id`` and ``fold`` are ``None``,
                    or if ``task`` is not a recognised key.
    """
    if task not in TASK_CLASS_NAMES:
        raise ValueError(f"Unknown task {task!r}. Valid options: {sorted(TASK_CLASS_NAMES)}.")
    if subject_id is None and fold is None:
        raise ValueError("At least one of subject_id or fold must be specified.")

    sfreq       = DATA_CFG["sfreq"]
    class_names = TASK_CLASS_NAMES[task]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader       = ASULoader(root=DATA_PATH, limit_chan=True)
    all_subjects = loader.list_subjects(task)
    subjects     = all_subjects if subject_id is None else [subject_id]
    folds        = list(range(1, n_splits + 1)) if fold is None else [fold]

    rollout_viz = AttentionRolloutVisualizer(class_names=class_names)
    gradcam_viz = SpatialGradCAMVisualizer(class_names=class_names)

    for sid in subjects:
        X, y      = loader.load(subject_name=sid, dataset=task)
        X         = preprocess(X, fs=sfreq, bandpass_hz=(8, 70), notch_hz=60)
        n_chans   = X.shape[1]
        n_times   = X.shape[2]
        n_classes = len(np.unique(y))
        sid_label = sid.removeprefix("sub_")

        for fld in folds:
            tag     = f"{task}_sub{sid_label}_fold{fld}"
            out_dir = Path(RESULTS_ROOT) / task / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            for f, _, _, test_dl, _ in stratified_kfold_loaders(
                X, y,
                n_splits     = n_splits,
                batch_size   = DATA_CFG["batch_size"],
                num_workers  = 0,
                aug_dict     = DATA_CFG["aug_dict"],
                sfreq        = sfreq,
                normalize    = DATA_CFG["normalize"],
                whiten       = DATA_CFG["whiten"],
                random_state = DATA_CFG["random_state"],
                pin_memory   = DATA_CFG["pin_memory"],
            ):
                if f != fld - 1:
                    continue

                if debug_mode:
                    test_dl = DataLoader(
                        Subset(test_dl.dataset, range(min(64, len(test_dl.dataset)))),
                        batch_size=DATA_CFG["batch_size"],
                    )

                model = _load_model(
                    os.path.join(WEIGHTS_ROOT, f"asu_{task}", f"weights_{sid}_fold{fld}.pth"),
                    n_chans, n_times, n_classes, device,
                )

                visualize_band_attention(model, test_dl, out_dir, class_names=class_names, subject_tag=tag)
                visualize_cls_umap(model, test_dl, out_dir, class_names=class_names, subject_tag=tag)
                rollout_viz.add_subject(model, test_dl, out_dir, subject_tag=tag)
                gradcam_viz.add_subject(model, test_dl, out_dir, subject_tag=tag)
                break

            print(f"  Task {task!r}  subject {sid}  fold {fld} → {out_dir}")

        if fold is None:
            avg_dir = Path(RESULTS_ROOT) / task / f"{sid_label}_avg"
            avg_dir.mkdir(parents=True, exist_ok=True)
            rollout_viz.save_fold_average(avg_dir, fold_tag=f"{task}_{sid_label}")
            gradcam_viz.save_fold_average(avg_dir, fold_tag=f"{task}_{sid_label}")

    if subject_id is None:
        avg_dir = Path(RESULTS_ROOT) / task / f"fold{fold}_avg"
        avg_dir.mkdir(parents=True, exist_ok=True)
        rollout_viz.save_fold_average(avg_dir, fold_tag=f"{task}_fold{fold}", average_label="subject average")
        gradcam_viz.save_fold_average(avg_dir, fold_tag=f"{task}_fold{fold}", average_label="subject average")

    print("Interpretation complete.")