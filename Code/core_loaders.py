"""EEG dataset loaders for ASU and BCI Competition imagined-speech datasets."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio


EEGData = Tuple[np.ndarray, Optional[np.ndarray]]


class ASULoader:
    """
    Loader for ASU imagined-speech EEG datasets.

    Args:
        root (str): Path to root directory containing dataset folders.
        limit_chan (bool): Clip to MAX_CHANNELS before EOG removal.
    """

    DATASETS: Dict[str, dict] = {
        "n1": {"folder": "Long_words",       "classes": 2},
        "n2": {"folder": "Short_words",      "classes": 3},
        "n3": {"folder": "Vowels",           "classes": 3},
        "n4": {"folder": "Short_Long_words", "classes": 2},
    }

    EOG_CHANNELS: List[int] = [0, 9, 32, 63]
    MAX_CHANNELS: int = 64

    def __init__(self, root: str, limit_chan: bool = False) -> None:
        self.root = root
        self.limit_chan = limit_chan

    def load(self, subject_name: str, dataset: str) -> EEGData:
        """
        Args:
            subject_name (str): Subject identifier, e.g. "sub_2b".
            dataset (str): One of "n1", "n2", "n3", "n4".

        Returns:
            X: (N, C, T) trials × channels × time.
            y: (N,) 0-indexed class labels.
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset!r}. Valid: {list(self.DATASETS)}")

        info = self.DATASETS[dataset]
        folder_path = self._get_folder_path(dataset)
        path = os.path.join(folder_path, self._find_file(subject_name, folder_path))

        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        data_cell = mat["eeg_data_wrt_task_rep_no_eog_256Hz_last_beep"]

        X_list, y_list = [], []
        for class_idx in range(info["classes"]):
            class_trials = data_cell[class_idx]
            trials = [class_trials] if class_trials.ndim == 2 else list(class_trials)
            for trial in trials:
                if self.limit_chan and trial.shape[0] > self.MAX_CHANNELS:
                    trial = trial[: self.MAX_CHANNELS, :]
                valid_ch = [i for i in range(trial.shape[0]) if i not in self.EOG_CHANNELS]
                X_list.append(trial[valid_ch, :])
                y_list.append(class_idx)

        return np.stack(X_list, axis=0), np.array(y_list, dtype=int)

    def list_subjects(self, dataset: str) -> List[str]:
        """
        Args:
            dataset (str): One of "n1", "n2", "n3", "n4".

        Returns:
            Sorted list of subject identifiers found in the dataset folder.
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset!r}. Valid: {list(self.DATASETS)}")

        folder_path = self._get_folder_path(dataset)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        subjects = set()
        for f in os.listdir(folder_path):
            if (
                f.endswith(".mat")
                and f.startswith("sub_")
                and "time_correlation_effect" not in f
                and "bw20_8s" not in f
            ):
                parts = f.split("_")
                if len(parts) >= 2:
                    subjects.add(f"{parts[0]}_{parts[1]}")

        return sorted(subjects)

    def _get_folder_path(self, dataset: str) -> str:
        folder_path = os.path.join(self.root, self.DATASETS[dataset]["folder"])
        nested = os.path.join(folder_path, self.DATASETS[dataset]["folder"])
        return nested if os.path.exists(nested) else folder_path

    def _find_file(self, subject_name: str, folder_path: str) -> str:
        matches = [
            f for f in os.listdir(folder_path)
            if f.startswith(f"{subject_name}_")
            and f.endswith(".mat")
            and "time_correlation_effect" not in f
            and "bw20_8s" not in f
        ]
        if not matches:
            raise FileNotFoundError(
                f"No file for subject {subject_name!r} in {folder_path}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple files for subject {subject_name!r}: {matches}"
            )
        return matches[0]


class BCILoader:
    """
    Loader for BCI Competition imagined-speech datasets.

    Args:
        root (str): Path to root directory containing the track folder.
        csv_labels (str, optional): Path to CSV with test-set labels.
            Expected columns: ``sample_id``, ``trial``, ``label``.
    """

    PARTITION_MAP: Dict[str, Tuple[str, str]] = {
        "trn": ("Training set",   "epo_train"),
        "val": ("Validation set", "epo_validation"),
        "tst": ("Test set",       "epo_test"),
    }

    def __init__(self, root: str, csv_labels: Optional[str] = None) -> None:
        self.root = os.path.join(root, "Track3 Imagined speech classification")
        self.csv = pd.read_csv(csv_labels) if csv_labels else None
        self.sfreq: Optional[float] = None

    def load(self, subject_id: int, partition: str) -> EEGData:
        """
        Args:
            subject_id (int): 1-based subject index.
            partition (str): One of "trn", "val", "tst".

        Returns:
            X: (N, C, T) trials × channels × time.
            y: (N,) 0-indexed labels, or None for test partition without CSV.
        """
        if partition not in self.PARTITION_MAP:
            raise ValueError(
                f"Unknown partition: {partition!r}. Valid: {list(self.PARTITION_MAP)}"
            )

        folder, epo_key = self.PARTITION_MAP[partition]
        path = os.path.join(self.root, folder, f"Data_Sample{subject_id:02d}.mat")

        if partition == "tst":
            return self._load_hdf5(path, epo_key, subject_id)
        return self._load_mat(path, epo_key)

    def load_channels(self, subject_id: int) -> Dict[str, Any]:
        """
        Args:
            subject_id (int): 1-based subject index.

        Returns:
            Dict with keys ``labels``, ``x``, ``y``, ``pos_3d``.
        """
        folder, _ = self.PARTITION_MAP["trn"]
        path = os.path.join(self.root, folder, f"Data_Sample{subject_id:02d}.mat")
        return self._channel_info_mat(path)

    def _load_hdf5(self, path: str, epo_key: str, subject_id: int) -> EEGData:
        with h5py.File(path, "r") as f:
            epo = f[epo_key]
            X = np.array(epo["x"])
            if "fs" in epo:
                self.sfreq = float(np.array(epo["fs"]))

        y = None
        if self.csv is not None:
            labels = (
                self.csv[self.csv["sample_id"] == subject_id]
                .sort_values("trial")["label"]
                .to_numpy()
            )
            y = (labels - 1 if labels.min() >= 1 else labels).astype(int)

        return X, y

    def _load_mat(self, path: str, epo_key: str) -> EEGData:
        mat = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
        epo = mat[epo_key][0, 0]
        X = np.transpose(epo.x, (2, 1, 0))

        if hasattr(epo, "fs"):
            self.sfreq = float(epo.fs[0][0])

        labels = epo.y
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=0) if labels.shape[0] > 1 else labels.flatten()
        y = (labels - 1 if labels.min() >= 1 else labels).astype(int)

        return X, y

    def _channel_info_mat(self, path: str) -> Dict[str, Any]:
        mat = sio.loadmat(path, squeeze_me=False)
        mnt = mat["mnt"][0, 0]
        clab = mnt["clab"][0]
        labels = [str(lbl[0]) if isinstance(lbl, np.ndarray) else str(lbl) for lbl in clab]
        return {
            "labels": labels,
            "x":      mnt["x"].flatten(),
            "y":      mnt["y"].flatten(),
            "pos_3d": mnt["pos_3d"],
        }