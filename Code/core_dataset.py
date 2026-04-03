"""
EEG preprocessing and dataloader pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from braindecode.augmentation import AugmentedDataLoader, ChannelsDropout, FrequencyShift, GaussianNoise
from collections import Counter
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Generator, Optional, Tuple


class ZCAWhitening:
    """
    ZCA whitening for channel decorrelation. Defaults to CPU to avoid CUDA kernel issues.

    Args:
        epsilon (float): Regularization constant. Default: 1e-5.
        device (str): Computation device. Default: 'cpu'.
    """

    def __init__(self, epsilon: float = 1e-5, device: str = "cpu") -> None:
        self.epsilon = epsilon
        self.device = device
        self.W: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self.n_channels: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self.W is not None

    def _prepare(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        X = X.cpu().to(dtype=torch.float32)
        if self.device != "cpu":
            try:
                X = X.to(self.device)
            except RuntimeError as e:
                print(f"Warning: falling back to CPU. Error: {e}")
                self.device = "cpu"
        if X.ndim == 2:
            X = X.unsqueeze(-1)
        elif X.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")
        return X

    @torch.no_grad()
    def fit(self, X: np.ndarray | torch.Tensor) -> ZCAWhitening:
        """
        Args:
            X: Shape (B, C, T).
        """
        X = self._prepare(X)
        B, C, T = X.shape
        self.n_channels = C
        X_flat = X.transpose(1, 2).reshape(B * T, C)
        self.mean = X_flat.mean(dim=0, keepdim=True)
        X_c = X_flat - self.mean
        cov = X_c.T @ X_c / X_c.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        idx = eigenvalues.argsort(descending=True)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        D_inv_sqrt = torch.diag(torch.rsqrt(eigenvalues + self.epsilon))
        self.W = eigenvectors @ D_inv_sqrt @ eigenvectors.T
        return self

    @torch.no_grad()
    def transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Args:
            X: Shape (B, C, T).

        Returns:
            np.ndarray: Shape (B, C, T).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")
        X = self._prepare(X)
        if X.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {X.shape[1]}.")
        B, C, T = X.shape
        X_flat = X.transpose(1, 2).reshape(B * T, C)
        X_white = (X_flat - self.mean) @ self.W.T
        return X_white.reshape(B, T, C).transpose(1, 2).cpu().numpy()


def preprocess(
    X: np.ndarray,
    fs: float = 256.0,
    clip_std: float = 5.0,
    bandpass_hz: Tuple[float, float] = (0.1, 100.0),
    filter_order: int = 5,
    notch_hz: Optional[float] = None,
    notch_Q: float = 30.0,
) -> np.ndarray:
    """
    Args:
        X (np.ndarray): Shape (n_epochs, n_channels, n_times).
        fs (float): Sampling frequency in Hz. Default: 256.0.
        clip_std (float): Clipping threshold in per-epoch std units. Default: 5.0.
        bandpass_hz (Tuple[float, float]): Bandpass (low, high) Hz. Default: (0.1, 100.0).
        filter_order (int): Butterworth filter order. Default: 5.
        notch_hz (float, optional): Notch centre frequency in Hz. Skipped if None. Default: None.
        notch_Q (float): Notch quality factor. Default: 30.0.

    Returns:
        np.ndarray: Shape (n_epochs, n_channels, n_times), float32.
    """
    X = X.astype(np.float64)
    std = X.std(axis=-1, keepdims=True).clip(min=1e-8)
    X = np.clip(X, -clip_std * std, clip_std * std)
    nyq = fs / 2.0
    sos = butter(filter_order, [bandpass_hz[0] / nyq, bandpass_hz[1] / nyq], btype="bandpass", output="sos")
    X = sosfiltfilt(sos, X, axis=-1)
    if notch_hz is not None:
        b, a = iirnotch(notch_hz / nyq, notch_Q)
        X = filtfilt(b, a, X, axis=-1)
    return X.astype(np.float32)


class _EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx].clone(), self.y[idx]


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    aug_dict: Optional[Dict] = None,
    sfreq: float = 256.0,
    normalize: bool = False,
    whiten: bool = False,
    random_state: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, torch.Tensor]:
    """
    Args:
        X_train (np.ndarray): Shape (n_train, n_channels, n_times).
        y_train (np.ndarray): Shape (n_train,).
        X_test (np.ndarray): Shape (n_test, n_channels, n_times).
        y_test (np.ndarray): Shape (n_test,).
        X_val (np.ndarray, optional): Shape (n_val, n_channels, n_times).
        y_val (np.ndarray, optional): Shape (n_val,).
        batch_size (int): Default: 32.
        num_workers (int): Default: 0.
        aug_dict (Dict, optional): Keys: "freq_shift", "gaussian", "ch_dropout",
            each with sub-keys "probability" and transform-specific params.
        sfreq (float): Sampling frequency in Hz, used by FrequencyShift. Default: 256.0.
        normalize (bool): Per-channel z-score using train stats. Default: False.
        whiten (bool): ZCA whitening fitted on train split. Default: False.
        random_state (int): Default: 42.

    Returns:
        Tuple: (train_loader, val_loader, test_loader, class_weights).
               val_loader is None when X_val is None.
    """
    if normalize:
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True).clip(min=1e-8)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        if X_val is not None:
            X_val = (X_val - mean) / std
    if whiten:
        whitener = ZCAWhitening()
        whitener.fit(X_train)
        X_train = whitener.transform(X_train)
        X_test = whitener.transform(X_test)
        if X_val is not None:
            X_val = whitener.transform(X_val)
    counts = Counter(y_train.tolist())
    n_cls, total = len(counts), sum(counts.values())
    class_weights = torch.tensor(
        [total / (n_cls * counts[i]) for i in range(n_cls)], dtype=torch.float32
    )
    transforms = []
    if aug_dict:
        if cfg := aug_dict.get("freq_shift"):
            transforms.append(FrequencyShift(
                probability=cfg.get("probability", 0.5), sfreq=sfreq,
                max_delta_freq=cfg.get("max_delta_freq", 2.0), random_state=random_state,
            ))
        if cfg := aug_dict.get("gaussian"):
            transforms.append(GaussianNoise(
                probability=cfg.get("probability", 0.5),
                std=cfg.get("std", 0.01), random_state=random_state + 1,
            ))
        if cfg := aug_dict.get("ch_dropout"):
            transforms.append(ChannelsDropout(
                probability=cfg.get("probability", 0.2),
                p_drop=cfg.get("p_drop", 0.2), random_state=random_state + 2,
            ))
    if transforms:
        train_loader = AugmentedDataLoader(
            _EEGDataset(X_train, y_train), transforms=transforms,
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        g = torch.Generator()
        g.manual_seed(random_state)
        train_loader = DataLoader(
            _EEGDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, generator=g,
        )
    val_loader = (
        DataLoader(_EEGDataset(X_val, y_val), batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory,)
        if X_val is not None else None
    )
    test_loader = DataLoader(
        _EEGDataset(X_test, y_test), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, class_weights


def stratified_kfold_loaders(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    random_state: int = 42,
    aug_dict: Optional[Dict] = None,
    sfreq: float = 256.0,
    normalize: bool = False,
    whiten: bool = False,
    pin_memory: bool = True,
    X_test_fixed: Optional[np.ndarray] = None,
    y_test_fixed: Optional[np.ndarray] = None,
) -> Generator[Tuple[int, DataLoader, Optional[DataLoader], DataLoader, torch.Tensor], None, None]:
    """
    Args:
        X (np.ndarray): Shape (n_trials, n_channels, n_times).
        y (np.ndarray): Shape (n_trials,).
        n_splits (int): Default: 5.
        batch_size (int): Default: 32.
        num_workers (int): Default: 0.
        random_state (int): Default: 42.
        aug_dict (Dict, optional): Forwarded to create_data_loaders.
        sfreq (float): Default: 256.0.
        normalize (bool): Default: False.
        whiten (bool): Default: False.
        X_test_fixed (np.ndarray, optional): Shared test set across all folds;
            held-out fold becomes val_loader. When None, held-out fold is
            test_loader and val_loader is None.
        y_test_fixed (np.ndarray, optional): Labels for X_test_fixed.

    Yields:
        Tuple: (fold_index, train_loader, val_loader, test_loader, class_weights).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    shared = dict(batch_size=batch_size, num_workers=num_workers, aug_dict=aug_dict,
                  sfreq=sfreq, normalize=normalize, whiten=whiten, random_state=random_state,
                  pin_memory=pin_memory)
    for fold, (tr_idx, held_idx) in enumerate(skf.split(X, y)):
        if X_test_fixed is not None:
            loaders = create_data_loaders(
                X[tr_idx], y[tr_idx], X_test_fixed, y_test_fixed,
                X_val=X[held_idx], y_val=y[held_idx], **shared,
            )
        else:
            loaders = create_data_loaders(
                X[tr_idx], y[tr_idx], X[held_idx], y[held_idx], **shared,
            )
        yield (fold, *loaders)