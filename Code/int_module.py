"""Band attention, attention rollout, UMAP, and GradCAM interpretation utilities for SincTran."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from umap import UMAP

from core_model import SincTran, TransparentEncoderLayer

# ---------------------------------------------------------------------------
# Palettes — Okabe & Ito (2008), colorblind- and grayscale-safe
# ---------------------------------------------------------------------------

_PALETTES: Dict[int, List[str]] = {
    2: ["#E69F00", "#0072B2"],
    3: ["#E69F00", "#009E73", "#0072B2"],
    4: ["#E69F00", "#009E73", "#56B4E9", "#0072B2"],
    5: ["#E69F00", "#009E73", "#56B4E9", "#0072B2", "#D55E00"],
    6: ["#E69F00", "#009E73", "#56B4E9", "#0072B2", "#D55E00", "#CC79A7"],
    7: ["#E69F00", "#009E73", "#56B4E9", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"],
    8: ["#E69F00", "#009E73", "#56B4E9", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"],
}
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

_CMAP_ROLLOUT = LinearSegmentedColormap.from_list("rollout", ["#ffffff", "#2c2c54"])
_CMAP_GRADCAM = LinearSegmentedColormap.from_list("gradcam", ["#f7f7f7", "#4d004b"])

# Heatmaps: slightly larger than original, but not as large as scatter/boxplots
_FS_TICK_HM   = 22
_FS_LABEL_HM  = 22
_FS_TITLE_HM  = 24

# UMAP and boxplots: larger sizes for legibility at single-column width
_FS_TICK   = 26
_FS_LABEL  = 26
_FS_LEGEND = 26
_FS_TITLE  = 28

# Figure widths: narrower to leave room for outside legends
_FIG_W_SCATTER = 20
_FIG_W_BOX     = 20
_FIG_H         = 8
_FIG_W_HM      = 16
_FIG_H_HM      = 6

# Fraction of figure width reserved for the legend (right-side)
_PLOT_LEGEND_RATIO = [4, 1]


def _palette(n: int) -> List[str]:
    if not 1 <= n <= 8:
        raise ValueError(f"Palette supports 1–8 classes; got {n}.")
    return _PALETTES[max(2, n)][:n]


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

@dataclass
class _HeatmapSpec:
    """Display parameters for one heatmap type."""
    xlabel:     str
    title:      str
    cbar_label: str
    cmap:       LinearSegmentedColormap


def _save_heatmap(
    matrix: np.ndarray,
    col_labels: List[str],
    col_ticks: List[int],
    spec: _HeatmapSpec,
    save_path: Path,
) -> None:
    n_rows, _ = matrix.shape
    fig, ax = plt.subplots(figsize=(_FIG_W_HM, _FIG_H_HM))
    im = ax.imshow(matrix, aspect="auto", cmap=spec.cmap, interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(range(n_rows), fontsize=_FS_TICK_HM)
    ax.set_ylabel("Class", fontsize=_FS_LABEL_HM)
    ax.set_xticks(col_ticks)
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=_FS_TICK_HM)
    ax.set_xlabel(spec.xlabel, fontsize=_FS_LABEL_HM)
    ax.set_title(spec.title, fontsize=_FS_TITLE_HM)
    ax.tick_params(axis="both", labelsize=_FS_TICK_HM)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=_FS_TICK_HM)
    cb.set_label(spec.cbar_label, fontsize=_FS_TICK_HM)
    fig.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _minmax_rows(m: np.ndarray) -> np.ndarray:
    lo, hi = m.min(axis=1, keepdims=True), m.max(axis=1, keepdims=True)
    return (m - lo) / (hi - lo + 1e-8)


def _col_ticks(n: int, max_ticks: int = 20) -> List[int]:
    return list(range(0, n, max(1, n // max_ticks)))


# ---------------------------------------------------------------------------
# Heatmap visualizer base
# ---------------------------------------------------------------------------

class _HeatmapVisualizer:
    """Accumulate per-subject matrices and write subject-wise and fold-average heatmaps.

    Subclasses implement ``_collect`` and ``_heatmap_kwargs``.
    """

    def __init__(self) -> None:
        # Per-subject unnormalized matrices: (n_classes, n_features)
        self._subject_matrices: List[np.ndarray] = []

    def reset(self) -> None:
        """Clear accumulated data to reuse across pipeline runs."""
        self._subject_matrices.clear()

    def _collect(self, model: SincTran, loader: DataLoader) -> np.ndarray:
        raise NotImplementedError

    def _heatmap_kwargs(self, matrix: np.ndarray) -> dict:
        raise NotImplementedError

    def add_subject(
        self,
        model: SincTran,
        loader: DataLoader,
        output_path: Path,
        subject_tag: str = "",
    ) -> None:
        """Collect data for one subject, save its heatmap, and accumulate.

        Args:
            model: SincTran in eval mode with weights loaded.
            loader: DataLoader for this subject.
            output_path: Directory for output figures.
            subject_tag: Appended to the output filename only.
        """
        matrix = self._collect(model, loader)
        self._subject_matrices.append(matrix)
        tag = f"_{subject_tag}" if subject_tag else ""
        _save_heatmap(
            _minmax_rows(matrix), **self._heatmap_kwargs(matrix),
            save_path=output_path / f"{self._stem}{tag}.svg",
        )

    def save_fold_average(self, output_path: Path, fold_tag: str = "") -> None:
        """Average accumulated subjects, normalize, and save heatmap.

        Args:
            output_path: Directory for output figures.
            fold_tag: Appended to the output filename only.
        """
        if not self._subject_matrices:
            raise RuntimeError("No subjects accumulated. Call add_subject first.")
        mean_matrix = np.stack(self._subject_matrices).mean(axis=0)
        tag = f"_{fold_tag}" if fold_tag else ""
        _save_heatmap(
            _minmax_rows(mean_matrix), **self._heatmap_kwargs(mean_matrix),
            save_path=output_path / f"{self._stem}_fold_avg{tag}.svg",
        )


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

def _rollout(attentions: List[np.ndarray], discard_ratio: float) -> np.ndarray:
    """Attention rollout across transformer layers (Abnar & Zuidema 2020).

    Args:
        attentions: Per-layer attention weights, each ``(B, H, S, S)``.
        discard_ratio: Fraction of lowest-weight entries zeroed per layer.

    Returns:
        Rollout matrix ``(B, S, S)``.
    """
    result = np.eye(attentions[0].shape[-1])[None]
    for attn in attentions:
        mean_head = attn.mean(axis=1)
        flat = mean_head.reshape(mean_head.shape[0], -1)
        threshold = np.quantile(flat, discard_ratio, axis=1, keepdims=True)
        mean_head[mean_head < threshold.reshape(mean_head.shape[0], 1, 1)] = 0.0
        mean_head = mean_head + np.eye(mean_head.shape[-1])[None]
        mean_head /= mean_head.sum(axis=-1, keepdims=True) + 1e-8
        result = np.matmul(mean_head, result)
    return result


_ROLLOUT_SPEC = _HeatmapSpec(
    xlabel="Temporal token index",
    title="Attention rollout - CLS token",
    cbar_label="Normalised rollout weight",
    cmap=_CMAP_ROLLOUT,
)


class AttentionRolloutVisualizer(_HeatmapVisualizer):
    """Per-subject and fold-average CLS-token attention rollout heatmaps.

    Args:
        class_names: Display label for each class index.
        discard_ratio: Fraction of lowest rollout weights zeroed per layer.
    """

    _stem = "attn_rollout"

    def __init__(
        self,
        class_names: Optional[Sequence[str]] = None,
        discard_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.class_names = list(class_names) if class_names else None
        self.discard_ratio = discard_ratio

    def _collect(self, model: SincTran, loader: DataLoader) -> np.ndarray:
        if not model.use_cls:
            raise ValueError("Attention rollout requires use_cls=True.")
        layers = [m for m in model.encoder.modules() if isinstance(m, TransparentEncoderLayer)]
        per_class: Dict[int, List[np.ndarray]] = {}
        with torch.no_grad():
            for X, y in loader:
                model(X)
                attns = [l.last_attn_weights.cpu().numpy() for l in layers]
                cls_row = _rollout(attns, self.discard_ratio)[:, 0, 1:]
                for i, label in enumerate(y.numpy()):
                    per_class.setdefault(int(label), []).append(cls_row[i])
        n_classes = max(per_class) + 1
        seq_len = per_class[0][0].shape[0]
        matrix = np.zeros((n_classes, seq_len))
        for c, rows in per_class.items():
            matrix[c] = np.stack(rows).mean(axis=0)
        return matrix

    def _heatmap_kwargs(self, matrix: np.ndarray) -> dict:
        _, seq_len = matrix.shape
        ticks = _col_ticks(seq_len)
        return dict(col_labels=[str(i) for i in ticks], col_ticks=ticks, spec=_ROLLOUT_SPEC)


# ---------------------------------------------------------------------------
# Spatial GradCAM
# ---------------------------------------------------------------------------

def _collect_gradcam(
    model: SincTran,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """Input-gradient saliency for EEG channel importance.

    Computes ``|∂logit/∂X|`` averaged over time to attribute prediction
    importance to individual EEG channels.

    Args:
        model: SincTran in eval mode with weights loaded.
        loader: DataLoader yielding ``(X, y)`` batches.

    Returns:
        cam: ``(N, n_chans)`` per-sample channel importance scores.
        labels: ``(N,)`` integer class indices.
    """
    all_cam: List[np.ndarray] = []
    all_labels: List[int] = []
    model.eval()

    for X, y in loader:
        X_in = X.requires_grad_(True)
        logits = model(X_in)
        model.zero_grad()
        logits.gather(1, logits.argmax(dim=1, keepdim=True)).sum().backward()

        with torch.no_grad():
            cam = F.relu(X_in.grad).mean(dim=2)

        all_cam.append(cam.cpu().numpy())
        all_labels.extend(y.numpy().tolist())

    return np.concatenate(all_cam), np.array(all_labels)


_GRADCAM_SPEC = _HeatmapSpec(
    xlabel="Channel",
    title="Spatial GradCAM - per class",
    cbar_label="Normalised importance",
    cmap=_CMAP_GRADCAM,
)


class SpatialGradCAMVisualizer(_HeatmapVisualizer):
    """Per-subject and fold-average electrode-level GradCAM heatmaps.

    Args:
        channel_names: EEG electrode labels; ``"Ch"``/``"ch"`` prefixes are stripped.
        class_names: Display label for each class index.
    """

    _stem = "gradcam_per_class"

    def __init__(
        self,
        channel_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.class_names = list(class_names) if class_names else None
        self._ch_names_raw = list(channel_names) if channel_names else None
        # Per-subject class-mean matrices: (n_classes, n_chans)
        self._subject_matrices: List[np.ndarray] = []

    def _collect(self, model: SincTran, loader: DataLoader) -> np.ndarray:
        cam, labels = _collect_gradcam(model, loader)
        n_classes = int(labels.max()) + 1
        matrix = np.zeros((n_classes, cam.shape[1]))
        for c in range(n_classes):
            if (mask := labels == c).any():
                matrix[c] = cam[mask].mean(axis=0)
        return matrix

    def _heatmap_kwargs(self, matrix: np.ndarray) -> dict:
        _, n_chans = matrix.shape
        if self._ch_names_raw is not None:
            ch_names = [str(n).replace("Ch", "").replace("ch", "").strip() for n in self._ch_names_raw]
        else:
            ch_names = [str(i) for i in range(n_chans)]
        ticks = _col_ticks(n_chans)
        return dict(col_labels=[ch_names[i] for i in ticks], col_ticks=ticks, spec=_GRADCAM_SPEC)


# ---------------------------------------------------------------------------
# Band attention
# ---------------------------------------------------------------------------

def _collect_band_weights(
    model: SincTran,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample band scores before and after L1 normalization.

    Returns:
        raw: ``(N, n_bands)`` pre-softplus linear scores.
        normed: ``(N, n_bands)`` L1-normalized attention weights.
        labels: ``(N,)`` integer class indices.
    """
    all_raw, all_normed, all_labels = [], [], []
    with torch.no_grad():
        for X, y in loader:
            band_repr = torch.stack([blk(X) for blk in model.sinc_banks], dim=1).mean(dim=(-1, -2))
            raw = model.band_score(band_repr)
            sp = F.softplus(raw)
            normed = sp / (sp.sum(dim=1, keepdim=True) + 1e-6)
            all_raw.append(raw.squeeze(-1).cpu())
            all_normed.append(normed.squeeze(-1).cpu())
            all_labels.append(y.cpu())
    return (
        torch.cat(all_raw).numpy(),
        torch.cat(all_normed).numpy(),
        torch.cat(all_labels).numpy(),
    )


def _save_band_boxplot(
    weights: np.ndarray,
    labels: np.ndarray,
    band_labels: List[str],
    cls_names: List[str],
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    n_bands, n_classes = len(band_labels), len(cls_names)
    colors = _palette(n_classes)

    width = 4.5 / (n_bands * n_classes + (n_bands - 1) * 0.6)
    gap = width * 0.6
    positions_base = np.arange(n_bands) * (n_classes * width + gap)

    fig = plt.figure(figsize=(_FIG_W_BOX, _FIG_H))
    gs = fig.add_gridspec(1, 2, width_ratios=_PLOT_LEGEND_RATIO)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")

    patches = []
    for c in range(n_classes):
        bp = ax.boxplot(
            [weights[labels == c, b] for b in range(n_bands)],
            positions=positions_base + c * width,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=0.9),
            capprops=dict(linewidth=0.9),
            boxprops=dict(linewidth=0.9),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[c])
            patch.set_alpha(0.75)
        patches.append(mpatches.Patch(facecolor=colors[c], alpha=0.75, label=cls_names[c]))

    tick_centers = positions_base + width * (n_classes - 1) / 2
    ax.set_xticks(tick_centers)
    ax.set_xticklabels(band_labels, rotation=0, ha="center", fontsize=_FS_TICK)

    box_half     = (width * 0.85) / 2
    first_center = positions_base[0]
    last_center  = positions_base[-1] + (n_classes - 1) * width
    buffer       = width * 0.5
    ax.set_xlim(first_center - box_half - buffer, last_center + box_half + buffer)

    ax.set_ylabel(ylabel, fontsize=_FS_LABEL, labelpad=10)
    ax.set_title(title, fontsize=_FS_TITLE, pad=10)
    ax.tick_params(axis="both", labelsize=_FS_TICK)

    ax_leg.legend(
        handles=patches,
        title="Class",
        fontsize=_FS_LEGEND,
        title_fontsize=_FS_LEGEND,
        loc="center",
        borderpad=0.5,
        framealpha=0.85,
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def visualize_band_attention(
    model: SincTran,
    loader: DataLoader,
    output_path: Path,
    class_names: Optional[Sequence[str]] = None,
    subject_tag: str = "",
) -> None:
    """Save two band-attention boxplot figures per subject.

    Saves ``band_attn_norm{tag}.svg`` (L1-normalised weights) and
    ``band_attn_lin{tag}.svg`` (raw pre-softplus linear scores).

    Args:
        model: SincTran in eval mode with weights loaded.
        loader: DataLoader yielding ``(X, y)`` batches.
        output_path: Directory for output figures.
        class_names: Display label for each class index.
        subject_tag: Appended to output filenames only.
    """
    band_labels = [f"{lo}–{hi} Hz" for (lo, hi) in model.DEFAULT_EEG_BANDS]
    raw_weights, normed_weights, labels = _collect_band_weights(model, loader)
    n_classes = int(labels.max().item()) + 1
    cls_names = list(class_names) if class_names else [str(c) for c in range(n_classes)]
    tag = f"_{subject_tag}" if subject_tag else ""

    for weights, ylabel, title, fname in (
        (normed_weights, "Attention weight", "Band attention - Normalized",    f"band_attn_norm{tag}.svg"),
        (raw_weights,    "Linear score",     "Band attention - Linear Scores", f"band_attn_lin{tag}.svg"),
    ):
        _save_band_boxplot(weights, labels, band_labels, cls_names, ylabel, title, output_path / fname)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def visualize_cls_umap(
    model: SincTran,
    loader: DataLoader,
    output_path: Path,
    class_names: Optional[Sequence[str]] = None,
    subject_tag: str = "",
    umap_kwargs: Optional[Dict] = None,
) -> None:
    """Save two UMAP figures: CLS-token embeddings and MLP first-layer activations.

    Saves ``cls_umap{tag}.svg`` and ``mlp_umap{tag}.svg``.

    Args:
        model: SincTran in eval mode with ``use_cls=True``.
        loader: DataLoader yielding ``(X, y)`` batches.
        output_path: Directory for output figures.
        class_names: Display label for each class index.
        subject_tag: Appended to output filenames only.
        umap_kwargs: Forwarded to ``umap.UMAP``; overrides defaults.

    Raises:
        ValueError: If ``model.use_cls`` is False.
    """
    if not model.use_cls:
        raise ValueError("CLS UMAP requires use_cls=True.")

    all_cls, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            all_cls.append(model.features(X).cpu().numpy())
            all_labels.append(y.numpy())
    cls_emb = np.concatenate(all_cls)
    labels = np.concatenate(all_labels)

    mlp_emb, mlp_labels = [], []
    handle = model.head[1].register_forward_hook(lambda _, __, out: mlp_emb.append(out.detach().cpu()))
    with torch.no_grad():
        for X, y in loader:
            model(X)
            mlp_labels.extend(y.numpy().tolist())
    handle.remove()
    mlp_emb = np.concatenate(mlp_emb)
    mlp_labels = np.array(mlp_labels)

    n_classes = int(labels.max()) + 1
    cls_names = list(class_names) if class_names else [str(c) for c in range(n_classes)]
    tag = f"_{subject_tag}" if subject_tag else ""

    kwargs = {"n_neighbors": 15, "min_dist": 0.1, "random_state": 42, "n_jobs": 1}
    if umap_kwargs:
        kwargs.update(umap_kwargs)

    for emb, lbl, title, fname in (
        (cls_emb, labels,     "UMAP - CLS Token",     f"cls_umap{tag}.svg"),
        (mlp_emb, mlp_labels, "UMAP - MLP Embedding", f"mlp_umap{tag}.svg"),
    ):
        proj = UMAP(**kwargs).fit_transform(emb)

        fig = plt.figure(figsize=(_FIG_W_SCATTER, _FIG_H))
        gs = fig.add_gridspec(1, 2, width_ratios=_PLOT_LEGEND_RATIO)
        ax = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])
        ax_leg.axis("off")

        colors = _palette(len(cls_names))
        handles = []
        for c, name in enumerate(cls_names):
            pts = proj[lbl == c]
            sc = ax.scatter(
                pts[:, 0], pts[:, 1],
                s=420,
                alpha=0.75,
                color=colors[c],
                marker=_MARKERS[c % len(_MARKERS)],
                linewidths=0.5,
                edgecolors="white",
            )
            handles.append(plt.Line2D(
                [0], [0],
                marker=_MARKERS[c % len(_MARKERS)],
                color="w",
                markerfacecolor=colors[c],
                markersize=20,
                label=name,
            ))

        ax.set_xlabel("UMAP-1", fontsize=_FS_LABEL)
        ax.set_ylabel("UMAP-2", fontsize=_FS_LABEL)
        ax.set_title(title, fontsize=_FS_TITLE)
        ax.tick_params(labelsize=_FS_TICK)

        ax_leg.legend(
            handles=handles,
            title="Class",
            fontsize=_FS_LEGEND,
            title_fontsize=_FS_LEGEND,
            loc="center",
            markerscale=1.0,
            borderpad=0.5,
            framealpha=0.85,
        )

        fig.tight_layout()
        fig.savefig(output_path / fname, bbox_inches="tight", dpi=300)
        plt.close(fig)