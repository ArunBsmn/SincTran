"""
EEG training and evaluation utilities.
"""
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: DataLoader,
    num_epochs: int,
    config: Dict,
) -> Dict:
    """
    Args:
        model (nn.Module): Model to train.
        criterion (nn.Module): Loss function.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader, optional): Validation data; falls back to test_loader if None.
        test_loader (DataLoader): Test data for final evaluation.
        num_epochs (int): Number of training epochs.
        config (Dict): Training config with keys: device, optimizer_class, optimizer_kwargs,
            scheduler_class, scheduler_kwargs, monitor_metric, early_stopping, use_amp,
            grad_clip, subject_name, model_path, metrics_root.
    """
    monitor_metric = config.get('monitor_metric', 'f1')
    valid_metrics  = {'loss', 'acc', 'precision', 'recall', 'f1', 'kappa'}
    if monitor_metric not in valid_metrics:
        raise ValueError(f"Unknown monitor_metric: '{monitor_metric}'. Must be one of {valid_metrics}.")
    minimize        = monitor_metric in {'loss'}
    es              = config.get('early_stopping', {'enabled': False})
    es_enabled      = es.get('enabled', False)
    es_patience     = es.get('patience', 10)
    es_min_delta    = es.get('min_delta', 0.0)
    per_batch_sched = config.get('scheduler_class') in {
        torch.optim.lr_scheduler.OneCycleLR,
        torch.optim.lr_scheduler.CyclicLR,
    }
    cfg = {
        'device':           config.get('device', 'cpu'),
        'optimizer_class':  config['optimizer_class'],
        'optimizer_kwargs': config.get('optimizer_kwargs', {}),
        'scheduler_class':  config.get('scheduler_class'),
        'scheduler_kwargs': config.get('scheduler_kwargs', {}),
        'monitor_metric':   monitor_metric,
        'minimize':         minimize,
        'es_enabled':       es_enabled,
        'es_patience':      es_patience,
        'es_min_delta':     es_min_delta,
        'per_batch_sched':  per_batch_sched,
        'grad_clip':        config.get('grad_clip'),
        'use_amp':          config.get('use_amp', False),
        'subject_name':     config.get('subject_name'),
        'cm_tag':           config.get('cm_tag', config.get('subject_name')),
        'model_path':       config.get('model_path'),
        'metrics_root':     config.get('metrics_root', 'results/metrics'),
    }

    model.to(cfg['device'])
    optimizer = cfg['optimizer_class'](model.parameters(), **cfg['optimizer_kwargs'])

    scheduler_kwargs = dict(cfg['scheduler_kwargs'])
    if cfg['scheduler_class'] is torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler_kwargs.setdefault('mode', 'min' if minimize else 'max')
        # Mirror RLP's own threshold/patience into ES so both mechanisms agree on "improvement".
        es_min_delta = scheduler_kwargs.get('threshold', es_min_delta)
        es_patience  = scheduler_kwargs.get('patience', es_patience) + 2
        print(f"ES params overridden from RLP — patience: {es_patience}, min_delta: {es_min_delta}")

    scheduler = (
        cfg['scheduler_class'](optimizer, **scheduler_kwargs)
        if cfg['scheduler_class'] else None
    )

    # Defer ES until the scheduler has exhausted LR decay; patience accumulated
    # during decay is discarded at the moment ES arms (see counter reset below).
    es_min_lr = cfg['scheduler_kwargs'].get('min_lr', cfg['scheduler_kwargs'].get('eta_min', None))
    es_active = es_min_lr is None

    best_metric      = float('inf') if minimize else -float('inf')
    best_model       = None
    best_val_metrics = {}
    patience_counter = 0
    history          = defaultdict(list)
    eval_loader      = val_loader if val_loader is not None else test_loader

    for epoch in range(1, num_epochs + 1):
        train_metrics = _train_epoch(model, criterion, train_loader, optimizer, cfg, epoch, scheduler)
        val_metrics   = evaluate_model(model, criterion, eval_loader, cfg['device'])

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics[cfg['monitor_metric']])
            elif not per_batch_sched:
                scheduler.step()
        lr = scheduler.optimizer.param_groups[0]['lr'] if scheduler else cfg['optimizer_kwargs']['lr']

        monitored = val_metrics[cfg['monitor_metric']]
        improved  = (
            monitored < best_metric - es_min_delta if minimize
            else monitored > best_metric + es_min_delta
        )
        if improved:
            best_metric      = monitored
            best_val_metrics = val_metrics
            best_model       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if cfg['model_path'] is None:
                raise ValueError("model_path must be set in config to save the best model.")
            path = Path(cfg['model_path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_model, path)
        else:
            patience_counter += 1

        # Arm ES once LR reaches its floor; discard patience from the decay phase.
        if not es_active and lr <= es_min_lr + 1e-12:
            es_active        = True
            patience_counter = 0

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            if k not in ('preds', 'targets'):
                history[f"val_{k}"].append(v)
        history['lr'].append(lr)

        nan = float('nan')
        print(f"Epoch {epoch}/{num_epochs} | LR: {lr:.2e}")
        for tag, m in (("Train", train_metrics), ("Valid", val_metrics)):
            print(
                f"  {tag} — "
                f"L: {m.get('loss', nan):.4f} | "
                f"A: {m.get('acc', nan):.4f} | "
                f"P: {m.get('precision', nan):.4f} | "
                f"R: {m.get('recall', nan):.4f} | "
                f"F1: {m.get('f1', nan):.4f} | "
                f"κ: {m.get('kappa', nan):.4f}"
            )
        if improved:
            print(f"  ✓ Best {cfg['monitor_metric']}: {monitored:.4f}")
        elif es_enabled and es_active:
            if patience_counter >= es_patience:
                print(f"  Early stopping at epoch {epoch}")
                print()
                break
            else:
                print(f"  ES patience: {patience_counter}/{es_patience}")
        print()

    _finalize_and_save(model, criterion, dict(history), cfg, val_loader, test_loader, best_val_metrics)
    return dict(history)


def _train_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    cfg: Dict,
    epoch: int,
    scheduler: Optional[Any],
) -> Dict[str, float]:
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    tgt_list, pred_list = [], []

    device_type = str(cfg['device']).split(':')[0]
    # AMP is only stable on CUDA; silently disable on CPU to avoid autocast warnings.
    use_amp = cfg['use_amp'] and device_type == 'cuda'
    scaler  = torch.amp.GradScaler(device_type, enabled=use_amp)

    for x, y in loader:
        x, y = x.to(cfg['device']), y.to(cfg['device'])
        optimizer.zero_grad()
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            feat = model(x)
            loss, pred = criterion(feat, y)
        scaler.scale(loss).backward()
        if cfg['grad_clip'] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            # p.grad is None for parameters that received no gradient this step.
            if any(
                p.grad is not None
                and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                warnings.warn(f"Unstable gradients at epoch {epoch}")
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and cfg['per_batch_sched']:
            scheduler.step()
        loss_sum += loss.item()
        correct  += (pred == y).sum().item()
        total    += y.size(0)
        tgt_list.append(y.cpu().numpy())
        pred_list.append(pred.cpu().numpy())

    tgt  = np.concatenate(tgt_list)
    pred = np.concatenate(pred_list)
    precision, recall, f1, _ = precision_recall_fscore_support(tgt, pred, average='macro', zero_division=0)
    return {
        'loss':      loss_sum / len(loader),
        'acc':       correct / total,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'kappa':     cohen_kappa_score(tgt, pred),
    }


def evaluate_model(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    tgt_list, pred_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feat = model(x)
            loss, pred = criterion(feat, y)
            loss_sum += loss.item()
            correct  += (pred == y).sum().item()
            total    += y.size(0)
            tgt_list.append(y.cpu().numpy())
            pred_list.append(pred.cpu().numpy())
    targets = np.concatenate(tgt_list)
    preds   = np.concatenate(pred_list)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    return {
        'loss':      loss_sum / len(loader),
        'acc':       correct / total,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'kappa':     cohen_kappa_score(targets, preds),
        'preds':     preds,
        'targets':   targets,
    }


def _finalize_and_save(
    model: nn.Module,
    criterion: nn.Module,
    history: Dict,
    cfg: Dict,
    val_loader: Optional[DataLoader],
    test_loader: DataLoader,
    best_val_metrics: Dict,
) -> None:
    if cfg['subject_name']:
        save_metrics(history, Path(cfg['metrics_root']) / f"{cfg['subject_name']}_train_val_metrics.csv")

    if val_loader is not None:
        model.load_state_dict(torch.load(cfg['model_path'], map_location='cpu'))
        test_metrics = evaluate_model(model, criterion, test_loader, cfg['device'])
    else:
        test_metrics = best_val_metrics

    nan = float('nan')
    print(
        f"Test — "
        f"L: {test_metrics.get('loss', nan):.4f} | "
        f"A: {test_metrics.get('acc', nan):.4f} | "
        f"P: {test_metrics.get('precision', nan):.4f} | "
        f"R: {test_metrics.get('recall', nan):.4f} | "
        f"F1: {test_metrics.get('f1', nan):.4f} | "
        f"κ: {test_metrics.get('kappa', nan):.4f}"
    )
    if cfg['subject_name']:
        root        = Path(cfg['metrics_root'])
        class_names = [str(c) for c in np.unique(test_metrics['targets'])]
        plot_confusion_matrix(
            test_metrics['targets'], test_metrics['preds'],
            class_names, cfg['cm_tag'],
            root.parent / 'confusion_matrices' / f"{cfg['cm_tag']}_cm.png",
        )
        save_test_metrics(
            cfg['subject_name'],
            {k: test_metrics[k] for k in ('loss', 'acc', 'f1', 'kappa')},
            root / 'test_metrics.csv',
        )


def save_metrics(metrics: Dict[str, List[float]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.insert(0, 'epoch', range(1, len(df) + 1))
    df.to_csv(path, index=False)
    print(f"Saved metrics to {path}")


def save_test_metrics(
    subject_name: str,
    test_metrics: Dict[str, float],
    path: Union[str, Path],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{'subject': subject_name, **test_metrics}])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"Saved test metrics to {path}")


def plot_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    subject_name: str,
    path: Union[str, Path],
) -> None:
    cm = confusion_matrix(targets, predictions, normalize='true')
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticks(ticks); ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(f'Confusion Matrix — {subject_name}', fontsize=11, fontweight='bold')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CM to {path}")