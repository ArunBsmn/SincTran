"""Training utilities for EEG model pipeline: reproducibility, timing, FLOPs, and metadata."""

import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml


def set_all_seeds(seed: int) -> None:
    """
    Args:
        seed (int): Global random seed.
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """
    Args:
        seconds (float): Elapsed time in seconds.

    Returns:
        str: Human-readable string, e.g. "2h 15m 30s".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def calculate_model_flops(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> Optional[float]:
    """
    Args:
        model (torch.nn.Module): PyTorch model.
        input_shape (Tuple[int, ...]): Input tensor shape (batch, channels, time).
        device (str): Device for calculation.

    Returns:
        FLOPs in GFLOPs, or None if calculation fails.
    """
    try:
        from calflops import calculate_flops

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            flops, _, _ = calculate_flops(
                model=model,
                input_shape=input_shape,
                output_as_string=False,
                print_results=False,
            )
        return flops / 1e9
    except Exception as e:
        warnings.warn(f"Could not calculate FLOPs: {e}")
        return None


def _make_yaml_serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _make_yaml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_yaml_serializable(item) for item in obj]
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "__class__"):
        return obj.__class__.__name__
    return str(obj)


def save_model_summary(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    save_path: str,
) -> None:
    """
    Args:
        model (torch.nn.Module): PyTorch model.
        input_shape (Tuple[int, ...]): Input tensor shape (batch, channels, time).
        save_path (str): Destination file path.
    """
    try:
        from torchinfo import summary

        model_stats = summary(
            model,
            input_size=input_shape,
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
        )
        device = str(next(model.parameters()).device)
        flops_gflops = calculate_model_flops(model, input_shape, device)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write("=" * 100 + "\n")
            f.write("MODEL SUMMARY\n")
            f.write("=" * 100 + "\n\n")
            f.write(str(model_stats))
            f.write("\n\n" + "=" * 100 + "\n")
            f.write(
                f"FLOPs: {flops_gflops:.2f} GFLOPs\n"
                if flops_gflops is not None
                else "FLOPs: Could not be calculated\n"
            )

        print(f"✓ Saved model summary to {save_path}")
        if flops_gflops is not None:
            print(f"  FLOPs: {flops_gflops:.2f} GFLOPs")

    except Exception as e:
        warnings.warn(f"Could not save model summary: {e}")


def save_metadata(
    configs: Dict[str, Dict],
    metadata_root: str,
    task: str,
    training_times: Optional[Dict[str, str]] = None,
) -> None:
    """
    Args:
        configs (Dict[str, Dict]): Config dicts keyed by name ("data", "model", "loss", "training").
        metadata_root (str): Root directory for saved metadata.
        task (str): Task identifier used in data config filename.
        training_times (Dict[str, str], optional): Pre-formatted subject ID -> time strings.
    """
    os.makedirs(metadata_root, exist_ok=True)

    for name, config in configs.items():
        config = config.copy()

        if name == "data" and training_times is not None:
            config["training_time"] = training_times

        if name == "model" and "eeg_bands" in config:
            config["eeg_bands"] = {
                f"{low}-{high}Hz": n_heads
                for (low, high), n_heads in config["eeg_bands"].items()
            }

        config = _make_yaml_serializable(config)
        filename = f"{name}_{task}.yaml" if name in ("data", "fsl") else f"{name}.yaml"

        with open(os.path.join(metadata_root, filename), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"✓ Saved experiment metadata to {metadata_root}")