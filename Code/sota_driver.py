"""Central driver for imagined-speech SOTA benchmarking: ASU and BCI Competition Track 3."""
from __future__ import annotations

import gc
import torch


def clear_memory() -> None:
    """Flush Python garbage collector and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":

    from sota_asu import main_asu_sota
    from sota_bci import main_bci_sota

    VALID_MODELS = {
        "eegnet",
        "eegnex",
        "shallow",
        "ctnet",
        "atcnet",
        "eegconformer",
        "msvt",
    }

    RUN_CFG: dict = {
        "dataset":    "asu",      # "asu" or "bci"
        "model_name": "shallow",  # see VALID_MODELS above
        "task":       "n2",       # "n1"–"n4"; ASU only, ignored for BCI
        "subjects":   None,       # None = all; List[str] for ASU, List[int] for BCI
        "debug":      False,
        "epochs":     100,
        "n_splits":   10,
        "early_stop": True,
    }

    if RUN_CFG["dataset"] not in ("asu", "bci"):
        raise ValueError(f"Unknown dataset: {RUN_CFG['dataset']!r}. Choose 'asu' or 'bci'.")

    if RUN_CFG["model_name"] not in VALID_MODELS:
        raise ValueError(f"Unknown model_name: {RUN_CFG['model_name']!r}. Choose from {sorted(VALID_MODELS)}.")

    clear_memory()

    if RUN_CFG["dataset"] == "asu":
        main_asu_sota(
            model_name = RUN_CFG["model_name"],
            task       = RUN_CFG["task"],
            debug_mode = RUN_CFG["debug"],
            subjects   = RUN_CFG["subjects"],
            num_epochs = RUN_CFG["epochs"],
            n_splits   = RUN_CFG["n_splits"],
            early_stop = RUN_CFG["early_stop"],
        )
    else:
        main_bci_sota(
            model_name = RUN_CFG["model_name"],
            debug_mode = RUN_CFG["debug"],
            subjects   = RUN_CFG["subjects"],
            num_epochs = RUN_CFG["epochs"],
            n_splits   = RUN_CFG["n_splits"],
            early_stop = RUN_CFG["early_stop"],
        )