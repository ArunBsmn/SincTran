"""Central driver for EEG training pipelines: ASU speech imagery and BCI Competition Track 3."""
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

    from main_asu import main_asu
    from main_bci import main_bci

    RUN_CFG: dict = {
        "dataset":    "asu",   # "asu" or "bci"
        "task":       "n2",    # one of "n1", "n2", "n3", "n4"; ASU only, ignored for BCI
        "subjects":   None,    # None = all; List[str] for ASU, List[int] for BCI
        "debug":      False,
        "epochs":     100,
        "n_splits":   10,
        "early_stop": True,
    }

    if RUN_CFG["dataset"] not in ("asu", "bci"):
        raise ValueError(f"Unknown dataset: {RUN_CFG['dataset']!r}. Choose 'asu' or 'bci'.")

    clear_memory()

    if RUN_CFG["dataset"] == "asu":
        main_asu(
            RUN_CFG["task"],
            debug_mode = RUN_CFG["debug"],
            subjects   = RUN_CFG["subjects"],
            num_epochs = RUN_CFG["epochs"],
            n_splits   = RUN_CFG["n_splits"],
            early_stop = RUN_CFG["early_stop"],
        )
    else:
        main_bci(
            RUN_CFG["task"],
            debug_mode = RUN_CFG["debug"],
            subjects   = RUN_CFG["subjects"],
            num_epochs = RUN_CFG["epochs"],
            n_splits   = RUN_CFG["n_splits"],
            early_stop = RUN_CFG["early_stop"],
        )