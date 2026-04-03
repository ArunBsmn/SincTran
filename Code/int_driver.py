"""Central driver for SincTran interpretation pipelines."""
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

    from int_asu import main_asu_interpret
    from int_bci import main_bci_interpret

    RUN_CFG: dict = {
        "dataset":    "bci",   # "asu" or "bci"
        "task":       "n4",    # ASU only: one of "n1", "n2", "n3", "n4"; ignored for BCI
        "subject_id": 4,       # int for BCI (1–15); str for ASU (e.g. "sub_11b"); None → all subjects
        "fold":       None,    # 1-indexed fold; None → all folds (fold-average written after last fold)
        "n_splits":   10,      # must match the value used during training
        "debug":      False,
    }

    if RUN_CFG["dataset"] not in ("asu", "bci"):
        raise ValueError(f"Unknown dataset: {RUN_CFG['dataset']!r}. Choose 'asu' or 'bci'.")

    clear_memory()

    if RUN_CFG["dataset"] == "asu":
        main_asu_interpret(
            task       = RUN_CFG["task"],
            subject_id = RUN_CFG["subject_id"],
            fold       = RUN_CFG["fold"],
            debug_mode = RUN_CFG["debug"],
            n_splits   = RUN_CFG["n_splits"],
        )
    else:
        main_bci_interpret(
            subject_id = RUN_CFG["subject_id"],
            fold       = RUN_CFG["fold"],
            debug_mode = RUN_CFG["debug"],
            n_splits   = RUN_CFG["n_splits"],
        )