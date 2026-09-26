"""Shared utilities: reproducibility, structured logging, metric tracking."""

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """Load the YAML config used by both the manual and TRL training scripts."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed all RNGs so manual and TRL runs are comparable under identical sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JsonlLogger:
    """Append-only per-step metric logger. One JSON object per line.

    Using jsonl (not a dict-of-lists in memory) means a crashed run still
    leaves a readable partial log, and compare_runs.py can stream large logs
    without loading everything at once.
    """

    def __init__(self, log_path: str):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any previous run with the same name.
        self.path.write_text("")

    def log(self, step: int, **metrics: Any) -> None:
        record = {"step": step, **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def read(log_path: str) -> list[dict]:
        records = []
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
