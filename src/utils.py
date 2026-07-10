import os
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Make experiments as reproducible as possible.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Select GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path: str) -> None:
    """
    Save dictionaries/lists as JSON.
    """
    ensure_dir(os.path.dirname(path))

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_numpy(array: np.ndarray, path: str) -> None:
    """
    Save NumPy arrays.
    """
    ensure_dir(os.path.dirname(path))
    np.save(path, array)
