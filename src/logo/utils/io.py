"""Basic I/O helpers shared by every stage: JSON/YAML, seeding, logging."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml

PathLike = Union[str, Path]

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _to_builtin(o: Any) -> Any:
    """json.dump `default` hook: converts NumPy and Path objects."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def load_json(p: PathLike) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(o: Any, p: PathLike) -> None:
    """Writes compact UTF-8 JSON atomically (tmp file + replace); creates parent dirs."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(o, f, ensure_ascii=False, separators=(",", ":"), default=_to_builtin)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_yaml(p: PathLike) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def set_seed(s: int) -> None:
    """Seeds Python, NumPy and (if installed) PyTorch CPU/CUDA."""
    s = int(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_logger(name: str, logfile: Optional[PathLike] = None) -> logging.Logger:
    """Logger that writes to stdout and appends to `logfile`.

    Calling it again with the same name/logfile does not add duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter(_LOG_FORMAT)

    has_console = any(
        type(h) is logging.StreamHandler and getattr(h, "stream", None) is sys.stdout
        for h in logger.handlers
    )
    if not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if logfile is not None:
        path = Path(logfile).resolve()
        has_file = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path
            for h in logger.handlers
        )
        if not has_file:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger
