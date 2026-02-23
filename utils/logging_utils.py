# utils/logging_utils.py
from __future__ import annotations

import logging
from pathlib import Path


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_root_logger(logs_dir: Path, name: str = "spurious_pipeline") -> logging.Logger:
    """
    Configures a root logger with:
      - console handler
      - file handler at outputs/logs/<name>.log
    Safe to call multiple times.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = _make_formatter()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(logs_dir / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_dimension_logger(base_logger_name: str, logs_dir: Path, dim_id: int) -> logging.Logger:
    """
    Returns a per-dimension file logger writing to outputs/logs/dim_<id>.log.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    lname = f"{base_logger_name}.dim_{dim_id}"
    logger = logging.getLogger(lname)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = _make_formatter()

    fh = logging.FileHandler(logs_dir / f"dim_{dim_id}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger