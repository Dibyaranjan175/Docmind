"""
DocMind Logger
Structured, colour-aware logging with file + console output.
"""

import logging
import os
import sys
from pathlib import Path

from src.config import cfg


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured for DocMind."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — return as-is to avoid duplicate handlers
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ── File handler ─────────────────────────────────────────────────────────
    logs_path = Path(cfg.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_path / "docmind.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
