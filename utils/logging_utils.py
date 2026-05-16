import logging
import os
import sys
from datetime import datetime
from typing import Optional


_LOGGERS: dict = {}


def setup_logger(
    name: str = "fedlease",
    log_dir: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """Configure and return a named logger with optional file output."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = "fedlease") -> logging.Logger:
    """Retrieve an existing logger or create a default one."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    return setup_logger(name)
