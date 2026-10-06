"""Centralised logging setup for the enforcement_app GUI.

Frozen exe (PyInstaller) hides stdout/stderr → operators can't see anything
the standard `logger.info / .warning / .error` calls produce. We attach a
RotatingFileHandler to the configured log path so every event lands on disk
alongside `enforce_watchdog.log` and `vlm_decisions.csv`.

This module is intentionally thin: just `configure_logging(...)` taking the
target path + level. Idempotent — safe to call multiple times during reload.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_OUR_HANDLER_TAG = "enforce_app_log_handlers"


def _strip_our_handlers(logger: logging.Logger) -> None:
    """Remove handlers we previously installed; leave others (e.g. test rigs) alone."""
    for h in list(logger.handlers):
        if getattr(h, _OUR_HANDLER_TAG, False):
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)


def configure_logging(
    log_path: Path,
    level: str = "INFO",
    *,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Attach a rotating file handler + stderr handler to the root logger.

    - `log_path` parent directory is created if missing
    - `level` accepts standard names (DEBUG / INFO / WARNING / ERROR)
    - Idempotent: previous handlers we installed are removed before re-adding
    """
    log_path = Path(log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Permission / read-only fs — fall through; FileHandler will raise
        # below with a clearer error.
        pass

    root = logging.getLogger()
    _strip_our_handlers(root)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)
    setattr(file_handler, _OUR_HANDLER_TAG, True)
    root.addHandler(file_handler)

    # Keep stderr handler so dev runs (`python -m enforcement_app.main`) still
    # show output in the terminal. Frozen exe has no console so this is a no-op.
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(numeric_level)
    setattr(stderr_handler, _OUR_HANDLER_TAG, True)
    root.addHandler(stderr_handler)
