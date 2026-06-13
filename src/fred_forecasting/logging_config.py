"""Centralized logging configuration."""

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once, with a concise, timestamped format.

    Args:
        level: Logging level name or value. Defaults to the ``LOG_LEVEL``
            environment variable, or ``INFO`` if unset.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = level if level is not None else os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    configure_logging()
    return logging.getLogger(name)
