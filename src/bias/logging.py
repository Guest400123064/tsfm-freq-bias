from __future__ import annotations

import logging

import trackio
from rich.console import Console
from rich.logging import RichHandler

registry: dict[str, tuple[logging.Logger, RichHandler]] = {}

console = Console()


def get_logger(name, level=logging.INFO):
    handler = RichHandler(level, console=console, rich_tracebacks=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Each logger owns its handler; without this, records also propagate to
    # ancestor loggers and print twice.
    logger.propagate = False
    logger.debug(f"Logger <{name}> initialized with level <{level}>.")

    registry[name] = (logger, handler)
    return logger


def set_level(level=logging.INFO):
    for logger, handler in registry.values():
        logger.setLevel(level)
        handler.setLevel(level)
