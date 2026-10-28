import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> None:
    logger = logging.getLogger("calchem")
    logger.setLevel(level)

    logger.handlers.clear()

    if use_rich:
        handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
    else:
        handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str = "calchem") -> logging.Logger:
    return logging.getLogger(name)
