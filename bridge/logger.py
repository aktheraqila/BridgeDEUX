"""
BridgeDEUX

Central Logging Framework

Every production module must obtain its logger from here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bridge.config import ProjectConfig


class BridgeLogger:
    """
    Creates standardized loggers for BridgeDEUX.
    """

    LOG_FORMAT = (
        "[%(asctime)s] "
        "[%(levelname)s] "
        "[%(name)s] "
        "%(message)s"
    )

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:

        ProjectConfig.initialize()

        logger = logging.getLogger(name)

        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt=cls.LOG_FORMAT,
            datefmt=cls.DATE_FORMAT
        )

        log_file = (
            ProjectConfig.LOG_DIR /
            ProjectConfig.LOG_FILE_NAME
        )

        file_handler = logging.FileHandler(
            filename=log_file,
            encoding="utf-8"
        )

        console_handler = logging.StreamHandler()

        file_handler.setFormatter(formatter)

        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        logger.addHandler(console_handler)

        logger.propagate = False

        return logger