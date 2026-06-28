from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from genai_template.config.settings import LOG_DIR, LOG_FILE, LOG_LEVEL


def configure_logging() -> None:
    """
    Configure application-wide logging.

    Call once during application startup.
    """

    LOG_DIR.mkdir(exist_ok=True)
    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    root_logger.setLevel(LOG_LEVEL)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
    )

    # File handler
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] "
            "[%(name)s] "
            "[%(levelname)s] "
            "[%(module)s:%(lineno)d] "
            "%(message)s"
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
