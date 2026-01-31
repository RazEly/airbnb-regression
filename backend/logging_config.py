"""Centralized logging configuration for the Airbnb regression project."""

import logging
import os


def configure_logging(level=None):
    """
    Configure logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses LOG_LEVEL env variable or defaults to INFO.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress noisy third-party libraries
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    return logging.getLogger(__name__)
