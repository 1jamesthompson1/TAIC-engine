"""Logging configuration for the TAIC engine.

This module provides centralized logging configuration with support for
different log levels and output destinations.
"""

import logging
import sys


def configure_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure logging for the engine.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to INFO.
        log_file: Optional path to log file. If provided, logs are written
            to both console and file.
        format_string: Optional custom format string. Defaults to a standard
            format with timestamp, level, and message.

    Returns:
        logging.Logger: Configured root logger instance.

    Example:
        >>> logger = configure_logging(log_level="DEBUG")
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: The name of the logger, typically __name__ from the calling module.

    Returns:
        logging.Logger: A logger instance configured with the engine's settings.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module starting")
    """
    return logging.getLogger(name)
