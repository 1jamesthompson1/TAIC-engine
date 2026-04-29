"""Logging configuration for the TAIC engine.

This module provides centralized logging configuration with support for
different log levels and output destinations.
"""

import logging
import sys


class EngineLogger(logging.Logger):
    """Custom logger with additional formatting methods for consistent output."""

    def welcome(
        self,
        title: str,
        details: dict[str, str] | None = None,
        width: int = 80,
    ) -> None:
        """Log a formatted welcome/section header message.

        Creates a visually distinguished section header with optional details.
        All output is combined into a single logger call.

        Args:
            title: Main title for the section.
            details: Optional dict of key-value pairs to display below title.
            width: Width of the formatted output box (default 80).

        Example:
            >>> logger.welcome(
            ...     "Scraping Recommendations",
            ...     {
            ...         "Output directory": "/path/to/output",
            ...         "Base URL": "https://example.com",
            ...         "Current reports": "42",
            ...     }
            ... )
        """
        message = "\n" + "=" * width + "\n"
        message += "|" * width + "\n"
        message += " " * ((width - len(title)) // 2) + title + "\n"
        message += "|" * width + "\n"

        if details:
            for key, value in details.items():
                message += f"  {key}: {value}\n"

        message += "=" * width

        self.info(message)


def configure_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
    suppress_azure_logging: bool = True,
) -> logging.Logger:
    """Configure logging for the engine.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to INFO.
        log_file: Optional path to log file. If provided, logs are written
            to both console and file.
        format_string: Optional custom format string. Defaults to a standard
            format with timestamp, level, and message.
        suppress_azure_logging: If True, suppresses verbose Azure SDK logging
            (default True).

    Returns:
        logging.Logger: Configured root logger instance.

    Example:
        >>> logger = configure_logging(log_level="DEBUG")
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
    """
    # Set the custom logger class globally
    logging.setLoggerClass(EngineLogger)

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

    # Suppress verbose Azure SDK logging
    if suppress_azure_logging:
        logging.getLogger("azure").setLevel(logging.WARNING)
        logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
            logging.WARNING
        )

    # Suppress noisy HTTP client request logs from SDK dependencies.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Returns an EngineLogger instance with custom methods for consistent
    output formatting throughout the application.

    Args:
        name: The name of the logger, typically __name__ from the calling module.

    Returns:
        logging.Logger: An EngineLogger instance with additional methods like
            welcome() for formatted output sections.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module starting")
        >>> logger.welcome("Processing Reports", {"Count": "42"})
    """
    # Ensure the custom logger class is set
    if logging.getLoggerClass() is not EngineLogger:
        logging.setLoggerClass(EngineLogger)

    return logging.getLogger(name)
