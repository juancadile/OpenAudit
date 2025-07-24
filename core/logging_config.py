"""
Centralized logging configuration for OpenAudit
Provides structured logging with appropriate levels and formatting
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def get_logging_config(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> Dict[str, Any]:
    """
    Get logging configuration dictionary.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Logging configuration dictionary
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(levelname)s - %(name)s - %(message)s",
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "file": "%(filename)s", "line": %(lineno)d, "message": "%(message)s"}',
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {},
        "loggers": {
            "openaudit": {
                "level": log_level,
                "handlers": [],
                "propagate": False,
            },
            "core": {
                "level": log_level,
                "handlers": [],
                "propagate": False,
            },
            "tests": {
                "level": log_level,
                "handlers": [],
                "propagate": False,
            },
        },
        "root": {
            "level": log_level,
            "handlers": [],
        },
    }

    handlers = []

    # Console handler
    if log_to_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        }
        handlers.append("console")

    # File handlers
    if log_to_file:
        # General application log
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "openaudit.log"),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf8",
        }
        handlers.append("file")

        # Error-only log
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "errors.log"),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf8",
        }
        handlers.append("error_file")

        # Experiment-specific log
        config["handlers"]["experiment_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": str(LOGS_DIR / "experiments.log"),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf8",
        }

    # Assign handlers to loggers
    for logger_name in ["openaudit", "core", "tests"]:
        config["loggers"][logger_name]["handlers"] = handlers

    config["root"]["handlers"] = handlers

    return config


def setup_logging(
    log_level: str = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    force_setup: bool = False,
) -> logging.Logger:
    """
    Setup centralized logging for OpenAudit.
    
    Args:
        log_level: Override log level from environment
        log_to_file: Whether to log to files
        log_to_console: Whether to log to console
        force_setup: Force reconfiguration even if already setup
        
    Returns:
        Configured logger instance
    """
    # Check if already configured
    if not force_setup and logging.getLogger("openaudit").handlers:
        return logging.getLogger("openaudit")

    # Get log level from environment or parameter
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        log_level = "INFO"

    # Get logging configuration
    config = get_logging_config(
        log_level=log_level,
        log_to_file=log_to_file,
        log_to_console=log_to_console,
    )

    # Apply configuration
    logging.config.dictConfig(config)

    # Get the main logger
    logger = logging.getLogger("openaudit")

    # Log startup message
    logger.info("OpenAudit logging initialized")
    logger.debug(f"Log level: {log_level}")
    logger.debug(f"Log to file: {log_to_file}")
    logger.debug(f"Log to console: {log_to_console}")
    logger.debug(f"Logs directory: {LOGS_DIR}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    # Ensure logging is setup
    if not logging.getLogger("openaudit").handlers:
        setup_logging()

    return logging.getLogger(f"openaudit.{name}")


def get_experiment_logger() -> logging.Logger:
    """
    Get a logger specifically for experiment data.
    
    Returns:
        Experiment logger instance
    """
    logger = logging.getLogger("openaudit.experiments")
    
    # Add experiment-specific handler if not already present
    if not any(h.get_name() == "experiment_file" for h in logger.handlers):
        config = get_logging_config()
        if "experiment_file" in config["handlers"]:
            handler_config = config["handlers"]["experiment_file"]
            handler = logging.handlers.RotatingFileHandler(
                filename=handler_config["filename"],
                maxBytes=handler_config["maxBytes"],
                backupCount=handler_config["backupCount"],
                encoding=handler_config["encoding"],
            )
            handler.setLevel(getattr(logging, handler_config["level"]))
            
            # Set JSON formatter
            formatter = logging.Formatter(
                config["formatters"]["json"]["format"],
                datefmt=config["formatters"]["json"]["datefmt"],
            )
            handler.setFormatter(formatter)
            handler.set_name("experiment_file")
            
            logger.addHandler(handler)
    
    return logger


def log_experiment_event(
    event_type: str,
    experiment_id: str = None,
    model_name: str = None,
    demographic: str = None,
    decision: str = None,
    **kwargs,
) -> None:
    """
    Log an experiment event with structured data.
    
    Args:
        event_type: Type of event (start, response, error, complete)
        experiment_id: Unique experiment identifier
        model_name: Name of the model being tested
        demographic: Demographic group being tested
        decision: Decision made by the model
        **kwargs: Additional structured data
    """
    logger = get_experiment_logger()
    
    event_data = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
    }
    
    if experiment_id:
        event_data["experiment_id"] = experiment_id
    if model_name:
        event_data["model_name"] = model_name
    if demographic:
        event_data["demographic"] = demographic
    if decision:
        event_data["decision"] = decision
    
    # Add any additional data
    event_data.update(kwargs)
    
    # Log as JSON-formatted message
    logger.info(f"EXPERIMENT_EVENT: {event_data}")


def configure_external_loggers() -> None:
    """Configure logging levels for external libraries."""
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("flask").setLevel(logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.WARNING)


# Auto-setup when module is imported in development
if __name__ != "__main__":
    # Check if we're in a test environment
    if "pytest" not in sys.modules and "unittest" not in sys.modules:
        setup_logging()
        configure_external_loggers()