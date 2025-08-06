"""
OpenAudit Decorators
Utility decorators for error handling, validation, and common functionality
"""

import functools
import time
from typing import Callable, Optional

from flask import jsonify, request

from .exceptions import OpenAuditError, ValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


def api_error_handler(func: Callable) -> Callable:
    """
    Decorator to handle API errors consistently.

    Converts OpenAudit exceptions to proper JSON responses and logs errors.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.debug(f"API call: {func.__name__} with args={args}, kwargs={kwargs}")
            start_time = time.time()

            result = func(*args, **kwargs)

            duration = time.time() - start_time
            logger.debug(f"API call completed: {func.__name__} in {duration:.3f}s")

            return result

        except OpenAuditError as e:
            logger.warning(
                f"OpenAudit error in {func.__name__}: {e.message}", extra=e.details
            )
            return jsonify(e.to_dict()), 400

        except ValidationError as e:
            logger.warning(
                f"Validation error in {func.__name__}: {e.message}", extra=e.details
            )
            return jsonify(e.to_dict()), 422

        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return (
                jsonify(
                    {
                        "error": "InternalServerError",
                        "message": "An unexpected error occurred",
                        "details": {"function": func.__name__},
                    }
                ),
                500,
            )

    return wrapper


def validate_json_request(required_fields: Optional[list] = None) -> Callable:
    """
    Decorator to validate JSON request data.

    Args:
        required_fields: List of required field names
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                logger.warning(f"Non-JSON request to {func.__name__}")
                return (
                    jsonify(
                        {"error": "ValidationError", "message": "Request must be JSON"}
                    ),
                    400,
                )

            data = request.get_json()
            if data is None:
                logger.warning(f"Empty JSON request to {func.__name__}")
                return (
                    jsonify(
                        {"error": "ValidationError", "message": "Invalid JSON data"}
                    ),
                    400,
                )

            # Check required fields
            if required_fields:
                missing_fields = [
                    field for field in required_fields if field not in data
                ]
                if missing_fields:
                    logger.warning(
                        f"Missing required fields in {func.__name__}: {missing_fields}"
                    )
                    return (
                        jsonify(
                            {
                                "error": "ValidationError",
                                "message": f"Missing required fields: {', '.join(missing_fields)}",
                                "details": {"missing_fields": missing_fields},
                            }
                        ),
                        422,
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_performance(threshold_seconds: float = 1.0) -> Callable:
    """
    Decorator to log performance of slow functions.

    Args:
        threshold_seconds: Log warning if function takes longer than this
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            if duration > threshold_seconds:
                logger.warning(f"Slow function {func.__name__}: {duration:.3f}s")
            else:
                logger.debug(f"Function {func.__name__}: {duration:.3f}s")

            return result

        return wrapper

    return decorator


def retry_on_failure(max_attempts: int = 3, delay_seconds: float = 1.0) -> Callable:
    """
    Decorator to retry function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay_seconds: Delay between attempts
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}"
                        )
                        time.sleep(delay_seconds)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper

    return decorator


def cache_result(ttl_seconds: int = 300) -> Callable:
    """
    Simple decorator to cache function results.

    Args:
        ttl_seconds: Time to live for cached results
    """

    def decorator(func: Callable) -> Callable:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            current_time = time.time()

            # Check if we have a valid cached result
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp < ttl_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    # Remove expired cache entry
                    del cache[cache_key]

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            logger.debug(f"Cached result for {func.__name__}")

            return result

        return wrapper

    return decorator


def validate_experiment_id(func: Callable) -> Callable:
    """Decorator to validate experiment ID parameter."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Look for experiment_id in kwargs or URL parameters
        experiment_id = kwargs.get("experiment_id") or request.view_args.get(
            "experiment_id"
        )

        if not experiment_id:
            logger.warning(f"Missing experiment_id in {func.__name__}")
            return (
                jsonify(
                    {"error": "ValidationError", "message": "Experiment ID is required"}
                ),
                400,
            )

        if not isinstance(experiment_id, str) or not experiment_id.strip():
            logger.warning(f"Invalid experiment_id in {func.__name__}: {experiment_id}")
            return (
                jsonify(
                    {
                        "error": "ValidationError",
                        "message": "Experiment ID must be a non-empty string",
                    }
                ),
                400,
            )

        return func(*args, **kwargs)

    return wrapper


def require_api_key(provider: Optional[str] = None) -> Callable:
    """
    Decorator to require API key for certain operations.

    Args:
        provider: Specific provider to check (if None, checks for any API key)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This is a placeholder - in a real implementation, you'd check
            # for API keys in headers or configuration
            # For now, we'll just log the requirement

            if provider:
                logger.debug(f"API key required for {provider} in {func.__name__}")
            else:
                logger.debug(f"API key required in {func.__name__}")

            return func(*args, **kwargs)

        return wrapper

    return decorator
