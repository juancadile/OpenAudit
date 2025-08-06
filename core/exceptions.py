"""
OpenAudit Custom Exceptions
Comprehensive error handling for bias testing platform
"""

from typing import Any, Dict, List


class OpenAuditError(Exception):
    """Base exception for all OpenAudit errors."""

    def __init__(
        self, message: str, error_code: str = None, details: Dict[str, Any] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(OpenAuditError):
    """Raised when there's a configuration problem."""


class ValidationError(OpenAuditError):
    """Raised when input validation fails."""


class ModelError(OpenAuditError):
    """Raised when there's an issue with AI model operations."""


class ProviderError(ModelError):
    """Raised when there's an issue with a specific AI provider."""

    def __init__(
        self,
        message: str,
        provider: str,
        error_code: str = None,
        details: Dict[str, Any] = None,
    ):
        self.provider = provider
        details = details or {}
        details["provider"] = provider
        super().__init__(message, error_code, details)


class APIKeyError(ProviderError):
    """Raised when API key is missing or invalid."""


class RateLimitError(ProviderError):
    """Raised when API rate limit is exceeded."""


class ExperimentError(OpenAuditError):
    """Raised when there's an issue with experiment execution."""


class TemplateError(OpenAuditError):
    """Raised when there's an issue with template processing."""


class DataError(OpenAuditError):
    """Raised when there's an issue with data processing or storage."""


class BiasTestError(OpenAuditError):
    """Raised when there's an issue with bias testing framework."""


class AnalysisError(OpenAuditError):
    """Raised when there's an issue with analysis execution."""


class CeterisParibusViolationError(BiasTestError):
    """Raised when ceteris paribus (all else equal) condition is violated."""

    def __init__(
        self,
        message: str,
        expected_variables: Dict[str, Any],
        actual_variables: Dict[str, Any],
        error_code: str = None,
    ):
        self.expected_variables = expected_variables
        self.actual_variables = actual_variables

        details = {
            "expected_variables": expected_variables,
            "actual_variables": actual_variables,
            "violations": self._find_violations(),
        }

        super().__init__(message, error_code, details)

    def _find_violations(self) -> List[str]:
        """Find specific variables that violate ceteris paribus."""
        violations = []
        for key, expected_value in self.expected_variables.items():
            if key == "name":  # Name is allowed to vary
                continue
            actual_value = self.actual_variables.get(key)
            if actual_value != expected_value:
                violations.append(
                    f"{key}: expected '{expected_value}', got '{actual_value}'"
                )
        return violations


class FileProcessingError(DataError):
    """Raised when there's an issue processing files."""

    def __init__(
        self,
        message: str,
        file_path: str,
        error_code: str = None,
        details: Dict[str, Any] = None,
    ):
        self.file_path = file_path
        details = details or {}
        details["file_path"] = file_path
        super().__init__(message, error_code, details)


class ExperimentNotFoundError(ExperimentError):
    """Raised when a requested experiment is not found."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        message = f"Experiment not found: {experiment_id}"
        details = {"experiment_id": experiment_id}
        super().__init__(message, details=details)


class TemplateNotFoundError(TemplateError):
    """Raised when a requested template is not found."""

    def __init__(self, template_name: str, template_type: str = "unknown"):
        self.template_name = template_name
        self.template_type = template_type
        message = f"{template_type.upper()} template not found: {template_name}"
        details = {"template_name": template_name, "template_type": template_type}
        super().__init__(message, details=details)


class ModelNotAvailableError(ModelError):
    """Raised when a requested model is not available."""

    def __init__(self, model_name: str, provider: str = None):
        self.model_name = model_name
        self.provider = provider

        if provider:
            message = f"Model '{model_name}' not available from provider '{provider}'"
        else:
            message = f"Model '{model_name}' not available"

        details = {"model_name": model_name}
        if provider:
            details["provider"] = provider

        super().__init__(message, details=details)


class InsufficientDataError(DataError):
    """Raised when there's insufficient data for analysis."""

    def __init__(
        self, required_count: int, actual_count: int, data_type: str = "data points"
    ):
        self.required_count = required_count
        self.actual_count = actual_count
        self.data_type = data_type

        message = f"Insufficient {data_type}: need {required_count}, got {actual_count}"
        details = {
            "required_count": required_count,
            "actual_count": actual_count,
            "data_type": data_type,
        }
        super().__init__(message, details=details)


class InvalidFormatError(ValidationError):
    """Raised when data format is invalid."""

    def __init__(
        self, expected_format: str, actual_format: str = None, field_name: str = None
    ):
        self.expected_format = expected_format
        self.actual_format = actual_format
        self.field_name = field_name

        if field_name:
            message = (
                f"Invalid format for field '{field_name}': expected {expected_format}"
            )
        else:
            message = f"Invalid format: expected {expected_format}"

        if actual_format:
            message += f", got {actual_format}"

        details = {"expected_format": expected_format}
        if actual_format:
            details["actual_format"] = actual_format
        if field_name:
            details["field_name"] = field_name

        super().__init__(message, details=details)


def handle_provider_error(error: Exception, provider: str) -> ProviderError:
    """
    Convert provider-specific errors to OpenAudit provider errors.

    Args:
        error: The original error from the provider
        provider: Name of the provider

    Returns:
        Appropriate ProviderError subclass
    """
    error_message = str(error)
    error_type = type(error).__name__

    # API key related errors
    if any(
        keyword in error_message.lower()
        for keyword in ["api key", "authentication", "unauthorized", "invalid key"]
    ):
        return APIKeyError(
            f"API key error for {provider}: {error_message}",
            provider=provider,
            details={"original_error": error_type},
        )

    # Rate limit errors
    if any(
        keyword in error_message.lower()
        for keyword in ["rate limit", "quota", "too many requests"]
    ):
        return RateLimitError(
            f"Rate limit exceeded for {provider}: {error_message}",
            provider=provider,
            details={"original_error": error_type},
        )

    # Generic provider error
    return ProviderError(
        f"Provider error for {provider}: {error_message}",
        provider=provider,
        details={"original_error": error_type},
    )


def validate_experiment_config(config: Dict[str, Any]) -> None:
    """
    Validate experiment configuration.

    Args:
        config: Experiment configuration dictionary

    Raises:
        ValidationError: If configuration is invalid
    """
    required_fields = ["models", "demographics", "prompt_template"]

    for field in required_fields:
        if field not in config:
            raise ValidationError(
                f"Missing required field: {field}",
                details={"missing_field": field, "required_fields": required_fields},
            )

    # Validate models list
    if not isinstance(config["models"], list) or not config["models"]:
        raise ValidationError(
            "Models must be a non-empty list", details={"models": config.get("models")}
        )

    # Validate demographics list
    if not isinstance(config["demographics"], list) or not config["demographics"]:
        raise ValidationError(
            "Demographics must be a non-empty list",
            details={"demographics": config.get("demographics")},
        )


def validate_cv_variables(variables: Dict[str, Any]) -> None:
    """
    Validate CV generation variables.

    Args:
        variables: Dictionary of CV variables

    Raises:
        ValidationError: If variables are invalid
    """
    required_fields = ["name", "university", "experience", "address"]

    for field in required_fields:
        if field not in variables:
            raise ValidationError(
                f"Missing required CV variable: {field}",
                details={"missing_field": field, "required_fields": required_fields},
            )

        if not isinstance(variables[field], str) or not variables[field].strip():
            raise ValidationError(
                f"CV variable '{field}' must be a non-empty string",
                details={"field": field, "value": variables[field]},
            )


def validate_api_key(api_key: str, provider: str) -> None:
    """
    Validate API key format.

    Args:
        api_key: API key to validate
        provider: Provider name

    Raises:
        APIKeyError: If API key is invalid
    """
    if not api_key or not api_key.strip():
        raise APIKeyError(f"Empty API key for {provider}", provider=provider)

    # Provider-specific validation
    if provider.lower() == "openai":
        if not api_key.startswith("sk-"):
            raise APIKeyError(
                f"Invalid OpenAI API key format (should start with 'sk-')",
                provider=provider,
                details={"key_prefix": api_key[:10] + "..."},
            )

    elif provider.lower() == "anthropic":
        if not api_key.startswith("sk-ant-"):
            raise APIKeyError(
                f"Invalid Anthropic API key format (should start with 'sk-ant-')",
                provider=provider,
                details={"key_prefix": api_key[:10] + "..."},
            )
