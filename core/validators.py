"""
OpenAudit Validation Utilities
Comprehensive validation functions for input data and configurations
"""

import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .exceptions import ValidationError, InvalidFormatError, CeterisParibusViolationError
from .logging_config import get_logger

logger = get_logger(__name__)


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_demographic_name(name: str) -> bool:
    """
    Validate demographic name format.
    Names should contain only letters, spaces, hyphens, and apostrophes.
    
    Args:
        name: Name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not name or not name.strip():
        return False
    
    # Allow letters, spaces, hyphens, apostrophes, and periods
    pattern = r"^[a-zA-Z\s\-'.]+$"
    return bool(re.match(pattern, name.strip()))


def validate_university_name(university: str) -> bool:
    """
    Validate university name format.
    
    Args:
        university: University name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not university or not university.strip():
        return False
    
    # Allow letters, spaces, numbers, hyphens, apostrophes, and common punctuation
    pattern = r"^[a-zA-Z0-9\s\-'.,&()]+$"
    return bool(re.match(pattern, university.strip()))


def validate_experience_years(experience: Union[str, int]) -> bool:
    """
    Validate years of experience.
    
    Args:
        experience: Years of experience (string or int)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        years = int(experience)
        return 0 <= years <= 50  # Reasonable range
    except (ValueError, TypeError):
        return False


def validate_address(address: str) -> bool:
    """
    Validate address format.
    
    Args:
        address: Address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not address or not address.strip():
        return False
    
    # Basic validation - contains letters, numbers, and address punctuation
    pattern = r"^[a-zA-Z0-9\s\-'.,#]+$"
    return bool(re.match(pattern, address.strip()))


def validate_cv_variables(variables: Dict[str, Any], strict: bool = True) -> None:
    """
    Validate CV generation variables with comprehensive checks.
    
    Args:
        variables: Dictionary of CV variables
        strict: If True, performs strict validation
        
    Raises:
        ValidationError: If validation fails
    """
    logger.debug(f"Validating CV variables: {list(variables.keys())}")
    
    required_fields = ["name", "university", "experience", "address"]
    
    # Check required fields
    for field in required_fields:
        if field not in variables:
            raise ValidationError(
                f"Missing required CV variable: {field}",
                details={"missing_field": field, "required_fields": required_fields}
            )
    
    # Validate each field
    name = variables["name"]
    if not isinstance(name, str) or not validate_demographic_name(name):
        raise ValidationError(
            f"Invalid name format: '{name}'. Names should contain only letters, spaces, hyphens, and apostrophes.",
            details={"field": "name", "value": name}
        )
    
    university = variables["university"]
    if not isinstance(university, str) or not validate_university_name(university):
        raise ValidationError(
            f"Invalid university name: '{university}'",
            details={"field": "university", "value": university}
        )
    
    experience = variables["experience"]
    if not validate_experience_years(experience):
        raise ValidationError(
            f"Invalid experience years: '{experience}'. Must be 0-50.",
            details={"field": "experience", "value": experience}
        )
    
    address = variables["address"]
    if not isinstance(address, str) or not validate_address(address):
        raise ValidationError(
            f"Invalid address format: '{address}'",
            details={"field": "address", "value": address}
        )
    
    if strict:
        # Additional strict validations
        if len(name.split()) < 2:
            raise ValidationError(
                "Name should contain at least first and last name",
                details={"field": "name", "value": name}
            )
        
        if len(university.strip()) < 3:
            raise ValidationError(
                "University name too short",
                details={"field": "university", "value": university}
            )
    
    logger.debug("CV variables validation passed")


def validate_ceteris_paribus(
    variables_list: List[Dict[str, Any]], 
    allowed_varying_fields: Optional[List[str]] = None
) -> None:
    """
    Validate that only allowed fields vary between variable sets (ceteris paribus).
    
    Args:
        variables_list: List of variable dictionaries to compare
        allowed_varying_fields: Fields that are allowed to vary (default: ["name"])
        
    Raises:
        CeterisParibusViolationError: If ceteris paribus is violated
    """
    if len(variables_list) < 2:
        return  # Nothing to compare
    
    if allowed_varying_fields is None:
        allowed_varying_fields = ["name"]
    
    logger.debug(f"Validating ceteris paribus for {len(variables_list)} variable sets")
    
    base_variables = variables_list[0]
    
    for i, variables in enumerate(variables_list[1:], 1):
        # Check if all fields except allowed ones are identical
        for field, base_value in base_variables.items():
            if field in allowed_varying_fields:
                continue
            
            current_value = variables.get(field)
            if current_value != base_value:
                raise CeterisParibusViolationError(
                    f"Ceteris paribus violation: field '{field}' varies between variable sets",
                    expected_variables=base_variables,
                    actual_variables=variables
                )
    
    logger.debug("Ceteris paribus validation passed")


def validate_experiment_config(config: Dict[str, Any]) -> None:
    """
    Validate experiment configuration with comprehensive checks.
    
    Args:
        config: Experiment configuration dictionary
        
    Raises:
        ValidationError: If configuration is invalid
    """
    logger.debug("Validating experiment configuration")
    
    required_fields = ["models", "demographics", "cv_template", "prompt_template"]
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValidationError(
                f"Missing required field: {field}",
                details={"missing_field": field, "required_fields": required_fields}
            )
    
    # Validate models
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValidationError(
            "Models must be a non-empty list",
            details={"models": models, "type": type(models).__name__}
        )
    
    # Validate demographics
    demographics = config["demographics"]
    if not isinstance(demographics, list) or not demographics:
        raise ValidationError(
            "Demographics must be a non-empty list",
            details={"demographics": demographics, "type": type(demographics).__name__}
        )
    
    # Validate CV template
    cv_template = config["cv_template"]
    if not isinstance(cv_template, str) or not cv_template.strip():
        raise ValidationError(
            "CV template must be a non-empty string",
            details={"cv_template": cv_template, "type": type(cv_template).__name__}
        )
    
    # Validate prompt template
    prompt_template = config["prompt_template"]
    if not isinstance(prompt_template, str) or not prompt_template.strip():
        raise ValidationError(
            "Prompt template must be a non-empty string",
            details={"prompt_template": prompt_template, "type": type(prompt_template).__name__}
        )
    
    # Optional fields validation
    if "iterations" in config:
        iterations = config["iterations"]
        if not isinstance(iterations, int) or iterations < 1:
            raise ValidationError(
                "Iterations must be a positive integer",
                details={"iterations": iterations, "type": type(iterations).__name__}
            )
    
    if "cv_level" in config:
        cv_level = config["cv_level"]
        valid_levels = ["weak", "borderline", "strong"]
        if cv_level not in valid_levels:
            raise ValidationError(
                f"CV level must be one of: {valid_levels}",
                details={"cv_level": cv_level, "valid_levels": valid_levels}
            )
    
    logger.debug("Experiment configuration validation passed")


def validate_model_response(response: Dict[str, Any]) -> None:
    """
    Validate model response format.
    
    Args:
        response: Model response dictionary
        
    Raises:
        ValidationError: If response format is invalid
    """
    required_fields = ["model_name", "provider", "response", "timestamp"]
    
    for field in required_fields:
        if field not in response:
            raise ValidationError(
                f"Missing required response field: {field}",
                details={"missing_field": field, "required_fields": required_fields}
            )
    
    # Validate response content
    if not isinstance(response["response"], str) or not response["response"].strip():
        raise ValidationError(
            "Response content must be a non-empty string",
            details={"response": response.get("response")}
        )
    
    # Validate model name
    if not isinstance(response["model_name"], str) or not response["model_name"].strip():
        raise ValidationError(
            "Model name must be a non-empty string",
            details={"model_name": response.get("model_name")}
        )


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: If True, file must exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    if isinstance(file_path, str):
        path = Path(file_path)
    else:
        path = file_path
    
    if must_exist and not path.exists():
        raise ValidationError(
            f"File does not exist: {path}",
            details={"file_path": str(path)}
        )
    
    if must_exist and not path.is_file():
        raise ValidationError(
            f"Path is not a file: {path}",
            details={"file_path": str(path)}
        )
    
    return path


def validate_json_structure(data: Any, expected_structure: Dict[str, type]) -> None:
    """
    Validate JSON data structure.
    
    Args:
        data: Data to validate
        expected_structure: Expected structure with field names and types
        
    Raises:
        ValidationError: If structure is invalid
    """
    if not isinstance(data, dict):
        raise ValidationError(
            "Data must be a dictionary",
            details={"data_type": type(data).__name__}
        )
    
    for field, expected_type in expected_structure.items():
        if field not in data:
            raise ValidationError(
                f"Missing required field: {field}",
                details={"missing_field": field, "expected_fields": list(expected_structure.keys())}
            )
        
        if not isinstance(data[field], expected_type):
            raise ValidationError(
                f"Field '{field}' must be of type {expected_type.__name__}",
                details={
                    "field": field,
                    "expected_type": expected_type.__name__,
                    "actual_type": type(data[field]).__name__
                }
            )


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > 200:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:200-len(ext)-1] + ('.' + ext if ext else '')
    
    return sanitized or "unnamed_file"