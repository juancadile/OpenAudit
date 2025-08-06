"""
OpenAudit Core Module

Central module for the OpenAudit bias testing framework.
Contains the main analysis classes and utilities.
"""

__version__ = "1.0.0"
__author__ = "OpenAudit Team"
__email__ = "team@openaudit.org"
__license__ = "MIT"

from .analysis_profiles import get_global_profile_manager
from .bias_testing_framework import BiasAnalyzer, HiringBiasTest
from .model_manager import ModelManager
from .modular_bias_analyzer import ModularBiasAnalyzer
from .module_registry import get_global_registry

__all__ = [
    "BiasAnalyzer",
    "HiringBiasTest",
    "ModelManager",
    "ModularBiasAnalyzer",
    "get_global_registry",
    "get_global_profile_manager",
]
