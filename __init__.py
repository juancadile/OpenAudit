"""
OpenAudit - Modular LLM Bias Auditing Platform

The leading open-source framework for detecting and analyzing bias in Large Language Models.
"""

__version__ = "1.0.0"
__author__ = "OpenAudit Team"
__email__ = "team@openaudit.org"
__license__ = "MIT"

# Import main components for easy access
from .core.bias_testing_framework import BiasAnalyzer, HiringBiasTest
from .core.model_manager import ModelManager
from .core.modular_bias_analyzer import ModularBiasAnalyzer
from .core.module_registry import get_global_registry
from .core.analysis_profiles import get_global_profile_manager

# Make key classes available at package level
__all__ = [
    "BiasAnalyzer",
    "HiringBiasTest",
    "ModelManager", 
    "ModularBiasAnalyzer",
    "get_global_registry",
    "get_global_profile_manager"
] 