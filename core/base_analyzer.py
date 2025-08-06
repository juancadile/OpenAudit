"""
Base Analysis Module Interface for OpenAudit

Provides standardized interface for all analysis modules to enable
plug-and-play modular architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ModuleCategory(Enum):
    """Categories of analysis modules"""

    STATISTICAL = "statistical"
    CULTURAL = "cultural"
    CLASSIFICATION = "classification"
    ALIGNMENT = "alignment"
    CONFLICT = "conflict"
    FAIRNESS = "fairness"
    CUSTOM = "custom"


@dataclass
class ModuleRequirements:
    """Requirements for running an analysis module"""

    min_samples: int = 1
    min_groups: int = 1
    data_types: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    optional_dependencies: Optional[List[str]] = None

    def __post_init__(self):
        if self.data_types is None:
            self.data_types = ["responses"]
        if self.dependencies is None:
            self.dependencies = []
        if self.optional_dependencies is None:
            self.optional_dependencies = []


@dataclass
class ModuleInfo:
    """Module metadata and information"""

    name: str
    version: str
    description: str
    author: str
    category: ModuleCategory
    tags: List[str]
    requirements: ModuleRequirements

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "category": self.category.value,
            "tags": self.tags,
            "requirements": {
                "min_samples": self.requirements.min_samples,
                "min_groups": self.requirements.min_groups,
                "data_types": self.requirements.data_types,
                "dependencies": self.requirements.dependencies,
                "optional_dependencies": self.requirements.optional_dependencies,
            },
        }


class BaseAnalysisModule(ABC):
    """
    Abstract base class for all analysis modules in OpenAudit.

    This standardized interface enables plug-and-play modular architecture
    where modules can be mixed and matched for different analysis needs.
    """

    def __init__(self):
        self._module_info = self._create_module_info()
        self._validated = False

    @abstractmethod
    def _create_module_info(self) -> ModuleInfo:
        """Create and return module information"""

    @abstractmethod
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Main analysis method - standardized input/output interface

        Args:
            data: Input data (format depends on module requirements)
            **kwargs: Additional module-specific parameters

        Returns:
            Dict with standardized result structure:
            {
                "summary": Dict[str, Any],        # High-level findings
                "detailed_results": Dict[str, Any], # Detailed analysis
                "key_findings": List[str],         # Important discoveries
                "confidence_score": float,         # Overall confidence (0-1)
                "recommendations": List[str],      # Actionable recommendations
                "metadata": Dict[str, Any]         # Analysis metadata
            }
        """

    def validate_input(self, data: Any) -> Dict[str, Any]:
        """
        Validate input data meets module requirements

        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        validation_result = {"valid": True, "errors": [], "warnings": []}

        try:
            # Basic validation - can be overridden by subclasses
            if hasattr(data, "__len__"):
                if len(data) < self._module_info.requirements.min_samples:
                    validation_result["errors"].append(
                        f"Insufficient samples: {len(data)} < {self._module_info.requirements.min_samples}"
                    )
                    validation_result["valid"] = False
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False

        return validation_result

    def get_supported_data_types(self) -> List[str]:
        """Return supported data input types"""
        return self._module_info.requirements.data_types

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available"""
        dependency_status = {}

        for dep in self._module_info.requirements.dependencies:
            try:
                __import__(dep)
                dependency_status[dep] = True
            except ImportError:
                dependency_status[dep] = False

        for dep in self._module_info.requirements.optional_dependencies:
            try:
                __import__(dep)
                dependency_status[f"{dep} (optional)"] = True
            except ImportError:
                dependency_status[f"{dep} (optional)"] = False

        return dependency_status

    @property
    def module_info(self) -> ModuleInfo:
        """Get module information"""
        return self._module_info

    @property
    def is_valid(self) -> bool:
        """Check if module is properly configured"""
        if not self._validated:
            deps = self.check_dependencies()
            required_deps_ok = all(
                deps.get(dep, False)
                for dep in self._module_info.requirements.dependencies
            )
            self._validated = required_deps_ok

        return self._validated

    def get_compatibility_info(self) -> Dict[str, Any]:
        """Get information about module compatibility"""
        return {
            "supported_data_types": self.get_supported_data_types(),
            "dependency_status": self.check_dependencies(),
            "is_valid": self.is_valid,
            "category": self._module_info.category.value,
            "requirements": self._module_info.requirements.__dict__,
        }

    def __str__(self) -> str:
        return f"{self._module_info.name} v{self._module_info.version}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._module_info.name}>"
