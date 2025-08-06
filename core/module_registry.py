"""
Analysis Module Registry for OpenAudit

Central registry system for managing analysis modules in a plug-and-play architecture.
Handles module discovery, registration, compatibility checking, and pipeline creation.
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base_analyzer import BaseAnalysisModule, ModuleCategory, ModuleInfo

logger = logging.getLogger(__name__)


class ModuleCompatibilityError(Exception):
    """Raised when modules are incompatible"""


class AnalysisModuleRegistry:
    """
    Central registry for all analysis modules.

    Provides module discovery, registration, compatibility checking,
    and pipeline creation capabilities.
    """

    def __init__(self, auto_discover: bool = True):
        self.modules: Dict[str, Type[BaseAnalysisModule]] = {}
        self.module_instances: Dict[str, BaseAnalysisModule] = {}
        self._compatibility_cache: Dict[str, Dict[str, Any]] = {}

        if auto_discover:
            self._auto_discover_modules()

    def register_module(
        self, name: str, module_class: Type[BaseAnalysisModule], override: bool = False
    ) -> bool:
        """
        Register an analysis module

        Args:
            name: Unique module name
            module_class: Module class implementing BaseAnalysisModule
            override: Whether to override existing module with same name

        Returns:
            True if registration successful, False otherwise
        """
        if name in self.modules and not override:
            logger.warning(
                f"Module '{name}' already registered. Use override=True to replace."
            )
            return False

        # Validate module class
        if not issubclass(module_class, BaseAnalysisModule):
            logger.error(
                f"Module class {module_class} does not inherit from BaseAnalysisModule"
            )
            return False

        try:
            # Create instance to validate
            instance = module_class()
            if not instance.is_valid:
                logger.warning(
                    f"Module '{name}' has dependency issues but will be registered"
                )

            self.modules[name] = module_class
            self.module_instances[name] = instance
            self._compatibility_cache.pop(name, None)  # Clear cache

            logger.info(f"Registered module: {name} ({instance.module_info.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to register module '{name}': {e}")
            return False

    def unregister_module(self, name: str) -> bool:
        """Unregister a module"""
        if name not in self.modules:
            return False

        del self.modules[name]
        del self.module_instances[name]
        self._compatibility_cache.pop(name, None)

        logger.info(f"Unregistered module: {name}")
        return True

    def get_available_modules(self) -> List[str]:
        """Get list of all registered module names"""
        return list(self.modules.keys())

    def get_modules_by_category(self, category: ModuleCategory) -> List[str]:
        """Get modules filtered by category"""
        return [
            name
            for name, instance in self.module_instances.items()
            if instance.module_info.category == category
        ]

    def get_modules_by_tags(self, tags: List[str]) -> List[str]:
        """Get modules that have any of the specified tags"""
        matching_modules = []
        for name, instance in self.module_instances.items():
            if any(tag in instance.module_info.tags for tag in tags):
                matching_modules.append(name)
        return matching_modules

    def get_module(self, name: str) -> Optional[BaseAnalysisModule]:
        """Get module instance by name"""
        return self.module_instances.get(name)

    def has_module(self, name: str) -> bool:
        """Check if module is registered"""
        return name in self.modules

    def create_module_instance(self, name: str) -> Optional[BaseAnalysisModule]:
        """Create fresh instance of a module"""
        if name not in self.modules:
            return None

        try:
            return self.modules[name]()
        except Exception as e:
            logger.error(f"Failed to create instance of module '{name}': {e}")
            return None

    def validate_module_compatibility(self, modules: List[str]) -> Dict[str, Any]:
        """
        Check if selected modules can work together

        Returns:
            Dict with compatibility analysis:
            {
                "compatible": bool,
                "issues": List[str],
                "warnings": List[str],
                "data_type_conflicts": List[str],
                "dependency_conflicts": List[str]
            }
        """
        result = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "data_type_conflicts": [],
            "dependency_conflicts": [],
        }

        if not modules:
            result["issues"].append("No modules specified")
            result["compatible"] = False
            return result

        # Check if all modules exist
        missing_modules = [m for m in modules if m not in self.modules]
        if missing_modules:
            result["issues"].extend([f"Module not found: {m}" for m in missing_modules])
            result["compatible"] = False
            return result

        # Get module instances
        instances = [self.module_instances[name] for name in modules]

        # Check validity of each module
        invalid_modules = []
        for name, instance in zip(modules, instances):
            if not instance.is_valid:
                invalid_modules.append(name)

        if invalid_modules:
            result["issues"].extend(
                [f"Module has dependency issues: {m}" for m in invalid_modules]
            )
            result["warnings"].append("Some modules have dependency issues")

        # Check data type compatibility
        data_types_by_module = {}
        for name, instance in zip(modules, instances):
            data_types_by_module[name] = set(instance.get_supported_data_types())

        # Find common data types
        if data_types_by_module:
            common_types = set.intersection(*data_types_by_module.values())
            if not common_types:
                result["data_type_conflicts"] = [
                    f"No common data types between modules: {list(data_types_by_module.keys())}"
                ]
                result["warnings"].append("Modules may not work with same data")

        # Check for conflicting dependencies (basic check)
        all_dependencies = []
        for instance in instances:
            all_dependencies.extend(instance.module_info.requirements.dependencies)

        # This is a basic check - could be enhanced with version checking
        dependency_counts = {}
        for dep in all_dependencies:
            dependency_counts[dep] = dependency_counts.get(dep, 0) + 1

        return result

    def create_analysis_pipeline(self, selected_modules: List[str]) -> Dict[str, Any]:
        """
        Create analysis pipeline from selected modules

        Returns:
            Dict with pipeline information and module instances
        """
        compatibility = self.validate_module_compatibility(selected_modules)

        if not compatibility["compatible"]:
            raise ModuleCompatibilityError(
                f"Modules incompatible: {compatibility['issues']}"
            )

        pipeline = {
            "modules": {},
            "execution_order": selected_modules,
            "compatibility_check": compatibility,
            "estimated_requirements": self._calculate_pipeline_requirements(
                selected_modules
            ),
        }

        # Create fresh instances for the pipeline
        for name in selected_modules:
            instance = self.create_module_instance(name)
            if instance:
                pipeline["modules"][name] = instance
            else:
                raise ModuleCompatibilityError(
                    f"Failed to create instance of module: {name}"
                )

        return pipeline

    def get_module_info(self, name: str) -> Optional[ModuleInfo]:
        """Get detailed information about a module"""
        instance = self.get_module(name)
        return instance.module_info if instance else None

    def get_all_modules_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered modules"""
        return {
            name: instance.module_info.to_dict()
            for name, instance in self.module_instances.items()
        }

    def search_modules(self, query: str) -> List[str]:
        """Search modules by name, description, or tags"""
        query_lower = query.lower()
        matching_modules = []

        for name, instance in self.module_instances.items():
            info = instance.module_info

            # Check name
            if query_lower in name.lower():
                matching_modules.append(name)
                continue

            # Check description
            if query_lower in info.description.lower():
                matching_modules.append(name)
                continue

            # Check tags
            if any(query_lower in tag.lower() for tag in info.tags):
                matching_modules.append(name)
                continue

        return matching_modules

    def _auto_discover_modules(self):
        """Automatically discover and register modules in the core directory"""
        try:
            core_dir = Path(__file__).parent
            logger.info(f"Auto-discovering modules in: {core_dir}")

            # Import known modules and register them
            self._register_builtin_modules()

        except Exception as e:
            logger.error(f"Auto-discovery failed: {e}")

    def _register_builtin_modules(self):
        """Register built-in analysis modules"""
        builtin_modules = [
            ("enhanced_statistics", "EnhancedStatisticalAnalyzer"),
            ("cultural_context", "CulturalBiasDetector"),
            ("multi_level_classifier", "MultiLevelBiasClassifier"),
        ]

        for module_name, class_name in builtin_modules:
            try:
                module = importlib.import_module(f"core.{module_name}")
                getattr(module, class_name)

                # We'll need to wrap these in BaseAnalysisModule interface
                # For now, register them as placeholder
                logger.info(f"Found builtin module: {module_name}.{class_name}")

            except Exception as e:
                logger.warning(f"Could not import builtin module {module_name}: {e}")

    def _calculate_pipeline_requirements(self, modules: List[str]) -> Dict[str, Any]:
        """Calculate combined requirements for a pipeline"""
        if not modules:
            return {}

        instances = [
            self.module_instances[name]
            for name in modules
            if name in self.module_instances
        ]

        if not instances:
            return {}

        # Calculate maximum requirements
        max_samples = max(
            inst.module_info.requirements.min_samples for inst in instances
        )
        max_groups = max(inst.module_info.requirements.min_groups for inst in instances)

        # Combine data types
        all_data_types = set()
        for inst in instances:
            all_data_types.update(inst.module_info.requirements.data_types)

        # Combine dependencies
        all_dependencies = set()
        for inst in instances:
            all_dependencies.update(inst.module_info.requirements.dependencies)

        return {
            "min_samples": max_samples,
            "min_groups": max_groups,
            "data_types": list(all_data_types),
            "dependencies": list(all_dependencies),
        }


# Global registry instance
_global_registry = None


def get_global_registry() -> AnalysisModuleRegistry:
    """Get the global module registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = AnalysisModuleRegistry()
    return _global_registry


def register_module(
    name: str, module_class: Type[BaseAnalysisModule], override: bool = False
) -> bool:
    """Convenience function to register module in global registry"""
    return get_global_registry().register_module(name, module_class, override)


def clear_global_registry():
    """Clear the global registry (useful for testing)"""
    global _global_registry
    _global_registry = None


def unregister_module(name: str) -> bool:
    """Convenience function to unregister module from global registry"""
    return get_global_registry().unregister_module(name)
