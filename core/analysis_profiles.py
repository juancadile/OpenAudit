"""
Analysis Profiles Configuration for OpenAudit

Predefined analysis profiles for different use cases, providing
easy selection of module combinations for specific scenarios.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ProfileCategory(Enum):
    """Categories of analysis profiles"""

    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    SPECIALIZED = "specialized"
    RESEARCH = "research"
    CUSTOM = "custom"


@dataclass
class AnalysisProfile:
    """Configuration for an analysis profile"""

    name: str
    description: str
    use_case: str
    category: ProfileCategory
    modules: List[str]
    default_parameters: Dict[str, Any]
    estimated_runtime: str
    min_samples_recommended: int
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "use_case": self.use_case,
            "category": self.category.value,
            "modules": self.modules,
            "default_parameters": self.default_parameters,
            "estimated_runtime": self.estimated_runtime,
            "min_samples_recommended": self.min_samples_recommended,
            "tags": self.tags,
        }


# Predefined analysis profiles
ANALYSIS_PROFILES = {
    "basic": AnalysisProfile(
        name="Basic Analysis",
        description="Simple statistical bias testing with essential metrics",
        use_case="Quick bias check for development and testing",
        category=ProfileCategory.BASIC,
        modules=["enhanced_statistics"],
        default_parameters={
            "correction_method": "fdr_bh",
            "alpha": 0.05,
            "effect_size_threshold": 0.3,
        },
        estimated_runtime="< 1 minute",
        min_samples_recommended=10,
        tags=["fast", "statistical", "basic"],
    ),
    "standard": AnalysisProfile(
        name="Standard Bias Audit",
        description="Comprehensive bias analysis with enhanced statistics and classification",
        use_case="Standard bias audit for production systems",
        category=ProfileCategory.STANDARD,
        modules=["enhanced_statistics", "multi_level_classifier"],
        default_parameters={
            "correction_method": "fdr_bh",
            "alpha": 0.05,
            "effect_size_threshold": 0.3,
            "confidence_threshold": 0.6,
            "soft_bias_threshold": 0.3,
        },
        estimated_runtime="2-5 minutes",
        min_samples_recommended=20,
        tags=["comprehensive", "statistical", "classification"],
    ),
    "cultural": AnalysisProfile(
        name="Cultural Context Analysis",
        description="Focus on cultural and linguistic bias detection",
        use_case="Cross-cultural AI deployment and international usage",
        category=ProfileCategory.SPECIALIZED,
        modules=["enhanced_statistics", "cultural_context", "multi_level_classifier"],
        default_parameters={
            "correction_method": "bonferroni",
            "alpha": 0.01,
            "num_samples": 5,
            "contexts_to_test": [
                "diverse_team",
                "global_perspective",
                "local_community",
            ],
        },
        estimated_runtime="5-10 minutes",
        min_samples_recommended=25,
        tags=["cultural", "linguistic", "context", "international"],
    ),
    "alignment_focused": AnalysisProfile(
        name="AI Alignment Analysis",
        description="Focus on goal conflicts and human-AI alignment",
        use_case="AI safety research and alignment verification",
        category=ProfileCategory.RESEARCH,
        modules=["goal_conflict", "human_ai_alignment", "multi_level_classifier"],
        default_parameters={
            "scenarios_to_test": ["diversity_vs_merit", "speed_vs_thoroughness"],
            "alignment_threshold": 0.7,
            "num_samples": 10,
        },
        estimated_runtime="10-15 minutes",
        min_samples_recommended=30,
        tags=["alignment", "safety", "conflict", "research"],
    ),
    "research_grade": AnalysisProfile(
        name="Research Grade Analysis",
        description="Comprehensive analysis with all available modules for academic research",
        use_case="Academic research, publication-quality analysis",
        category=ProfileCategory.RESEARCH,
        modules=[
            "enhanced_statistics",
            "cultural_context",
            "multi_level_classifier",
            "goal_conflict",
            "human_ai_alignment",
        ],
        default_parameters={
            "correction_method": "bonferroni",
            "alpha": 0.01,
            "effect_size_threshold": 0.2,
            "num_samples": 10,
            "confidence_threshold": 0.7,
            "comprehensive_reporting": True,
        },
        estimated_runtime="15-30 minutes",
        min_samples_recommended=50,
        tags=["research", "comprehensive", "academic", "publication"],
    ),
    "fairness_audit": AnalysisProfile(
        name="Fairness Audit",
        description="Specialized fairness testing with demographic analysis",
        use_case="Compliance auditing and fairness verification",
        category=ProfileCategory.SPECIALIZED,
        modules=["enhanced_statistics", "multi_level_classifier", "cultural_context"],
        default_parameters={
            "correction_method": "bonferroni",
            "alpha": 0.01,
            "demographic_focus": True,
            "intersectional_analysis": True,
            "compliance_reporting": True,
        },
        estimated_runtime="5-15 minutes",
        min_samples_recommended=40,
        tags=["fairness", "compliance", "demographic", "audit"],
    ),
    "hiring_specific": AnalysisProfile(
        name="Hiring Bias Analysis",
        description="Specialized analysis for hiring and recruitment scenarios",
        use_case="HR systems, recruitment AI, candidate evaluation",
        category=ProfileCategory.SPECIALIZED,
        modules=["enhanced_statistics", "multi_level_classifier", "cultural_context"],
        default_parameters={
            "contexts_to_test": [
                "diverse_team",
                "merit_only",
                "startup_culture",
                "traditional_corporate",
                "age_inclusive",
                "gender_neutral",
            ],
            "bias_types_focus": ["age", "gender", "race", "name", "education"],
            "hiring_specific_metrics": True,
        },
        estimated_runtime="10-20 minutes",
        min_samples_recommended=35,
        tags=["hiring", "recruitment", "hr", "employment"],
    ),
    "rapid_development": AnalysisProfile(
        name="Rapid Development Check",
        description="Fast analysis for development cycles",
        use_case="CI/CD integration, development iteration",
        category=ProfileCategory.BASIC,
        modules=["enhanced_statistics"],
        default_parameters={
            "correction_method": "none",
            "alpha": 0.05,
            "quick_mode": True,
            "essential_metrics_only": True,
        },
        estimated_runtime="< 30 seconds",
        min_samples_recommended=5,
        tags=["fast", "development", "ci-cd", "quick"],
    ),
    "complete": AnalysisProfile(
        name="Complete Analysis Suite",
        description="All available analysis modules with maximum depth",
        use_case="Maximum depth analysis, comprehensive evaluation",
        category=ProfileCategory.ADVANCED,
        modules=["all"],  # Special keyword for all available modules
        default_parameters={
            "correction_method": "bonferroni",
            "alpha": 0.001,
            "comprehensive_mode": True,
            "detailed_reporting": True,
            "export_raw_data": True,
        },
        estimated_runtime="30-60 minutes",
        min_samples_recommended=100,
        tags=["complete", "comprehensive", "maximum", "detailed"],
    ),
}


class ProfileManager:
    """Manager for analysis profiles"""

    def __init__(self):
        self.profiles = ANALYSIS_PROFILES.copy()
        self.custom_profiles: Dict[str, AnalysisProfile] = {}

    def get_profile(self, name: str) -> Optional[AnalysisProfile]:
        """Get profile by name"""
        if name in self.profiles:
            return self.profiles[name]
        return self.custom_profiles.get(name)

    def get_all_profiles(self) -> Dict[str, AnalysisProfile]:
        """Get all available profiles"""
        all_profiles = self.profiles.copy()
        all_profiles.update(self.custom_profiles)
        return all_profiles

    def get_profiles_by_category(self, category: ProfileCategory) -> List[str]:
        """Get profiles filtered by category"""
        matching_profiles = []
        all_profiles = self.get_all_profiles()

        for name, profile in all_profiles.items():
            if profile.category == category:
                matching_profiles.append(name)

        return matching_profiles

    def get_profiles_by_tags(self, tags: List[str]) -> List[str]:
        """Get profiles that have any of the specified tags"""
        matching_profiles = []
        all_profiles = self.get_all_profiles()

        for name, profile in all_profiles.items():
            if any(tag in profile.tags for tag in tags):
                matching_profiles.append(name)

        return matching_profiles

    def search_profiles(self, query: str) -> List[str]:
        """Search profiles by name, description, use case, or tags"""
        query_lower = query.lower()
        matching_profiles = []
        all_profiles = self.get_all_profiles()

        for name, profile in all_profiles.items():
            # Check name
            if query_lower in name.lower():
                matching_profiles.append(name)
                continue

            # Check description
            if query_lower in profile.description.lower():
                matching_profiles.append(name)
                continue

            # Check use case
            if query_lower in profile.use_case.lower():
                matching_profiles.append(name)
                continue

            # Check tags
            if any(query_lower in tag.lower() for tag in profile.tags):
                matching_profiles.append(name)
                continue

        return matching_profiles

    def add_custom_profile(
        self, name: str, profile: AnalysisProfile, override: bool = False
    ) -> bool:
        """Add a custom profile"""
        if name in self.profiles or (name in self.custom_profiles and not override):
            return False

        self.custom_profiles[name] = profile
        return True

    def remove_custom_profile(self, name: str) -> bool:
        """Remove a custom profile"""
        if name in self.custom_profiles:
            del self.custom_profiles[name]
            return True
        return False

    def get_profile_modules(self, profile_name: str, registry=None) -> List[str]:
        """
        Get modules for a given profile, resolving special keywords

        Args:
            profile_name: Name of the profile
            registry: Module registry instance (optional)

        Returns:
            List of module names
        """
        profile = self.get_profile(profile_name)
        if not profile:
            return []

        modules = profile.modules.copy()

        # Handle special keywords
        if "all" in modules:
            if registry:
                # Replace "all" with all available modules
                all_modules = registry.get_available_modules()
                modules.remove("all")
                modules.extend(all_modules)
            else:
                # Fallback to known modules
                modules.remove("all")
                modules.extend(
                    [
                        "enhanced_statistics",
                        "cultural_context",
                        "multi_level_classifier",
                        "goal_conflict",
                        "human_ai_alignment",
                    ]
                )

        return list(set(modules))  # Remove duplicates

    def validate_profile_compatibility(
        self, profile_name: str, registry=None
    ) -> Dict[str, Any]:
        """Validate that all modules in a profile are compatible"""
        modules = self.get_profile_modules(profile_name, registry)

        if not registry:
            return {
                "compatible": False,
                "error": "Registry required for compatibility checking",
            }

        return registry.validate_module_compatibility(modules)

    def get_profile_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all profiles"""
        summary = {}
        all_profiles = self.get_all_profiles()

        for name, profile in all_profiles.items():
            summary[name] = {
                "description": profile.description,
                "category": profile.category.value,
                "use_case": profile.use_case,
                "estimated_runtime": profile.estimated_runtime,
                "num_modules": len(profile.modules),
                "tags": profile.tags,
            }

        return summary


# Global profile manager instance
_global_profile_manager = None


def get_global_profile_manager() -> ProfileManager:
    """Get the global profile manager instance"""
    global _global_profile_manager
    if _global_profile_manager is None:
        _global_profile_manager = ProfileManager()
    return _global_profile_manager


def get_profile_modules(profile_name: str, registry=None) -> List[str]:
    """Convenience function to get modules for a profile"""
    return get_global_profile_manager().get_profile_modules(profile_name, registry)


def clear_global_profile_manager():
    """Clear the global profile manager (useful for testing)"""
    global _global_profile_manager
    _global_profile_manager = None
