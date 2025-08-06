"""
Enhanced Statistical Analysis Module - Modular Version

Refactored version of enhanced_statistics.py that implements the
BaseAnalysisModule interface for plug-and-play integration.
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base_analyzer import (
    BaseAnalysisModule,
    ModuleCategory,
    ModuleInfo,
    ModuleRequirements,
)
from .enhanced_statistics import (
    EffectSizeCalculator,
    EnhancedStatisticalAnalyzer,
    MultiGroupBiasAnalyzer,
    MultipleComparisonCorrector,
)

logger = logging.getLogger(__name__)


class EnhancedStatisticsModule(BaseAnalysisModule):
    """
    Enhanced Statistical Analysis Module for OpenAudit

    Provides advanced statistical methods including multi-group testing,
    effect size calculations, and multiple comparison corrections.
    """

    def _create_module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="Enhanced Statistics",
            version="1.0.0",
            description="Advanced statistical testing with effect sizes and multi-group analysis",
            author="OpenAudit Team",
            category=ModuleCategory.STATISTICAL,
            tags=[
                "statistics",
                "effect-size",
                "multi-group",
                "significance",
                "testing",
            ],
            requirements=ModuleRequirements(
                min_samples=10,
                min_groups=2,
                data_types=["responses", "grouped_responses", "demographic_data"],
                dependencies=["numpy", "pandas", "scipy", "statsmodels"],
                optional_dependencies=["matplotlib", "seaborn"],
            ),
        )

    def __init__(self):
        super().__init__()
        self.analyzer = EnhancedStatisticalAnalyzer()
        self.multi_group_analyzer = MultiGroupBiasAnalyzer()
        self.effect_calculator = EffectSizeCalculator()
        self.correction_handler = MultipleComparisonCorrector()

    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Main analysis method for enhanced statistical testing

        Args:
            data: Can be:
                - Dict[str, List[float]]: responses_by_group for multi-group analysis
                - List[LLMResponse]: response objects for comprehensive analysis
                - Dict with specific structure for other analyses
            **kwargs: Additional parameters:
                - control_group_name: str, name of control group
                - correction_method: str, multiple comparison method
                - alpha: float, significance level
                - analysis_type: str, specific analysis to run

        Returns:
            Standardized result structure with statistical analysis
        """
        try:
            analysis_type = kwargs.get("analysis_type", "comprehensive")

            if analysis_type == "multi_group":
                return self._multi_group_analysis(data, **kwargs)
            elif analysis_type == "comprehensive":
                return self._comprehensive_analysis(data, **kwargs)
            elif analysis_type == "effect_sizes":
                return self._effect_size_analysis(data, **kwargs)
            else:
                return self._comprehensive_analysis(data, **kwargs)

        except Exception as e:
            logger.error(f"Enhanced statistics analysis failed: {e}")
            return {
                "summary": {"error": str(e), "analysis_failed": True},
                "detailed_results": {},
                "key_findings": [f"Analysis failed: {str(e)}"],
                "confidence_score": 0.0,
                "recommendations": ["Check data format and requirements"],
                "metadata": {"module": "enhanced_statistics", "error": True},
            }

    def _multi_group_analysis(
        self, data: Dict[str, List[float]], **kwargs
    ) -> Dict[str, Any]:
        """Run multi-group bias analysis"""
        if not isinstance(data, dict):
            raise ValueError(
                "Multi-group analysis requires Dict[str, List[float]] format"
            )

        # Run multi-group analysis
        results = self.multi_group_analyzer.multi_group_bias_analysis(data)

        # Extract key findings
        key_findings = []
        if results["overall_test"]["significant"]:
            key_findings.append(
                f"Significant overall group differences detected (p={results['overall_test']['p_value']:.4f})"
            )

        # Count significant pairwise comparisons
        significant_pairs = sum(
            1
            for comp in results["pairwise_comparisons"].values()
            if comp.get("significant", False)
        )

        if significant_pairs > 0:
            key_findings.append(
                f"{significant_pairs} significant pairwise comparisons found"
            )

        # Identify large effect sizes
        large_effects = [
            name
            for name, effect in results["effect_sizes"].items()
            if effect.get("interpretation") in ["large", "medium"]
        ]

        if large_effects:
            key_findings.append(
                f"Large/medium effect sizes found: {', '.join(large_effects)}"
            )

        # Generate recommendations
        recommendations = []
        if results["overall_test"]["significant"]:
            recommendations.append("Investigate source of group differences")

        if large_effects:
            recommendations.append(
                "Focus attention on comparisons with large effect sizes"
            )

        if not key_findings:
            key_findings.append("No significant bias detected across groups")
            recommendations.append("Continue monitoring with larger sample sizes")

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(results)

        return {
            "summary": {
                "overall_significant": results["overall_test"]["significant"],
                "significant_pairs": significant_pairs,
                "large_effects_count": len(large_effects),
                "groups_tested": results["overall_test"]["groups_tested"],
            },
            "detailed_results": results,
            "key_findings": key_findings,
            "confidence_score": confidence_score,
            "recommendations": recommendations,
            "metadata": {
                "module": "enhanced_statistics",
                "analysis_type": "multi_group",
                "correction_method": kwargs.get("correction_method", "fdr_bh"),
            },
        }

    def _comprehensive_analysis(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Run comprehensive statistical analysis"""
        # Convert data to appropriate format
        if hasattr(data, "__iter__") and not isinstance(data, dict):
            # Assume it's a list of responses, convert to grouped format
            data = self._convert_to_grouped_format(data)

        if not isinstance(data, dict):
            raise ValueError("Comprehensive analysis requires grouped data format")

        # Run comprehensive analysis
        control_group = kwargs.get("control_group_name")
        correction_method = kwargs.get("correction_method", "fdr_bh")

        results = self.analyzer.comprehensive_bias_analysis(
            data, control_group_name=control_group, correction_method=correction_method
        )

        # Extract summary information
        summary = {
            "groups_analyzed": len(results.get("groups_analyzed", [])),
            "total_samples": sum(results.get("sample_sizes", {}).values()),
            "normality_passed": sum(
                1
                for norm in results.get("normality_tests", {}).values()
                if norm.get("is_normal", False)
            ),
        }

        # Add multi-group results if available
        if "multi_group_analysis" in results:
            mg_results = results["multi_group_analysis"]
            summary.update(
                {
                    "overall_significant": mg_results.get("overall_test", {}).get(
                        "significant", False
                    ),
                    "significant_pairs": sum(
                        1
                        for comp in mg_results.get("pairwise_comparisons", {}).values()
                        if comp.get("significant", False)
                    ),
                }
            )

        # Extract key findings
        key_findings = self._extract_key_findings(results)

        # Generate recommendations
        recommendations = self._generate_statistical_recommendations(results)

        # Calculate confidence
        confidence_score = self._calculate_comprehensive_confidence(results)

        return {
            "summary": summary,
            "detailed_results": results,
            "key_findings": key_findings,
            "confidence_score": confidence_score,
            "recommendations": recommendations,
            "metadata": {
                "module": "enhanced_statistics",
                "analysis_type": "comprehensive",
                "timestamp": results.get("timestamp"),
                "correction_method": correction_method,
            },
        }

    def _effect_size_analysis(
        self, data: Dict[str, List[float]], **kwargs
    ) -> Dict[str, Any]:
        """Focus on effect size calculations"""
        effect_results = {}
        key_findings = []

        group_names = list(data.keys())

        # Calculate pairwise effect sizes
        for i, group1 in enumerate(group_names):
            for group2 in group_names[i + 1 :]:
                try:
                    effect_size = self.effect_calculator.cohens_d(
                        data[group1], data[group2]
                    )
                    interpretation = self.effect_calculator.interpret_effect_size(
                        abs(effect_size)
                    )

                    comparison_key = f"{group1}_vs_{group2}"
                    effect_results[comparison_key] = {
                        "cohens_d": effect_size,
                        "interpretation": interpretation,
                        "practically_significant": abs(effect_size) >= 0.3,
                    }

                    if interpretation in ["large", "medium"]:
                        key_findings.append(
                            f"{interpretation.title()} effect size between {group1} and {group2} (d={effect_size:.3f})"
                        )

                except Exception as e:
                    logger.warning(
                        f"Effect size calculation failed for {group1} vs {group2}: {e}"
                    )

        # Generate recommendations based on effect sizes
        recommendations = []
        large_effects = [
            k for k, v in effect_results.items() if v["interpretation"] == "large"
        ]

        if large_effects:
            recommendations.append(
                "Large effect sizes detected - investigate practical significance"
            )

        if not key_findings:
            key_findings.append("No substantial effect sizes detected")
            recommendations.append("Current differences appear minimal")

        return {
            "summary": {
                "comparisons_made": len(effect_results),
                "large_effects": len(large_effects),
                "medium_effects": sum(
                    1
                    for v in effect_results.values()
                    if v["interpretation"] == "medium"
                ),
            },
            "detailed_results": {"effect_sizes": effect_results},
            "key_findings": key_findings,
            "confidence_score": 0.8 if effect_results else 0.5,
            "recommendations": recommendations,
            "metadata": {
                "module": "enhanced_statistics",
                "analysis_type": "effect_sizes",
            },
        }

    def validate_input(self, data: Any) -> Dict[str, Any]:
        """Enhanced input validation for statistical analysis"""
        validation = super().validate_input(data)

        try:
            if isinstance(data, dict):
                # Validate grouped data format
                for group_name, group_data in data.items():
                    if not isinstance(group_data, (list, np.ndarray)):
                        validation["errors"].append(
                            f"Group '{group_name}' data must be list or array"
                        )
                        validation["valid"] = False
                    elif len(group_data) < 3:
                        validation["warnings"].append(
                            f"Group '{group_name}' has very few samples ({len(group_data)})"
                        )

                if len(data) < 2:
                    validation["errors"].append("Need at least 2 groups for comparison")
                    validation["valid"] = False

            elif hasattr(data, "__iter__"):
                # Validate response list format
                if len(data) < self._module_info.requirements.min_samples:
                    validation["errors"].append(
                        f"Insufficient total samples: {len(data)}"
                    )
                    validation["valid"] = False

        except Exception as e:
            validation["errors"].append(f"Input validation error: {str(e)}")
            validation["valid"] = False

        return validation

    def _convert_to_grouped_format(self, responses) -> Dict[str, List[float]]:
        """Convert response list to grouped format"""
        # This is a placeholder - in real implementation, would extract
        # demographic information and group accordingly
        return {"all_responses": [getattr(r, "score", 0.0) for r in responses]}

    def _calculate_confidence_score(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score based on statistical results"""
        confidence = 0.5  # Base confidence

        # Boost confidence with significant results
        if results.get("overall_test", {}).get("significant", False):
            confidence += 0.2

        # Consider effect sizes
        large_effects = sum(
            1
            for effect in results.get("effect_sizes", {}).values()
            if effect.get("interpretation") == "large"
        )
        confidence += min(large_effects * 0.1, 0.3)

        return min(confidence, 1.0)

    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from comprehensive results"""
        findings = []

        # Overall test results
        if "multi_group_analysis" in results:
            mg = results["multi_group_analysis"]
            if mg.get("overall_test", {}).get("significant", False):
                findings.append("Significant overall group differences detected")

        # Normality test results
        if "normality_tests" in results:
            non_normal = sum(
                1
                for test in results["normality_tests"].values()
                if not test.get("is_normal", True)
            )
            if non_normal > 0:
                findings.append(
                    f"{non_normal} groups failed normality tests - using non-parametric methods"
                )

        # Multiple comparison results
        if "multiple_comparison_correction" in results:
            mcc = results["multiple_comparison_correction"]
            if "num_significant_corrected" in mcc:
                findings.append(
                    f"{mcc['num_significant_corrected']} comparisons remain significant after correction"
                )

        return (
            findings
            if findings
            else ["Statistical analysis completed - see detailed results"]
        )

    def _generate_statistical_recommendations(
        self, results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on statistical results"""
        recommendations = []

        # Sample size recommendations
        total_samples = sum(results.get("sample_sizes", {}).values())
        if total_samples < 50:
            recommendations.append(
                "Consider collecting more data for increased statistical power"
            )

        # Normality recommendations
        if "normality_tests" in results:
            non_normal_count = sum(
                1
                for test in results["normality_tests"].values()
                if not test.get("is_normal", True)
            )
            if non_normal_count > 0:
                recommendations.append(
                    "Non-parametric methods were used due to non-normal distributions"
                )

        # Significance recommendations
        if "multi_group_analysis" in results:
            if (
                results["multi_group_analysis"]
                .get("overall_test", {})
                .get("significant", False)
            ):
                recommendations.append("Investigate sources of group differences")
            else:
                recommendations.append(
                    "No significant group differences detected - continue monitoring"
                )

        return (
            recommendations
            if recommendations
            else ["Results appear statistically sound"]
        )

    def _calculate_comprehensive_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence for comprehensive analysis"""
        confidence = 0.6  # Base confidence for comprehensive analysis

        # Sample size factor
        total_samples = sum(results.get("sample_sizes", {}).values())
        if total_samples >= 100:
            confidence += 0.2
        elif total_samples >= 50:
            confidence += 0.1

        # Statistical significance factor
        if "multi_group_analysis" in results:
            if (
                results["multi_group_analysis"]
                .get("overall_test", {})
                .get("significant", False)
            ):
                confidence += 0.1

        # Multiple comparison correction factor
        if "multiple_comparison_correction" in results:
            confidence += 0.1  # Using proper correction methods

        return min(confidence, 1.0)
