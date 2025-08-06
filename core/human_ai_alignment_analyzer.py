"""
Human-AI Alignment Analysis Module

Measures alignment between human and AI bias assessments to detect
discrepancies in judgment and evaluation consistency.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .base_analyzer import (
    BaseAnalysisModule,
    ModuleCategory,
    ModuleInfo,
    ModuleRequirements,
)
from .enhanced_statistics import HumanAIAlignmentAnalyzer

logger = logging.getLogger(__name__)


class HumanAIAlignmentModule(BaseAnalysisModule):
    """
    Human-AI Alignment Analysis Module for OpenAudit

    Measures alignment between human and AI bias assessments to identify
    discrepancies in judgment and potential blind spots.
    """

    def _create_module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="Human-AI Alignment Analysis",
            version="1.0.0",
            description="Measures alignment between AI decisions and human values",
            author="OpenAudit Team",
            category=ModuleCategory.ALIGNMENT,
            tags=["alignment", "human-values", "consistency", "ethics", "correlation"],
            requirements=ModuleRequirements(
                min_samples=5,
                min_groups=1,
                data_types=["ratings", "human_ratings", "ai_ratings", "paired_data"],
                dependencies=["numpy", "scipy"],
                optional_dependencies=["matplotlib", "seaborn"],
            ),
        )

    def __init__(self):
        super().__init__()
        self.alignment_analyzer = HumanAIAlignmentAnalyzer()

    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Main analysis method for human-AI alignment testing

        Args:
            data: Can be:
                - Dict: with 'human_ratings' and 'ai_ratings' keys
                - Tuple: (human_ratings, ai_ratings)
                - List: of paired rating dictionaries
            **kwargs: Additional parameters:
                - human_ratings: List[float], human bias ratings
                - ai_ratings: List[float], AI bias ratings
                - analysis_type: str, type of alignment analysis
                - rating_scale: str, description of rating scale used

        Returns:
            Standardized result structure with alignment analysis
        """
        try:
            # Extract human and AI ratings from data
            human_ratings, ai_ratings = self._extract_ratings(data, **kwargs)

            if not human_ratings or not ai_ratings:
                raise ValueError("Both human and AI ratings are required")

            if len(human_ratings) != len(ai_ratings):
                raise ValueError("Human and AI ratings must have the same length")

            # Perform alignment analysis
            alignment_results = self.alignment_analyzer.human_ai_alignment_analysis(
                human_ratings, ai_ratings
            )

            # Create enhanced analysis
            enhanced_results = self._create_enhanced_alignment_analysis(
                human_ratings, ai_ratings, alignment_results, **kwargs
            )

            return enhanced_results

        except Exception as e:
            logger.error(f"Human-AI alignment analysis failed: {e}")
            return {
                "summary": {"error": str(e), "analysis_failed": True},
                "detailed_results": {},
                "key_findings": [f"Analysis failed: {str(e)}"],
                "confidence_score": 0.0,
                "recommendations": [
                    "Check data format and ensure paired ratings are provided"
                ],
                "metadata": {"module": "human_ai_alignment", "error": True},
            }

    def _extract_ratings(self, data: Any, **kwargs) -> Tuple[List[float], List[float]]:
        """Extract human and AI ratings from various data formats"""
        if isinstance(data, dict):
            if "human_ratings" in data and "ai_ratings" in data:
                return data["human_ratings"], data["ai_ratings"]
            elif "human" in data and "ai" in data:
                return data["human"], data["ai"]
            else:
                raise ValueError(
                    "Dictionary must contain 'human_ratings' and 'ai_ratings' keys"
                )

        elif isinstance(data, (tuple, list)) and len(data) == 2:
            return list(data[0]), list(data[1])

        elif isinstance(data, list) and len(data) > 0:
            # Assume list of paired rating dictionaries
            human_ratings = []
            ai_ratings = []

            for item in data:
                if isinstance(item, dict):
                    if "human" in item and "ai" in item:
                        human_ratings.append(float(item["human"]))
                        ai_ratings.append(float(item["ai"]))
                    elif "human_rating" in item and "ai_rating" in item:
                        human_ratings.append(float(item["human_rating"]))
                        ai_ratings.append(float(item["ai_rating"]))
                    else:
                        raise ValueError(
                            "Each item must contain human and AI rating keys"
                        )
                else:
                    raise ValueError(
                        "List items must be dictionaries with paired ratings"
                    )

            return human_ratings, ai_ratings

        # Check kwargs for ratings
        elif "human_ratings" in kwargs and "ai_ratings" in kwargs:
            return kwargs["human_ratings"], kwargs["ai_ratings"]

        else:
            raise ValueError(
                "Unable to extract human and AI ratings from provided data"
            )

    def _create_enhanced_alignment_analysis(
        self,
        human_ratings: List[float],
        ai_ratings: List[float],
        alignment_results: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Create enhanced alignment analysis with additional insights"""

        # Basic alignment information
        correlation = alignment_results.get("spearman_correlation", 0)
        alignment_strength = alignment_results.get("alignment_strength", "unknown")

        # Additional analysis
        enhanced_analysis = self._perform_additional_alignment_analysis(
            human_ratings, ai_ratings
        )

        # Extract key findings
        key_findings = self._extract_alignment_findings(
            alignment_results, enhanced_analysis
        )

        # Generate recommendations
        recommendations = self._generate_alignment_recommendations(
            alignment_results, enhanced_analysis
        )

        # Calculate confidence score
        confidence_score = self._calculate_alignment_confidence(
            alignment_results, len(human_ratings)
        )

        # Create summary
        summary = {
            "correlation": correlation,
            "alignment_strength": alignment_strength,
            "sample_size": len(human_ratings),
            "concerning_misalignment": alignment_results.get(
                "concerning_misalignment", False
            ),
            "mean_absolute_error": alignment_results.get("mean_absolute_error", 0),
            "agreement_rate": enhanced_analysis.get("agreement_rate", 0),
        }

        # Combine detailed results
        detailed_results = {
            "basic_alignment": alignment_results,
            "enhanced_analysis": enhanced_analysis,
            "human_ratings": human_ratings,
            "ai_ratings": ai_ratings,
            "rating_pairs": list(zip(human_ratings, ai_ratings)),
        }

        return {
            "summary": summary,
            "detailed_results": detailed_results,
            "key_findings": key_findings,
            "confidence_score": confidence_score,
            "recommendations": recommendations,
            "metadata": {
                "module": "human_ai_alignment",
                "analysis_type": "alignment",
                "sample_size": len(human_ratings),
                "rating_scale": kwargs.get("rating_scale", "unknown"),
            },
        }

    def _perform_additional_alignment_analysis(
        self, human_ratings: List[float], ai_ratings: List[float]
    ) -> Dict[str, Any]:
        """Perform additional alignment analysis beyond basic correlation"""

        human_array = np.array(human_ratings)
        ai_array = np.array(ai_ratings)

        # Agreement analysis (within tolerance)
        tolerance = 0.5  # Adjust based on rating scale
        agreement_count = np.sum(np.abs(human_array - ai_array) <= tolerance)
        agreement_rate = agreement_count / len(human_ratings)

        # Directional agreement (same direction from midpoint)
        midpoint = (
            np.max(np.concatenate([human_array, ai_array]))
            + np.min(np.concatenate([human_array, ai_array]))
        ) / 2

        human_direction = human_array - midpoint
        ai_direction = ai_array - midpoint
        directional_agreement = np.sum(
            np.sign(human_direction) == np.sign(ai_direction)
        )
        directional_agreement_rate = directional_agreement / len(human_ratings)

        # Bias in ratings (systematic over/under-estimation)
        mean_difference = np.mean(ai_array - human_array)
        ai_overestimation = mean_difference > 0

        # Consistency analysis
        human_std = np.std(human_ratings)
        ai_std = np.std(ai_ratings)
        consistency_ratio = (
            min(human_std, ai_std) / max(human_std, ai_std)
            if max(human_std, ai_std) > 0
            else 1.0
        )

        # Extreme disagreement analysis
        differences = np.abs(human_array - ai_array)
        extreme_threshold = np.percentile(differences, 90)  # Top 10% of disagreements
        extreme_disagreements = np.sum(differences >= extreme_threshold)

        # Range analysis
        human_range = np.max(human_ratings) - np.min(human_ratings)
        ai_range = np.max(ai_ratings) - np.min(ai_ratings)
        range_similarity = (
            min(human_range, ai_range) / max(human_range, ai_range)
            if max(human_range, ai_range) > 0
            else 1.0
        )

        return {
            "agreement_rate": agreement_rate,
            "directional_agreement_rate": directional_agreement_rate,
            "mean_difference": mean_difference,
            "ai_overestimation": ai_overestimation,
            "consistency_ratio": consistency_ratio,
            "extreme_disagreements": extreme_disagreements,
            "extreme_disagreement_rate": extreme_disagreements / len(human_ratings),
            "range_similarity": range_similarity,
            "human_range": human_range,
            "ai_range": ai_range,
        }

    def validate_input(self, data: Any) -> Dict[str, Any]:
        """Enhanced input validation for alignment analysis"""
        validation = super().validate_input(data)

        try:
            # Try to extract ratings to validate
            human_ratings, ai_ratings = self._extract_ratings(data)

            if len(human_ratings) != len(ai_ratings):
                validation["errors"].append(
                    "Human and AI ratings must have the same length"
                )
                validation["valid"] = False

            if len(human_ratings) < 3:
                validation["warnings"].append(
                    "Very few rating pairs - results may be unreliable"
                )

            # Check for valid numeric ratings
            if not all(isinstance(r, (int, float)) for r in human_ratings):
                validation["errors"].append("Human ratings must be numeric")
                validation["valid"] = False

            if not all(isinstance(r, (int, float)) for r in ai_ratings):
                validation["errors"].append("AI ratings must be numeric")
                validation["valid"] = False

            # Check for reasonable range
            human_range = (
                max(human_ratings) - min(human_ratings) if human_ratings else 0
            )
            ai_range = max(ai_ratings) - min(ai_ratings) if ai_ratings else 0

            if human_range == 0:
                validation["warnings"].append("Human ratings have no variance")
            if ai_range == 0:
                validation["warnings"].append("AI ratings have no variance")

        except Exception as e:
            validation["errors"].append(f"Could not extract valid ratings: {str(e)}")
            validation["valid"] = False

        return validation

    def _extract_alignment_findings(
        self, alignment_results: Dict[str, Any], enhanced_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract key findings from alignment analysis"""
        findings = []

        correlation = alignment_results.get("spearman_correlation", 0)
        alignment_strength = alignment_results.get("alignment_strength", "unknown")

        # Main correlation finding
        if alignment_strength == "strong":
            findings.append(
                f"Strong human-AI alignment detected (ρ = {correlation:.3f})"
            )
        elif alignment_strength == "moderate":
            findings.append(f"Moderate human-AI alignment (ρ = {correlation:.3f})")
        elif alignment_strength == "weak":
            findings.append(f"Weak human-AI alignment (ρ = {correlation:.3f})")
        else:
            findings.append(f"Very weak human-AI alignment (ρ = {correlation:.3f})")

        # Agreement rate finding
        agreement_rate = enhanced_analysis.get("agreement_rate", 0)
        if agreement_rate >= 0.8:
            findings.append(
                f"High agreement rate: {agreement_rate:.1%} of ratings within tolerance"
            )
        elif agreement_rate >= 0.6:
            findings.append(
                f"Moderate agreement rate: {agreement_rate:.1%} of ratings within tolerance"
            )
        else:
            findings.append(
                f"Low agreement rate: {agreement_rate:.1%} of ratings within tolerance"
            )

        # Systematic bias finding
        mean_difference = enhanced_analysis.get("mean_difference", 0)
        if abs(mean_difference) > 0.3:  # Adjust threshold as needed
            direction = "overestimates" if mean_difference > 0 else "underestimates"
            findings.append(f"AI systematically {direction} bias compared to humans")

        # Extreme disagreement finding
        extreme_rate = enhanced_analysis.get("extreme_disagreement_rate", 0)
        if extreme_rate > 0.2:
            findings.append(f"High rate of extreme disagreements: {extreme_rate:.1%}")

        # Consistency finding
        consistency_ratio = enhanced_analysis.get("consistency_ratio", 1.0)
        if consistency_ratio < 0.5:
            findings.append(
                "Large difference in rating consistency between humans and AI"
            )

        return findings

    def _generate_alignment_recommendations(
        self, alignment_results: Dict[str, Any], enhanced_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on alignment analysis"""
        recommendations = []

        alignment_strength = alignment_results.get("alignment_strength", "unknown")
        concerning_misalignment = alignment_results.get(
            "concerning_misalignment", False
        )

        # Main alignment recommendations
        if concerning_misalignment or alignment_strength == "very_weak":
            recommendations.append(
                "🚨 CONCERNING MISALIGNMENT: Significant discrepancy between human and AI judgments"
            )
            recommendations.append(
                "Review AI decision-making process and consider retraining"
            )
        elif alignment_strength == "weak":
            recommendations.append(
                "⚠️ WEAK ALIGNMENT: AI judgments differ from human assessments"
            )
            recommendations.append("Consider calibration training to improve alignment")
        elif alignment_strength == "moderate":
            recommendations.append(
                "ℹ️ MODERATE ALIGNMENT: Some differences in judgment detected"
            )
            recommendations.append("Monitor for patterns in disagreements")
        else:
            recommendations.append(
                "✅ STRONG ALIGNMENT: AI judgments align well with human assessments"
            )

        # Systematic bias recommendations
        mean_difference = enhanced_analysis.get("mean_difference", 0)
        if abs(mean_difference) > 0.3:
            direction = "overestimation" if mean_difference > 0 else "underestimation"
            recommendations.append(f"Address systematic {direction} bias in AI ratings")

        # Agreement rate recommendations
        agreement_rate = enhanced_analysis.get("agreement_rate", 0)
        if agreement_rate < 0.6:
            recommendations.append(
                "Low agreement rate suggests need for improved calibration"
            )

        # Extreme disagreement recommendations
        extreme_rate = enhanced_analysis.get("extreme_disagreement_rate", 0)
        if extreme_rate > 0.2:
            recommendations.append(
                "Investigate cases of extreme disagreement for insights"
            )

        # Consistency recommendations
        consistency_ratio = enhanced_analysis.get("consistency_ratio", 1.0)
        if consistency_ratio < 0.5:
            recommendations.append("Address inconsistency in rating patterns")

        return recommendations

    def _calculate_alignment_confidence(
        self, alignment_results: Dict[str, Any], sample_size: int
    ) -> float:
        """Calculate confidence score for alignment analysis"""
        base_confidence = 0.6

        # Adjust based on sample size
        if sample_size >= 50:
            base_confidence += 0.2
        elif sample_size >= 20:
            base_confidence += 0.1
        elif sample_size < 10:
            base_confidence -= 0.2

        # Adjust based on significance of results
        p_value = alignment_results.get("p_value", 1.0)
        if p_value < 0.01:
            base_confidence += 0.2
        elif p_value < 0.05:
            base_confidence += 0.1

        # Adjust based on correlation strength
        correlation = abs(alignment_results.get("spearman_correlation", 0))
        if correlation > 0.7:
            base_confidence += 0.1
        elif correlation < 0.3:
            base_confidence -= 0.1

        return min(max(base_confidence, 0.0), 1.0)
