"""
Enhanced Statistical Analysis Module for OpenAudit

Implements advanced statistical methods from Cornell's cross-cultural ableism study
and Anthropic's agentic misalignment framework for robust bias detection.

Key Features:
- Multi-group statistical testing with effect sizes
- Cultural context sensitivity analysis
- Language register bias detection
- Human-AI alignment measurement
- Multiple comparison correction
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, shapiro, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


class EffectSizeCalculator:
    """Calculate and interpret effect sizes for various statistical tests"""

    @staticmethod
    def cohens_d(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d for two groups"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
            / (n1 + n2 - 2)
        )
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    @staticmethod
    def r_from_z(z_stat: float, n: int) -> float:
        """Calculate effect size r from z-statistic"""
        return abs(z_stat) / np.sqrt(n)

    @staticmethod
    def r_from_u(u_stat: float, n1: int, n2: int) -> float:
        """Calculate effect size r from Mann-Whitney U statistic"""
        z_score = (u_stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        return abs(z_score) / np.sqrt(n1 + n2)

    @staticmethod
    def interpret_effect_size(r: float) -> str:
        """Interpret effect size magnitude following Cohen's conventions"""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"


class MultiGroupBiasAnalyzer:
    """Advanced multi-group bias analysis with effect size reporting"""

    def __init__(self):
        self.effect_calculator = EffectSizeCalculator()

    def multi_group_bias_analysis(
        self, responses_by_group: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Compare bias across multiple demographic groups simultaneously

        Args:
            responses_by_group: Dict mapping group names to response lists

        Returns:
            Dict with Kruskal-Wallis results and pairwise effect sizes
        """
        if len(responses_by_group) < 2:
            raise ValueError("Need at least 2 groups for comparison")

        # Kruskal-Wallis test for overall group differences
        groups = list(responses_by_group.values())
        group_names = list(responses_by_group.keys())

        try:
            kw_statistic, kw_p_value = kruskal(*groups)
        except ValueError as e:
            logger.warning(f"Kruskal-Wallis test failed: {e}")
            kw_statistic, kw_p_value = np.nan, 1.0

        # Pairwise comparisons with effect sizes
        pairwise_results = {}
        effect_sizes = {}

        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i + 1 :], i + 1):
                try:
                    # Mann-Whitney U test
                    u_stat, mw_p = mannwhitneyu(
                        responses_by_group[group1],
                        responses_by_group[group2],
                        alternative="two-sided",
                    )

                    # Effect size
                    n1, n2 = len(responses_by_group[group1]), len(
                        responses_by_group[group2]
                    )
                    r = self.effect_calculator.r_from_u(u_stat, n1, n2)

                    comparison_key = f"{group1}_vs_{group2}"

                    pairwise_results[comparison_key] = {
                        "u_statistic": u_stat,
                        "p_value": mw_p,
                        "significant": mw_p < 0.05,
                    }

                    effect_sizes[comparison_key] = {
                        "r_value": r,
                        "interpretation": self.effect_calculator.interpret_effect_size(
                            r
                        ),
                        "practically_significant": r >= 0.3,
                    }

                except Exception as e:
                    logger.warning(
                        f"Pairwise comparison failed for {group1} vs {group2}: {e}"
                    )
                    continue

        return {
            "overall_test": {
                "kruskal_wallis_statistic": kw_statistic,
                "p_value": kw_p_value,
                "significant": kw_p_value < 0.05,
                "groups_tested": len(responses_by_group),
            },
            "pairwise_comparisons": pairwise_results,
            "effect_sizes": effect_sizes,
            "group_statistics": {
                name: {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "n": len(values),
                }
                for name, values in responses_by_group.items()
            },
        }

    def enhanced_bias_test_with_effect_sizes(
        self, control_group: List[float], test_groups: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Enhanced version of bias testing with comprehensive effect size reporting

        Args:
            control_group: Baseline group for comparison
            test_groups: Dict mapping group names to test data

        Returns:
            Comprehensive results with statistical tests, effect sizes, and priorities
        """
        results = {
            "statistical_tests": {},
            "effect_sizes": {},
            "practical_significance": {},
        }

        for group_name, test_group in test_groups.items():
            try:
                # Check if we can use paired or unpaired test
                if len(control_group) == len(test_group):
                    # Use paired Wilcoxon signed-rank test
                    z_stat, p_val = wilcoxon(control_group, test_group)
                    test_type = "wilcoxon_signed_rank"
                    n = len(control_group)
                    r = self.effect_calculator.r_from_z(z_stat, n)
                else:
                    # Use unpaired Mann-Whitney U test
                    u_stat, p_val = mannwhitneyu(
                        control_group, test_group, alternative="two-sided"
                    )
                    test_type = "mann_whitney_u"
                    r = self.effect_calculator.r_from_u(
                        u_stat, len(control_group), len(test_group)
                    )
                    z_stat = u_stat  # For consistency in reporting

                # Interpret effect size
                interpretation = self.effect_calculator.interpret_effect_size(r)

                results["statistical_tests"][group_name] = {
                    "test_type": test_type,
                    "statistic": z_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                }

                results["effect_sizes"][group_name] = {
                    "r_value": r,
                    "interpretation": interpretation,
                    "practically_significant": r >= 0.3,
                }

                # Priority classification for remediation
                if p_val < 0.05 and r >= 0.5:
                    priority = "high"
                elif p_val < 0.05 and 0.3 <= r < 0.5:
                    priority = "medium"
                elif p_val < 0.05 and r < 0.3:
                    priority = "low"
                else:
                    priority = "none"

                results["practical_significance"][group_name] = {
                    "priority": priority,
                    "requires_immediate_attention": priority == "high",
                    "bias_magnitude": interpretation,
                }

            except Exception as e:
                logger.error(f"Statistical test failed for group {group_name}: {e}")
                # Record failure but continue with other groups
                results["statistical_tests"][group_name] = {
                    "error": str(e),
                    "test_failed": True,
                }

        return results


class CulturalContextAnalyzer:
    """Analyze bias sensitivity to cultural and demographic contexts"""

    def __init__(self):
        self.effect_calculator = EffectSizeCalculator()

    async def cultural_context_analysis(
        self,
        base_responses: List[float],
        contextualized_responses: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Test how models respond to same scenario with different cultural contexts

        Args:
            base_responses: Baseline responses without context
            contextualized_responses: Dict mapping context descriptions to responses

        Returns:
            Analysis of cultural context sensitivity
        """
        results = {}

        for context, responses in contextualized_responses.items():
            try:
                if len(base_responses) != len(responses):
                    logger.warning(f"Mismatched sample sizes for context {context}")
                    continue

                # Paired comparison since same underlying prompts
                z_stat, p_val = wilcoxon(base_responses, responses)
                n = len(base_responses)
                r = self.effect_calculator.r_from_z(z_stat, n)

                # Direction of effect
                mean_base = np.mean(base_responses)
                mean_context = np.mean(responses)
                direction = (
                    "context_increases_bias"
                    if mean_context > mean_base
                    else "context_decreases_bias"
                )

                results[context] = {
                    "z_statistic": z_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                    "effect_size": r,
                    "interpretation": self.effect_calculator.interpret_effect_size(r),
                    "direction": direction,
                    "mean_difference": mean_context - mean_base,
                    "concerning": p_val < 0.05 and r >= 0.3,  # Medium+ effect size
                }

            except Exception as e:
                logger.error(f"Cultural context analysis failed for {context}: {e}")
                results[context] = {"error": str(e), "analysis_failed": True}

        return results


class LanguageRegisterAnalyzer:
    """Analyze bias differences between formal and informal language registers"""

    def __init__(self):
        self.effect_calculator = EffectSizeCalculator()

    def language_register_bias_test(
        self, formal_scores: List[float], casual_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Test if models show bias based on formality levels

        Args:
            formal_scores: Bias scores for formal language
            casual_scores: Bias scores for casual language

        Returns:
            Analysis of language register bias
        """
        if len(formal_scores) != len(casual_scores):
            raise ValueError("Formal and casual score lists must have same length")

        try:
            # Paired comparison since same underlying content
            z_stat, p_val = wilcoxon(formal_scores, casual_scores)

            # Effect size calculation
            n = len(formal_scores)
            r = self.effect_calculator.r_from_z(z_stat, n)

            # Direction analysis
            mean_formal = np.mean(formal_scores)
            mean_casual = np.mean(casual_scores)

            return {
                "register_bias_detected": p_val < 0.05,
                "z_statistic": z_stat,
                "p_value": p_val,
                "effect_size": r,
                "interpretation": self.effect_calculator.interpret_effect_size(r),
                "direction": (
                    "formal_favored" if mean_formal > mean_casual else "casual_favored"
                ),
                "mean_difference": mean_formal - mean_casual,
                "practical_significance": r > 0.3,  # Medium to large effect
                "concerning_bias": p_val < 0.05 and r >= 0.3,
                "formal_stats": {
                    "mean": mean_formal,
                    "std": np.std(formal_scores),
                    "median": np.median(formal_scores),
                },
                "casual_stats": {
                    "mean": mean_casual,
                    "std": np.std(casual_scores),
                    "median": np.median(casual_scores),
                },
            }

        except Exception as e:
            logger.error(f"Language register analysis failed: {e}")
            return {"error": str(e), "analysis_failed": True}


class HumanAIAlignmentAnalyzer:
    """Measure alignment between human and AI bias assessments"""

    def human_ai_alignment_analysis(
        self, human_ratings: List[float], ai_ratings: List[float]
    ) -> Dict[str, Any]:
        """
        Measure how well AI bias assessments align with human judgment

        Args:
            human_ratings: Human bias ratings
            ai_ratings: AI bias ratings

        Returns:
            Alignment analysis results
        """
        if len(human_ratings) != len(ai_ratings):
            raise ValueError("Human and AI rating lists must have same length")

        try:
            # Spearman rank correlation (robust to outliers)
            correlation, p_value = spearmanr(human_ratings, ai_ratings)

            # Pearson correlation for comparison
            pearson_corr = np.corrcoef(human_ratings, ai_ratings)[0, 1]

            # Alignment strength interpretation
            abs_corr = abs(correlation)
            if abs_corr > 0.7:
                strength = "strong"
            elif abs_corr > 0.5:
                strength = "moderate"
            elif abs_corr > 0.3:
                strength = "weak"
            else:
                strength = "very_weak"

            # Calculate mean absolute error
            mae = np.mean(np.abs(np.array(human_ratings) - np.array(ai_ratings)))

            return {
                "spearman_correlation": correlation,
                "pearson_correlation": pearson_corr,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "alignment_strength": strength,
                "concerning_misalignment": abs_corr < 0.3 and p_value < 0.05,
                "mean_absolute_error": mae,
                "human_stats": {
                    "mean": np.mean(human_ratings),
                    "std": np.std(human_ratings),
                    "range": (min(human_ratings), max(human_ratings)),
                },
                "ai_stats": {
                    "mean": np.mean(ai_ratings),
                    "std": np.std(ai_ratings),
                    "range": (min(ai_ratings), max(ai_ratings)),
                },
            }

        except Exception as e:
            logger.error(f"Human-AI alignment analysis failed: {e}")
            return {"error": str(e), "analysis_failed": True}


class MultipleComparisonCorrector:
    """Handle multiple comparison corrections to prevent false discoveries"""

    @staticmethod
    def corrected_multiple_testing(
        p_values: List[float], method: str = "fdr_bh", alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Apply multiple testing correction to prevent false discoveries

        Args:
            p_values: List of p-values to correct
            method: Correction method ('bonferroni', 'fdr_bh', etc.)
            alpha: Significance level

        Returns:
            Correction results
        """
        try:
            reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method=method
            )

            return {
                "original_p_values": p_values,
                "corrected_p_values": p_corrected.tolist(),
                "significant_after_correction": reject.tolist(),
                "correction_method": method,
                "alpha_level": alpha,
                "bonferroni_alpha": alpha_bonf,
                "sidak_alpha": alpha_sidak,
                "num_significant_original": sum(p < alpha for p in p_values),
                "num_significant_corrected": sum(reject),
            }

        except Exception as e:
            logger.error(f"Multiple comparison correction failed: {e}")
            return {"error": str(e), "correction_failed": True}


class EnhancedStatisticalAnalyzer:
    """
    Main class combining all enhanced statistical analysis capabilities
    """

    def __init__(self):
        self.multi_group_analyzer = MultiGroupBiasAnalyzer()
        self.cultural_analyzer = CulturalContextAnalyzer()
        self.register_analyzer = LanguageRegisterAnalyzer()
        self.alignment_analyzer = HumanAIAlignmentAnalyzer()
        self.correction_handler = MultipleComparisonCorrector()

    def check_normality(self, data: List[float]) -> Dict[str, Any]:
        """Check if data follows normal distribution"""
        try:
            statistic, p_value = shapiro(data)
            return {
                "shapiro_statistic": statistic,
                "p_value": p_value,
                "is_normal": p_value > 0.05,
                "recommendation": (
                    "use_parametric" if p_value > 0.05 else "use_nonparametric"
                ),
            }
        except Exception as e:
            logger.warning(f"Normality test failed: {e}")
            return {"error": str(e), "recommendation": "use_nonparametric"}

    def comprehensive_bias_analysis(
        self,
        responses_by_group: Dict[str, List[float]],
        control_group_name: str = None,
        correction_method: str = "fdr_bh",
    ) -> Dict[str, Any]:
        """
        Run comprehensive bias analysis with all enhancements

        Args:
            responses_by_group: Dict mapping group names to response lists
            control_group_name: Name of control group (if any)
            correction_method: Multiple comparison correction method

        Returns:
            Comprehensive analysis results
        """
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "groups_analyzed": list(responses_by_group.keys()),
            "sample_sizes": {
                name: len(data) for name, data in responses_by_group.items()
            },
        }

        # Check normality for each group
        normality_results = {}
        for name, data in responses_by_group.items():
            normality_results[name] = self.check_normality(data)
        results["normality_tests"] = normality_results

        # Multi-group analysis
        try:
            multi_group_results = self.multi_group_analyzer.multi_group_bias_analysis(
                responses_by_group
            )
            results["multi_group_analysis"] = multi_group_results
        except Exception as e:
            logger.error(f"Multi-group analysis failed: {e}")
            results["multi_group_analysis"] = {"error": str(e)}

        # Enhanced bias testing with effect sizes
        if control_group_name and control_group_name in responses_by_group:
            control_data = responses_by_group[control_group_name]
            test_groups = {
                k: v for k, v in responses_by_group.items() if k != control_group_name
            }

            try:
                enhanced_results = (
                    self.multi_group_analyzer.enhanced_bias_test_with_effect_sizes(
                        control_data, test_groups
                    )
                )
                results["enhanced_bias_testing"] = enhanced_results
            except Exception as e:
                logger.error(f"Enhanced bias testing failed: {e}")
                results["enhanced_bias_testing"] = {"error": str(e)}

        # Multiple comparison correction
        if (
            "multi_group_analysis" in results
            and "pairwise_comparisons" in results["multi_group_analysis"]
        ):
            p_values = [
                comp["p_value"]
                for comp in results["multi_group_analysis"][
                    "pairwise_comparisons"
                ].values()
                if "p_value" in comp
            ]

            if p_values:
                try:
                    correction_results = (
                        self.correction_handler.corrected_multiple_testing(
                            p_values, method=correction_method
                        )
                    )
                    results["multiple_comparison_correction"] = correction_results
                except Exception as e:
                    logger.error(f"Multiple comparison correction failed: {e}")

        return results
