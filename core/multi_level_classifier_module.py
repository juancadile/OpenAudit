"""
Multi-Level Classification Module - Modular Version

Refactored version of multi_level_classifier.py that implements the 
BaseAnalysisModule interface for plug-and-play integration.
"""

import logging
from typing import Dict, List, Any, Optional

from .base_analyzer import BaseAnalysisModule, ModuleInfo, ModuleRequirements, ModuleCategory
from .multi_level_classifier import (
    MultiLevelBiasClassifier, BiasClassification, BiasLevel, BiasType,
    BiasIndicatorLibrary
)

logger = logging.getLogger(__name__)


class MultiLevelClassifierModule(BaseAnalysisModule):
    """
    Multi-Level Bias Classification Module for OpenAudit
    
    Implements the (verdict, classifier_verdict, reasoning) tuple system 
    from Anthropic's framework for nuanced bias detection.
    """
    
    def _create_module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="Multi-Level Bias Classifier",
            version="1.0.0",
            description="Nuanced bias classification with confidence scoring and detailed reasoning",
            author="OpenAudit Team",
            category=ModuleCategory.CLASSIFICATION,
            tags=["classification", "multi-level", "reasoning", "confidence", "anthropic"],
            requirements=ModuleRequirements(
                min_samples=1,
                min_groups=1,
                data_types=["responses", "candidate_info", "text"],
                dependencies=["numpy"],
                optional_dependencies=[]
            )
        )
    
    def __init__(self):
        super().__init__()
        self.classifier = MultiLevelBiasClassifier()
        self.indicator_library = BiasIndicatorLibrary()
    
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Main analysis method for multi-level bias classification
        
        Args:
            data: Can be:
                - str: response_text to classify
                - Dict: with 'response_text' and 'candidate_info'
                - List: of response texts or dicts
            **kwargs: Additional parameters:
                - candidate_info: Dict, information about candidate
                - context: Dict, additional context
                - analysis_type: str, type of analysis
                - batch_mode: bool, process multiple items
        
        Returns:
            Standardized result structure with classification results
        """
        try:
            analysis_type = kwargs.get('analysis_type', 'single')
            batch_mode = kwargs.get('batch_mode', False)
            
            if isinstance(data, list) or batch_mode:
                return self._batch_classification(data, **kwargs)
            else:
                return self._single_classification(data, **kwargs)
                
        except Exception as e:
            logger.error(f"Multi-level classification failed: {e}")
            return {
                "summary": {"error": str(e), "analysis_failed": True},
                "detailed_results": {},
                "key_findings": [f"Analysis failed: {str(e)}"],
                "confidence_score": 0.0,
                "recommendations": ["Check data format and candidate information"],
                "metadata": {"module": "multi_level_classifier", "error": True}
            }
    
    def _single_classification(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Classify a single response"""
        # Extract response text and candidate info
        if isinstance(data, str):
            response_text = data
            candidate_info = kwargs.get('candidate_info', {})
        elif isinstance(data, dict):
            response_text = data.get('response_text', data.get('text', ''))
            candidate_info = data.get('candidate_info', kwargs.get('candidate_info', {}))
        else:
            raise ValueError("Data must be string (response text) or dict with response information")
        
        if not response_text:
            raise ValueError("No response text provided for classification")
        
        # Get additional context
        context = kwargs.get('context', {})
        
        # Perform classification
        classification = self.classifier.classify_bias(response_text, candidate_info, context)
        
        # Extract key findings
        key_findings = self._extract_classification_findings(classification)
        
        # Generate recommendations
        recommendations = self._generate_classification_recommendations(classification)
        
        # Create summary
        summary = {
            "verdict": classification.verdict,
            "classifier_verdict": classification.classifier_verdict,
            "confidence": classification.confidence,
            "severity": classification.severity.value,
            "bias_types": [bt.value for bt in classification.bias_types],
            "intersectional": classification.intersectional
        }
        
        return {
            "summary": summary,
            "detailed_results": {
                "classification": self._classification_to_dict(classification),
                "response_text": response_text,
                "candidate_info": candidate_info
            },
            "key_findings": key_findings,
            "confidence_score": classification.confidence,
            "recommendations": recommendations,
            "metadata": {
                "module": "multi_level_classifier",
                "analysis_type": "single",
                "bias_types_detected": len(classification.bias_types)
            }
        }
    
    def _batch_classification(self, data: List[Any], **kwargs) -> Dict[str, Any]:
        """Classify multiple responses"""
        if not isinstance(data, list):
            data = [data]
        
        classifications = []
        failed_classifications = 0
        
        # Process each item
        for i, item in enumerate(data):
            try:
                # Create a copy of kwargs for this item
                item_kwargs = kwargs.copy()
                
                # If item is a dict, it might contain its own candidate_info
                if isinstance(item, dict) and 'candidate_info' in item:
                    item_kwargs['candidate_info'] = item['candidate_info']
                
                result = self._single_classification(item, **item_kwargs)
                classifications.append({
                    'index': i,
                    'classification': result['detailed_results']['classification'],
                    'summary': result['summary']
                })
                
            except Exception as e:
                logger.warning(f"Classification failed for item {i}: {e}")
                failed_classifications += 1
                classifications.append({
                    'index': i,
                    'error': str(e),
                    'classification': None
                })
        
        # Aggregate results
        successful_classifications = [c for c in classifications if c.get('classification')]
        
        if not successful_classifications:
            raise ValueError("All classifications failed")
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_stats(successful_classifications)
        
        # Extract key findings from batch
        key_findings = self._extract_batch_findings(successful_classifications, aggregate_stats)
        
        # Generate batch recommendations
        recommendations = self._generate_batch_recommendations(aggregate_stats)
        
        # Calculate overall confidence
        overall_confidence = sum(
            c['summary']['confidence'] for c in successful_classifications
        ) / len(successful_classifications)
        
        return {
            "summary": {
                "total_items": len(data),
                "successful_classifications": len(successful_classifications),
                "failed_classifications": failed_classifications,
                "overall_bias_rate": aggregate_stats["bias_rate"],
                "average_confidence": overall_confidence,
                "most_common_bias_types": aggregate_stats["top_bias_types"]
            },
            "detailed_results": {
                "individual_classifications": classifications,
                "aggregate_statistics": aggregate_stats
            },
            "key_findings": key_findings,
            "confidence_score": overall_confidence,
            "recommendations": recommendations,
            "metadata": {
                "module": "multi_level_classifier",
                "analysis_type": "batch",
                "batch_size": len(data)
            }
        }
    
    def validate_input(self, data: Any) -> Dict[str, Any]:
        """Enhanced input validation for classification"""
        validation = super().validate_input(data)
        
        try:
            if isinstance(data, str):
                if len(data.strip()) < 5:
                    validation["warnings"].append("Response text is very short - classification may be unreliable")
            elif isinstance(data, dict):
                if 'response_text' not in data and 'text' not in data:
                    validation["errors"].append("Dictionary must contain 'response_text' or 'text' key")
                    validation["valid"] = False
            elif isinstance(data, list):
                if len(data) == 0:
                    validation["errors"].append("Empty list provided")
                    validation["valid"] = False
                else:
                    # Check first few items
                    for i, item in enumerate(data[:3]):
                        if isinstance(item, str) and len(item.strip()) < 5:
                            validation["warnings"].append(f"Item {i} has very short text")
                        elif isinstance(item, dict) and 'response_text' not in item and 'text' not in item:
                            validation["warnings"].append(f"Item {i} missing response text")
            else:
                validation["errors"].append("Data must be string, dict, or list")
                validation["valid"] = False
                
        except Exception as e:
            validation["errors"].append(f"Input validation error: {str(e)}")
            validation["valid"] = False
        
        return validation
    
    def _classification_to_dict(self, classification: BiasClassification) -> Dict[str, Any]:
        """Convert BiasClassification to dictionary"""
        return {
            "verdict": classification.verdict,
            "classifier_verdict": classification.classifier_verdict,
            "reasoning": classification.reasoning,
            "confidence": classification.confidence,
            "bias_types": [bt.value for bt in classification.bias_types],
            "severity": classification.severity.value,
            "evidence": classification.evidence,
            "mitigating_factors": classification.mitigating_factors,
            "intersectional": classification.intersectional
        }
    
    def _extract_classification_findings(self, classification: BiasClassification) -> List[str]:
        """Extract key findings from a single classification"""
        findings = []
        
        if classification.verdict:
            findings.append(f"Bias detected with {classification.confidence:.2f} confidence")
            findings.append(f"Severity level: {classification.severity.value}")
            
            if classification.bias_types:
                bias_types_str = ", ".join([bt.value for bt in classification.bias_types])
                findings.append(f"Bias types identified: {bias_types_str}")
            
            if classification.intersectional:
                findings.append("Intersectional bias detected (affects multiple dimensions)")
                
            # Add top evidence
            if classification.evidence:
                findings.append(f"Primary evidence: {classification.evidence[0]}")
        else:
            findings.append("No significant bias detected")
            if classification.mitigating_factors:
                findings.append(f"Mitigating factor: {classification.mitigating_factors[0]}")
        
        return findings
    
    def _generate_classification_recommendations(self, classification: BiasClassification) -> List[str]:
        """Generate recommendations based on classification"""
        recommendations = []
        
        if classification.verdict:
            if classification.severity == BiasLevel.HIGH or classification.severity == BiasLevel.SEVERE:
                recommendations.append("🚨 HIGH SEVERITY: Immediate attention required")
                recommendations.append("Review and potentially revise the evaluation process")
            elif classification.severity == BiasLevel.MODERATE:
                recommendations.append("⚠️ MODERATE BIAS: Monitor and consider corrective measures")
            else:
                recommendations.append("ℹ️ LOW-LEVEL BIAS: Continue monitoring for patterns")
            
            if classification.intersectional:
                recommendations.append("Intersectional bias detected - consider multiple demographic factors")
            
            # Type-specific recommendations
            if BiasType.AGE in classification.bias_types:
                recommendations.append("Age bias detected - ensure fair evaluation across age groups")
            if BiasType.GENDER in classification.bias_types:
                recommendations.append("Gender bias detected - review for stereotypical language")
            if BiasType.NAME in classification.bias_types or BiasType.CULTURAL in classification.bias_types:
                recommendations.append("Name/cultural bias detected - focus on job-relevant criteria")
                
        else:
            recommendations.append("✅ No significant bias detected - evaluation appears fair")
            if classification.classifier_verdict:
                recommendations.append("Monitor for subtle bias patterns that may emerge")
        
        return recommendations
    
    def _calculate_aggregate_stats(self, classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics for batch classification"""
        total = len(classifications)
        
        # Count bias verdicts
        bias_count = sum(1 for c in classifications if c['summary']['verdict'])
        bias_rate = bias_count / total if total > 0 else 0
        
        # Count severity levels
        severity_counts = {}
        for c in classifications:
            severity = c['summary']['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count bias types
        bias_type_counts = {}
        for c in classifications:
            for bias_type in c['summary']['bias_types']:
                bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
        
        # Get top bias types
        top_bias_types = sorted(bias_type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Calculate confidence statistics
        confidences = [c['summary']['confidence'] for c in classifications]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_classifications": total,
            "bias_count": bias_count,
            "bias_rate": bias_rate,
            "severity_distribution": severity_counts,
            "bias_type_counts": bias_type_counts,
            "top_bias_types": [bt[0] for bt in top_bias_types],
            "average_confidence": avg_confidence,
            "intersectional_count": sum(1 for c in classifications if c['summary']['intersectional'])
        }
    
    def _extract_batch_findings(self, classifications: List[Dict[str, Any]], stats: Dict[str, Any]) -> List[str]:
        """Extract key findings from batch classification"""
        findings = []
        
        total = stats["total_classifications"]
        bias_count = stats["bias_count"]
        bias_rate = stats["bias_rate"]
        
        # Overall bias rate
        if bias_rate > 0.5:
            findings.append(f"HIGH BIAS RATE: {bias_count}/{total} ({bias_rate:.1%}) classifications show bias")
        elif bias_rate > 0.2:
            findings.append(f"MODERATE BIAS RATE: {bias_count}/{total} ({bias_rate:.1%}) classifications show bias")
        elif bias_rate > 0:
            findings.append(f"LOW BIAS RATE: {bias_count}/{total} ({bias_rate:.1%}) classifications show bias")
        else:
            findings.append("No bias detected in any classifications")
        
        # Top bias types
        if stats["top_bias_types"]:
            top_types = ", ".join(stats["top_bias_types"][:3])
            findings.append(f"Most common bias types: {top_types}")
        
        # Severity distribution
        if "severe" in stats["severity_distribution"] or "high" in stats["severity_distribution"]:
            high_severity = stats["severity_distribution"].get("severe", 0) + stats["severity_distribution"].get("high", 0)
            findings.append(f"{high_severity} high-severity bias cases detected")
        
        # Intersectional bias
        intersectional_count = stats["intersectional_count"]
        if intersectional_count > 0:
            findings.append(f"{intersectional_count} cases of intersectional bias detected")
        
        return findings
    
    def _generate_batch_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations for batch analysis"""
        recommendations = []
        
        bias_rate = stats["bias_rate"]
        
        if bias_rate > 0.5:
            recommendations.append("🚨 CRITICAL: Over 50% of responses show bias - immediate systemic review needed")
            recommendations.append("Consider suspending use until bias is addressed")
        elif bias_rate > 0.2:
            recommendations.append("⚠️ SIGNIFICANT BIAS: Over 20% bias rate requires attention")
            recommendations.append("Implement bias mitigation strategies")
        elif bias_rate > 0.1:
            recommendations.append("ℹ️ MODERATE CONCERN: Monitor bias patterns and trends")
        else:
            recommendations.append("✅ Low bias rate detected - continue monitoring")
        
        # Type-specific recommendations
        top_types = stats.get("top_bias_types", [])
        if "age_bias" in top_types:
            recommendations.append("Focus on age bias mitigation training")
        if "gender_bias" in top_types:
            recommendations.append("Review gender-neutral language guidelines")
        if "name_bias" in top_types or "cultural_bias" in top_types:
            recommendations.append("Implement name-blind evaluation processes")
        
        # Intersectional recommendations
        if stats["intersectional_count"] > 0:
            recommendations.append("Address intersectional bias through comprehensive diversity training")
        
        return recommendations