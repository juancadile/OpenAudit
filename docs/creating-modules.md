# 🧩 Creating Analysis Modules

Learn how to create custom analysis modules for OpenAudit. Extend the platform with your own bias detection algorithms, research methods, and specialized analysis techniques.

## 🎯 **Overview**

OpenAudit's modular architecture makes it easy to:
- 🔬 **Implement custom bias detection algorithms**
- 📊 **Add new statistical methods**
- 🌍 **Create domain-specific analysis**
- 🔄 **Share modules with the community that facilitate reproducible results**

## 🏗️ **Module Architecture**

### Module Interface

All analysis modules inherit from `BaseAnalysisModule`:

```python
from core.base_analyzer import (
    BaseAnalysisModule, 
    ModuleInfo, 
    ModuleRequirements, 
    ModuleCategory
)

class MyCustomModule(BaseAnalysisModule):
    def _create_module_info(self) -> ModuleInfo:
        """Define module metadata"""
        pass
    
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Main analysis implementation"""
        pass
    
    def get_requirements(self) -> Dict[str, Any]:
        """Return module requirements"""
        pass
    
    def get_supported_data_types(self) -> List[str]:
        """Return supported input data types"""
        pass
```

### Standard Result Format

All modules must return results in this format:

```python
{
    "summary": {
        "bias_detected": bool,
        "confidence_score": float,  # 0.0 to 1.0
        "total_samples": int,
        "custom_metric": Any
    },
    "detailed_results": {
        "analysis_details": Dict,
        "statistical_tests": Dict,
        "intermediate_results": Dict
    },
    "key_findings": [
        "Human-readable finding 1",
        "Human-readable finding 2"
    ],
    "confidence_score": float,  # Overall confidence
    "recommendations": [
        "Actionable recommendation 1",
        "Actionable recommendation 2"
    ],
    "metadata": {
        "module": "module_name",
        "version": "1.0.0",
        "parameters": Dict,
        "execution_time": float
    }
}
```

## 🚀 **Quick Start: Your First Module**

### Step 1: Generate Template

```bash
# Use CLI to generate a module template
openaudit modules create sentiment_analyzer \
    --author "Your Name" \
    --description "Sentiment-based bias detection" \
    --category custom

# This creates:
# - external_modules/sentiment_analyzer.py
# - external_modules/sentiment_analyzer.manifest.json
```

### Step 2: Implement Analysis Logic

```python
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from core.base_analyzer import BaseAnalysisModule, ModuleInfo, ModuleRequirements, ModuleCategory

logger = logging.getLogger(__name__)

class SentimentBiasAnalyzer(BaseAnalysisModule):
    """
    Sentiment-based bias detection module.
    
    Analyzes sentiment differences in AI responses across demographic groups
    to detect potential bias in language tone and positivity.
    """
    
    def _create_module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="Sentiment Bias Analyzer",
            version="1.0.0",
            description="Detects bias through sentiment analysis of AI responses",
            author="Your Name",
            category=ModuleCategory.CUSTOM,
            tags=["sentiment", "nlp", "bias", "language"],
            requirements=ModuleRequirements(
                min_samples=10,
                min_groups=2,
                data_types=["responses", "text"],
                dependencies=["numpy", "pandas", "textblob"],
                optional_dependencies=["vaderSentiment", "matplotlib"]
            )
        )
    
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Analyze sentiment bias in responses
        
        Args:
            data: List of LLMResponse objects or pandas DataFrame
            **kwargs: Additional parameters
                - sentiment_method: 'textblob' or 'vader' (default: 'textblob')
                - alpha: Significance level (default: 0.05)
        """
        logger.info("Starting sentiment bias analysis")
        
        # Extract parameters
        sentiment_method = kwargs.get('sentiment_method', 'textblob')
        alpha = kwargs.get('alpha', 0.05)
        
        # Convert data to DataFrame if needed
        if hasattr(data, '__iter__') and not isinstance(data, pd.DataFrame):
            df = self._responses_to_dataframe(data)
        else:
            df = data.copy()
        
        # Perform sentiment analysis
        sentiment_scores = self._calculate_sentiment_scores(df, sentiment_method)
        df['sentiment_score'] = sentiment_scores
        
        # Analyze bias across demographics
        bias_results = self._analyze_sentiment_bias(df, alpha)
        
        # Generate summary
        summary = self._generate_summary(df, bias_results)
        
        # Create detailed results
        detailed_results = {
            "sentiment_analysis": {
                "method": sentiment_method,
                "overall_sentiment": {
                    "mean": float(np.mean(sentiment_scores)),
                    "std": float(np.std(sentiment_scores)),
                    "median": float(np.median(sentiment_scores))
                }
            },
            "demographic_analysis": bias_results,
            "statistical_tests": self._run_statistical_tests(df, alpha)
        }
        
        # Generate findings and recommendations
        key_findings = self._generate_key_findings(summary, bias_results)
        recommendations = self._generate_recommendations(summary, bias_results)
        
        return {
            "summary": summary,
            "detailed_results": detailed_results,
            "key_findings": key_findings,
            "confidence_score": summary.get("confidence_score", 0.5),
            "recommendations": recommendations,
            "metadata": {
                "module": "sentiment_bias_analyzer",
                "version": "1.0.0",
                "parameters": {
                    "sentiment_method": sentiment_method,
                    "alpha": alpha
                }
            }
        }
    
    def _calculate_sentiment_scores(self, df: pd.DataFrame, method: str) -> List[float]:
        """Calculate sentiment scores for each response"""
        scores = []
        
        if method == 'textblob':
            from textblob import TextBlob
            for response in df['response']:
                blob = TextBlob(str(response))
                scores.append(blob.sentiment.polarity)
        
        elif method == 'vader':
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                for response in df['response']:
                    score = analyzer.polarity_scores(str(response))
                    scores.append(score['compound'])
            except ImportError:
                logger.warning("VADER not available, falling back to TextBlob")
                return self._calculate_sentiment_scores(df, 'textblob')
        
        return scores
    
    def _analyze_sentiment_bias(self, df: pd.DataFrame, alpha: float) -> Dict[str, Any]:
        """Analyze sentiment bias across demographic groups"""
        results = {}
        
        # Get demographic columns
        demo_cols = [col for col in df.columns if col in ['race', 'gender', 'ethnicity', 'age_group']]
        
        for demo_col in demo_cols:
            if demo_col not in df.columns:
                continue
                
            group_sentiments = df.groupby(demo_col)['sentiment_score'].agg(['mean', 'std', 'count'])
            
            # Calculate bias gap (difference between highest and lowest mean sentiment)
            max_sentiment = group_sentiments['mean'].max()
            min_sentiment = group_sentiments['mean'].min()
            bias_gap = max_sentiment - min_sentiment
            
            # Statistical significance test (ANOVA)
            from scipy.stats import f_oneway
            groups = [group['sentiment_score'].values for name, group in df.groupby(demo_col)]
            f_stat, p_value = f_oneway(*groups)
            
            results[demo_col] = {
                "group_sentiments": group_sentiments.to_dict(),
                "bias_gap": float(bias_gap),
                "bias_percentage": float(bias_gap / abs(min_sentiment) * 100) if min_sentiment != 0 else 0,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "effect_size": self._calculate_effect_size(groups)
            }
        
        return results
    
    def _calculate_effect_size(self, groups: List[np.ndarray]) -> float:
        """Calculate eta-squared effect size for ANOVA"""
        # Simple eta-squared calculation
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        ss_total = sum((x - grand_mean)**2 for x in all_data)
        
        return ss_between / ss_total if ss_total > 0 else 0
    
    def _run_statistical_tests(self, df: pd.DataFrame, alpha: float) -> Dict[str, Any]:
        """Run additional statistical tests"""
        from scipy.stats import normaltest, levene
        
        tests = {}
        
        # Normality test
        stat, p_val = normaltest(df['sentiment_score'])
        tests['normality'] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "normal": p_val > alpha
        }
        
        # Equal variance test
        demo_cols = [col for col in df.columns if col in ['race', 'gender']]
        if demo_cols:
            demo_col = demo_cols[0]  # Use first available
            groups = [group['sentiment_score'].values for name, group in df.groupby(demo_col)]
            if len(groups) > 1:
                stat, p_val = levene(*groups)
                tests['equal_variance'] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "equal_variance": p_val > alpha
                }
        
        return tests
    
    def _generate_summary(self, df: pd.DataFrame, bias_results: Dict) -> Dict[str, Any]:
        """Generate analysis summary"""
        # Determine if bias is detected
        significant_biases = [demo for demo, result in bias_results.items() if result.get('significant', False)]
        bias_detected = len(significant_biases) > 0
        
        # Calculate overall confidence
        p_values = [result.get('p_value', 1.0) for result in bias_results.values()]
        confidence_score = 1.0 - min(p_values) if p_values else 0.5
        
        return {
            "bias_detected": bias_detected,
            "significant_demographics": significant_biases,
            "total_samples": len(df),
            "confidence_score": float(confidence_score),
            "overall_sentiment_range": {
                "min": float(df['sentiment_score'].min()),
                "max": float(df['sentiment_score'].max()),
                "mean": float(df['sentiment_score'].mean())
            }
        }
    
    def _generate_key_findings(self, summary: Dict, bias_results: Dict) -> List[str]:
        """Generate human-readable key findings"""
        findings = []
        
        if summary["bias_detected"]:
            findings.append(f"Sentiment bias detected across {len(summary['significant_demographics'])} demographic group(s)")
            
            for demo, result in bias_results.items():
                if result.get('significant', False):
                    bias_gap = result['bias_gap']
                    findings.append(f"{demo.title()} shows {bias_gap:.3f} sentiment bias gap (p < {result['p_value']:.3f})")
        else:
            findings.append("No significant sentiment bias detected across demographic groups")
        
        # Add sentiment analysis insights
        overall_sentiment = summary["overall_sentiment_range"]["mean"]
        if overall_sentiment > 0.1:
            findings.append("Overall response sentiment is positive")
        elif overall_sentiment < -0.1:
            findings.append("Overall response sentiment is negative")
        else:
            findings.append("Overall response sentiment is neutral")
        
        return findings
    
    def _generate_recommendations(self, summary: Dict, bias_results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if summary["bias_detected"]:
            recommendations.extend([
                "Review model training data for sentiment bias patterns",
                "Implement sentiment normalization in post-processing",
                "Monitor sentiment consistency across demographic groups",
                "Consider bias mitigation techniques during fine-tuning"
            ])
            
            # Specific recommendations based on effect size
            for demo, result in bias_results.items():
                if result.get('significant', False):
                    effect_size = result.get('effect_size', 0)
                    if effect_size > 0.14:  # Large effect
                        recommendations.append(f"Prioritize addressing {demo} sentiment bias (large effect size: {effect_size:.3f})")
        else:
            recommendations.extend([
                "Continue monitoring sentiment patterns in future evaluations",
                "Maintain current model performance regarding sentiment consistency"
            ])
        
        return recommendations
    
    def _responses_to_dataframe(self, responses) -> pd.DataFrame:
        """Convert LLMResponse objects to DataFrame"""
        data = []
        for response in responses:
            row = {
                "response": response.response,
                "model": response.model_name,
                "timestamp": response.timestamp
            }
            # Add demographic info if available
            if hasattr(response, 'metadata') and response.metadata:
                row.update(response.metadata)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_requirements(self) -> Dict[str, Any]:
        """Return module requirements"""
        return self.module_info.requirements.__dict__
    
    def get_supported_data_types(self) -> List[str]:
        """Return supported data types"""
        return ["responses", "dataframe", "text"]
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data"""
        if isinstance(data, pd.DataFrame):
            return 'response' in data.columns
        elif hasattr(data, '__iter__'):
            return len(data) > 0
        return False
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if dependencies are available"""
        deps = {}
        
        try:
            import textblob
            deps["textblob"] = True
        except ImportError:
            deps["textblob"] = False
        
        try:
            import vaderSentiment
            deps["vaderSentiment"] = True
        except ImportError:
            deps["vaderSentiment"] = False
        
        return deps
```

### Step 3: Test Your Module

```python
# test_sentiment_analyzer.py
import pytest
import pandas as pd
from sentiment_analyzer import SentimentBiasAnalyzer

class TestSentimentBiasAnalyzer:
    def setup_method(self):
        self.analyzer = SentimentBiasAnalyzer()
    
    def test_module_info(self):
        """Test module metadata"""
        info = self.analyzer.module_info
        assert info.name == "Sentiment Bias Analyzer"
        assert info.category.value == "custom"
    
    def test_sentiment_analysis(self):
        """Test sentiment calculation"""
        # Create test data
        test_data = pd.DataFrame({
            'response': [
                "I'm very excited about this opportunity!",
                "This seems like an okay choice.",
                "I have serious concerns about this decision."
            ],
            'race': ['white', 'black', 'hispanic'],
            'gender': ['male', 'female', 'male']
        })
        
        # Run analysis
        results = self.analyzer.analyze(test_data)
        
        # Assertions
        assert "summary" in results
        assert "detailed_results" in results
        assert "key_findings" in results
        assert isinstance(results["summary"]["bias_detected"], bool)
    
    def test_requirements_validation(self):
        """Test requirements are properly defined"""
        reqs = self.analyzer.get_requirements()
        assert reqs["min_samples"] == 10
        assert "textblob" in reqs["dependencies"]
```

### Step 4: Register and Use

```python
from core.module_registry import register_module
from sentiment_analyzer import SentimentBiasAnalyzer

# Register the module
success = register_module("sentiment_analyzer", SentimentBiasAnalyzer)
print(f"Module registered: {success}")

# Use in analysis
from core.bias_testing_framework import BiasAnalyzer

analyzer = BiasAnalyzer(responses)
results = analyzer.run_modular_analysis(["sentiment_analyzer"])
```

## 📊 **Advanced Module Examples**

### Statistical Testing Module

```python
class AdvancedStatisticalModule(BaseAnalysisModule):
    """Advanced statistical testing with multiple methods"""
    
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        # Multiple testing correction
        correction_method = kwargs.get('correction_method', 'fdr_bh')
        
        # Run multiple statistical tests
        tests = {
            'chi_square': self._chi_square_test(data),
            'fishers_exact': self._fishers_exact_test(data),
            'mcnemar': self._mcnemar_test(data),
            'permutation': self._permutation_test(data)
        }
        
        # Apply multiple comparison correction
        corrected_results = self._apply_correction(tests, correction_method)
        
        return {
            "summary": self._generate_summary(corrected_results),
            "detailed_results": {
                "statistical_tests": corrected_results,
                "correction_method": correction_method
            },
            "key_findings": self._interpret_results(corrected_results),
            "confidence_score": self._calculate_confidence(corrected_results),
            "recommendations": self._generate_recommendations(corrected_results)
        }
```

### Machine Learning Module

```python
class MLBiasDetectorModule(BaseAnalysisModule):
    """Machine learning-based bias detection"""
    
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        
        # Feature extraction
        features = self._extract_features(data)
        
        # Train bias detection model
        model = RandomForestClassifier()
        X, y = self._prepare_training_data(features)
        model.fit(X, y)
        
        # Predict bias
        predictions = model.predict_proba(X)
        
        # Feature importance analysis
        feature_importance = dict(zip(features.columns, model.feature_importances_))
        
        return {
            "summary": {
                "bias_detected": bool(predictions.max() > 0.7),
                "confidence_score": float(predictions.max()),
                "model_accuracy": self._calculate_accuracy(model, X, y)
            },
            "detailed_results": {
                "predictions": predictions.tolist(),
                "feature_importance": feature_importance,
                "classification_report": classification_report(y, model.predict(X), output_dict=True)
            },
            "key_findings": self._interpret_ml_results(predictions, feature_importance),
            "confidence_score": float(predictions.max()),
            "recommendations": self._generate_ml_recommendations(feature_importance)
        }
```

### Network Analysis Module

```python
class NetworkBiasModule(BaseAnalysisModule):
    """Network analysis for bias detection"""
    
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        import networkx as nx
        
        # Build response similarity network
        G = self._build_response_network(data)
        
        # Detect communities (demographic clustering)
        communities = self._detect_communities(G)
        
        # Calculate network metrics
        metrics = {
            'modularity': nx.community.modularity(G, communities),
            'clustering_coefficient': nx.average_clustering(G),
            'assortativity': nx.degree_assortativity_coefficient(G)
        }
        
        # Analyze bias through network structure
        bias_analysis = self._analyze_network_bias(G, communities, data)
        
        return {
            "summary": {
                "bias_detected": bias_analysis['significant_clustering'],
                "network_modularity": metrics['modularity'],
                "confidence_score": bias_analysis['confidence']
            },
            "detailed_results": {
                "network_metrics": metrics,
                "communities": [list(community) for community in communities],
                "bias_analysis": bias_analysis
            },
            "key_findings": self._interpret_network_results(metrics, bias_analysis),
            "confidence_score": bias_analysis['confidence'],
            "recommendations": self._generate_network_recommendations(bias_analysis)
        }
```

## 🔧 **Module Configuration**

### Configurable Parameters

```python
class ConfigurableModule(BaseAnalysisModule):
    """Module with extensive configuration options"""
    
    def get_configuration_options(self) -> Dict[str, Any]:
        """Return all configurable parameters"""
        return {
            "algorithm": {
                "type": "choice",
                "choices": ["method_a", "method_b", "method_c"],
                "default": "method_a",
                "description": "Analysis algorithm to use",
                "help": "Different algorithms provide different trade-offs between speed and accuracy"
            },
            "threshold": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "description": "Bias detection threshold",
                "help": "Lower values are more sensitive but may produce false positives"
            },
            "bootstrap_samples": {
                "type": "integer",
                "default": 1000,
                "min": 100,
                "max": 10000,
                "description": "Number of bootstrap samples for confidence intervals"
            },
            "enable_preprocessing": {
                "type": "boolean",
                "default": True,
                "description": "Enable data preprocessing",
                "help": "Preprocessing can improve accuracy but increases computation time"
            },
            "custom_weights": {
                "type": "dict",
                "default": {"accuracy": 0.6, "fairness": 0.4},
                "description": "Custom weights for multi-objective optimization",
                "schema": {
                    "type": "object",
                    "properties": {
                        "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                        "fairness": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            }
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters"""
        errors = []
        warnings = []
        
        # Validate weights sum to 1
        if "custom_weights" in config:
            weights = config["custom_weights"]
            if abs(sum(weights.values()) - 1.0) > 0.01:
                errors.append("Custom weights must sum to 1.0")
        
        # Check performance implications
        if config.get("bootstrap_samples", 0) > 5000:
            warnings.append("High bootstrap samples may significantly increase computation time")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
```

### Environment-Specific Behavior

```python
class AdaptiveModule(BaseAnalysisModule):
    """Module that adapts to different environments"""
    
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        # Detect environment
        environment = self._detect_environment()
        
        # Adapt algorithm based on environment
        if environment["memory_limited"]:
            return self._memory_efficient_analysis(data, **kwargs)
        elif environment["gpu_available"]:
            return self._gpu_accelerated_analysis(data, **kwargs)
        else:
            return self._standard_analysis(data, **kwargs)
    
    def _detect_environment(self) -> Dict[str, Any]:
        """Detect current execution environment"""
        import psutil
        
        return {
            "memory_limited": psutil.virtual_memory().available < 4 * 1024**3,  # < 4GB
            "gpu_available": self._check_gpu_availability(),
            "cpu_count": psutil.cpu_count(),
            "is_production": os.getenv("OPENAUDIT_ENV") == "production"
        }
```

## 🔍 **Testing & Validation**

### Comprehensive Test Suite

```python
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

class TestMyModule:
    """Comprehensive test suite for custom module"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample test data"""
        return pd.DataFrame({
            'response': [
                "Positive response for candidate",
                "Negative response for candidate",
                "Neutral response for candidate"
            ] * 10,
            'race': ['white', 'black', 'hispanic'] * 10,
            'gender': ['male', 'female'] * 15,
            'decision': ['hire', 'reject'] * 15
        })
    
    @pytest.fixture
    def module(self):
        """Create module instance"""
        return MyCustomModule()
    
    def test_module_interface_compliance(self, module):
        """Test module follows interface requirements"""
        # Test required methods exist
        assert hasattr(module, 'analyze')
        assert hasattr(module, 'get_requirements')
        assert hasattr(module, 'get_supported_data_types')
        
        # Test module info is properly defined
        info = module.module_info
        assert info.name
        assert info.version
        assert info.description
        assert info.author
    
    def test_analysis_with_valid_data(self, module, sample_data):
        """Test analysis with valid input data"""
        results = module.analyze(sample_data)
        
        # Test result structure
        required_keys = ['summary', 'detailed_results', 'key_findings', 'confidence_score', 'recommendations', 'metadata']
        for key in required_keys:
            assert key in results
        
        # Test result types
        assert isinstance(results['summary'], dict)
        assert isinstance(results['key_findings'], list)
        assert isinstance(results['confidence_score'], (int, float))
        assert 0 <= results['confidence_score'] <= 1
    
    def test_analysis_with_minimal_data(self, module):
        """Test analysis with minimal valid data"""
        minimal_data = pd.DataFrame({
            'response': ['test response 1', 'test response 2'],
            'race': ['white', 'black']
        })
        
        results = module.analyze(minimal_data)
        assert results is not None
    
    def test_analysis_with_invalid_data(self, module):
        """Test module handles invalid data gracefully"""
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        # Should either return error in results or raise appropriate exception
        try:
            results = module.analyze(invalid_data)
            # If no exception, check for error indication in results
            if 'error' not in results:
                assert results['summary']['bias_detected'] is not None
        except (ValueError, KeyError):
            # Expected for invalid data
            pass
    
    def test_parameter_handling(self, module, sample_data):
        """Test module handles different parameters correctly"""
        # Test with custom parameters
        results1 = module.analyze(sample_data, alpha=0.01, custom_param=True)
        results2 = module.analyze(sample_data, alpha=0.1, custom_param=False)
        
        # Results should vary based on parameters
        assert results1['metadata']['parameters']['alpha'] == 0.01
        assert results2['metadata']['parameters']['alpha'] == 0.1
    
    def test_reproducibility(self, module, sample_data):
        """Test analysis results are reproducible"""
        results1 = module.analyze(sample_data, random_seed=42)
        results2 = module.analyze(sample_data, random_seed=42)
        
        # Key metrics should be identical
        assert results1['confidence_score'] == results2['confidence_score']
    
    def test_performance(self, module, sample_data):
        """Test module performance"""
        import time
        
        start_time = time.time()
        results = module.analyze(sample_data)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert execution_time < 30  # 30 seconds max
        
        # Should handle larger datasets efficiently
        large_data = pd.concat([sample_data] * 10)
        start_time = time.time()
        results = module.analyze(large_data)
        large_execution_time = time.time() - start_time
        
        # Should scale reasonably
        assert large_execution_time < execution_time * 20  # Not more than 20x slower
    
    def test_edge_cases(self, module):
        """Test edge cases and boundary conditions"""
        # Single sample
        single_sample = pd.DataFrame({
            'response': ['single response'],
            'race': ['white']
        })
        
        # Should handle gracefully
        results = module.analyze(single_sample)
        assert results is not None
        
        # Empty data
        empty_data = pd.DataFrame(columns=['response', 'race'])
        
        try:
            results = module.analyze(empty_data)
            # Should indicate no analysis possible
            assert not results['summary']['bias_detected']
        except ValueError:
            # Acceptable to raise error for empty data
            pass
    
    @patch('module.external_dependency')
    def test_dependency_handling(self, mock_dependency, module, sample_data):
        """Test handling of external dependencies"""
        # Mock dependency failure
        mock_dependency.side_effect = ImportError("Dependency not available")
        
        # Should handle gracefully or provide alternative
        results = module.analyze(sample_data)
        assert results is not None
    
    def test_memory_usage(self, module, sample_data):
        """Test memory usage stays reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run analysis
        results = module.analyze(sample_data)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory (< 100MB for small dataset)
        assert memory_increase < 100 * 1024 * 1024
```

### Integration Testing

```python
def test_module_integration():
    """Test module integrates properly with OpenAudit"""
    from core.module_registry import get_global_registry
    from core.bias_testing_framework import BiasAnalyzer
    
    # Register module
    registry = get_global_registry()
    success = registry.register_module("test_module", MyCustomModule)
    assert success
    
    # Use in analysis pipeline
    responses = generate_test_responses()
    analyzer = BiasAnalyzer(responses)
    
    # Should work with modular analysis
    results = analyzer.run_modular_analysis(["test_module"])
    assert "test_module" in results["module_results"]
    
    # Should work with profiles
    results = analyzer.run_modular_analysis(profile="custom_profile")
    assert results["success"]
```

## 📦 **Module Distribution**

### Package Structure

```
my_bias_module/
├── __init__.py
├── analyzer.py              # Main module code
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py    # Test suite
├── examples/
│   └── usage_example.py    # Usage examples
├── setup.py               # Package configuration
└── manifest.json         # Module manifest
```

### Manifest File

```json
{
  "name": "sentiment_bias_analyzer",
  "version": "1.0.0",
  "description": "Sentiment-based bias detection for OpenAudit",
  "author": "Your Name",
  "email": "your.email@example.com",
  "license": "MIT",
  "module_class": "SentimentBiasAnalyzer",
  "file_path": "analyzer.py",
  "category": "custom",
  "tags": ["sentiment", "nlp", "bias"],
  "requirements": {
    "min_samples": 10,
    "min_groups": 2,
    "dependencies": ["numpy", "pandas", "textblob"],
    "optional_dependencies": ["vaderSentiment"]
  },
  "compatibility": {
    "openaudit_version": ">=1.0.0",
    "python_version": ">=3.8"
  },
  "github_url": "https://github.com/yourusername/sentiment-bias-analyzer",
  "documentation_url": "https://github.com/yourusername/sentiment-bias-analyzer/blob/main/README.md"
}
```

### Setup Configuration

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="openaudit-sentiment-analyzer",
    version="1.0.0",
    description="Sentiment-based bias detection module for OpenAudit",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "textblob>=0.15.0"
    ],
    extras_require={
        "enhanced": ["vaderSentiment>=3.3.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.0.0"]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "openaudit.modules": [
            "sentiment_analyzer = sentiment_analyzer.analyzer:SentimentBiasAnalyzer"
        ]
    }
)
```

## 🌍 **Community & Sharing**

### Publishing to GitHub

```bash
# Create repository
git init
git add .
git commit -m "Initial commit: Sentiment bias analyzer module"
git remote add origin https://github.com/yourusername/sentiment-bias-analyzer
git push -u origin main

# Tag release
git tag v1.0.0
git push origin v1.0.0
```

### Loading from Community

```python
# Users can load your module from GitHub
from core.external_module_loader import get_global_module_loader

loader = get_global_module_loader()
result = loader.install_module_from_github(
    "https://github.com/yourusername/sentiment-bias-analyzer",
    module_name="sentiment_analyzer"
)

if result["success"]:
    print("Module installed successfully!")
    
    # Use immediately
    from core.bias_testing_framework import BiasAnalyzer
    analyzer = BiasAnalyzer(responses)
    results = analyzer.run_modular_analysis(["sentiment_analyzer"])
```

## 🏆 **Best Practices**

### Code Quality

1. **Follow PEP 8** styling guidelines
2. **Add type hints** for better code clarity
3. **Include docstrings** for all public methods
4. **Handle errors gracefully** with informative messages
5. **Use logging** instead of print statements
6. **Validate inputs** and provide clear error messages

### Performance

1. **Use vectorized operations** with NumPy/Pandas
2. **Cache expensive computations** when appropriate
3. **Provide progress callbacks** for long-running operations
4. **Optimize memory usage** for large datasets
5. **Profile your code** to identify bottlenecks

### Reliability

1. **Write comprehensive tests** with good coverage
2. **Handle edge cases** (empty data, single samples, etc.)
3. **Validate dependencies** and provide fallbacks
4. **Use reproducible random seeds** when applicable
5. **Document limitations** and assumptions

### Usability

1. **Provide clear documentation** with examples
2. **Use descriptive parameter names** and defaults
3. **Generate actionable recommendations**
4. **Include configuration options** for flexibility
5. **Follow the standard result format**

## 🎯 **Next Steps**

Ready to build your own module?

1. **🚀 Start with the template**: `openaudit modules create your_module`
2. **📚 Study existing modules**: Check out the built-in modules for reference
3. **🧪 Test thoroughly**: Write comprehensive tests for your module
4. **📖 Document well**: Create clear documentation and examples
5. **🤝 Share with community**: Publish to GitHub and share with others

## 📞 **Get Help**

Need assistance creating your module?

- 💬 **Discord**: [Join our developer community](https://discord.gg/openaudit-dev)
- 📚 **Docs**: [Developer Guide](developer-guide.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/openaudit/openaudit/issues)
- 📧 **Email**: modules@openaudit.org

**Happy module building! 🎉** 