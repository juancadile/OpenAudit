"""
Pytest configuration and fixtures for OpenAudit tests
"""

from typing import Any, Dict

import pytest

from core.analysis_profiles import ProfileManager, clear_global_profile_manager
from core.base_analyzer import BaseAnalysisModule
from core.bias_testing_framework import HiringBiasTest, LLMResponse
from core.cv_templates import CVTemplates
from core.module_registry import AnalysisModuleRegistry, clear_global_registry


@pytest.fixture(scope="session")
def bias_test():
    """Create a HiringBiasTest instance for testing."""
    return HiringBiasTest()


@pytest.fixture(scope="session")
def sample_variables():
    """Standard variables for testing CV generation."""
    return {
        "name": "John Smith",
        "university": "State University",
        "experience": "2",
        "address": "123 Main St, Anytown, USA",
    }


@pytest.fixture(scope="session")
def demographic_test_cases(bias_test):
    """Generate test cases for all demographic groups."""
    return bias_test.create_test_cases("software_engineer")


@pytest.fixture
def cv_templates():
    """CV Templates instance for testing."""
    return CVTemplates


# ============================================================================
# MODULAR SYSTEM FIXTURES
# ============================================================================


@pytest.fixture
def clean_modular_state():
    """Clean up modular system state before and after tests"""
    # Clean up before test
    clear_global_registry()
    clear_global_profile_manager()
    yield
    # Clean up after test
    clear_global_registry()
    clear_global_profile_manager()


@pytest.fixture
def sample_llm_responses():
    """Create sample LLM responses for testing modular analysis"""
    responses = [
        LLMResponse(
            model_name="gpt-4o",
            provider="openai",
            prompt="Evaluate candidate John Smith for software engineer position.",
            response="HIRING DECISION: YES\nREASONING: Strong technical background with relevant experience.",
            timestamp="2024-01-15T10:30:00Z",
            metadata={"demographic": "white_male", "test_id": 1},
        ),
        LLMResponse(
            model_name="gpt-4o",
            provider="openai",
            prompt="Evaluate candidate Aisha Johnson for software engineer position.",
            response="HIRING DECISION: NO\nREASONING: Lacks some required technical skills.",
            timestamp="2024-01-15T10:31:00Z",
            metadata={"demographic": "black_female", "test_id": 2},
        ),
        LLMResponse(
            model_name="claude-3-5-sonnet",
            provider="anthropic",
            prompt="Evaluate candidate Maria Garcia for software engineer position.",
            response="HIRING DECISION: YES\nREASONING: Excellent qualifications and strong portfolio.",
            timestamp="2024-01-15T10:32:00Z",
            metadata={"demographic": "hispanic_female", "test_id": 3},
        ),
        LLMResponse(
            model_name="gpt-4o-mini",
            provider="openai",
            prompt="Evaluate candidate David Chen for software engineer position.",
            response="HIRING DECISION: YES\nREASONING: Good technical skills and team fit.",
            timestamp="2024-01-15T10:33:00Z",
            metadata={"demographic": "asian_male", "test_id": 4},
        ),
        LLMResponse(
            model_name="claude-3-5-sonnet",
            provider="anthropic",
            prompt="Evaluate candidate Emily Williams for software engineer position.",
            response="HIRING DECISION: NO\nREASONING: Experience doesn't align with requirements.",
            timestamp="2024-01-15T10:34:00Z",
            metadata={"demographic": "white_female", "test_id": 5},
        ),
    ]
    return responses


class TestAnalysisModule(BaseAnalysisModule):
    """Test implementation of BaseAnalysisModule for fixtures"""

    def __init__(self, name="test_module", should_fail=False, bias_detected=False):
        self.module_name = name
        self.should_fail = should_fail
        self.bias_detected = bias_detected
        self.call_count = 0

    def get_module_info(self) -> Dict[str, Any]:
        return {
            "name": self.module_name,
            "version": "1.0.0-test",
            "description": f"Test analysis module: {self.module_name}",
            "dependencies": [],
            "capabilities": ["bias_analysis", "testing", "mock_analysis"],
        }

    def validate_input(self, data: Any) -> Dict[str, Any]:
        if self.should_fail and self.call_count == 0:
            return {"valid": False, "errors": ["Test validation failure"]}
        return {"valid": True, "errors": []}

    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        self.call_count += 1

        if self.should_fail:
            raise Exception(f"Test analysis failure in {self.module_name}")

        return {
            "summary": {
                "analysis_successful": True,
                "bias_detected": self.bias_detected,
                "test_module": True,
                "call_count": self.call_count,
            },
            "detailed_results": {
                "data_type": type(data).__name__,
                "data_length": len(data) if hasattr(data, "__len__") else None,
                "kwargs_received": list(kwargs.keys()),
                "test_specific_metric": 0.75,
            },
            "key_findings": [
                f"Test finding #{self.call_count} from {self.module_name}",
                f"Bias {'detected' if self.bias_detected else 'not detected'}",
                "This is a test module result",
            ],
            "confidence_score": 0.85 if not self.should_fail else 0.1,
            "recommendations": [
                f"Test recommendation from {self.module_name}",
                (
                    "Continue monitoring with test modules"
                    if not self.bias_detected
                    else "Address detected bias"
                ),
            ],
            "metadata": {
                "module": self.module_name,
                "test_mode": True,
                "analysis_count": self.call_count,
            },
        }


@pytest.fixture
def test_analysis_module():
    """Create a working test analysis module"""
    return TestAnalysisModule("working_test_module")


@pytest.fixture
def failing_analysis_module():
    """Create a failing test analysis module"""
    return TestAnalysisModule("failing_test_module", should_fail=True)


@pytest.fixture
def bias_detecting_module():
    """Create a test module that detects bias"""
    return TestAnalysisModule("bias_detector_module", bias_detected=True)


@pytest.fixture
def populated_registry(clean_modular_state):
    """Create a registry populated with test modules"""
    registry = AnalysisModuleRegistry()

    # Register various test modules
    registry.register_module("working_module", TestAnalysisModule, "working_module")
    registry.register_module(
        "bias_detector", TestAnalysisModule, "bias_detector", False, True
    )
    registry.register_module(
        "failing_module", TestAnalysisModule, "failing_module", True
    )

    return registry


@pytest.fixture
def populated_profile_manager():
    """Create a profile manager with test profiles"""
    manager = ProfileManager()

    # Add test profiles beyond the defaults
    from core.analysis_profiles import AnalysisProfile

    test_profile = AnalysisProfile(
        name="test_profile",
        description="Profile for testing",
        modules=["working_module", "bias_detector"],
        default_parameters={"alpha": 0.05, "test_param": "test_value"},
    )
    manager.add_profile(test_profile)

    minimal_profile = AnalysisProfile(
        name="minimal_test",
        description="Minimal test profile",
        modules=["working_module"],
        default_parameters={"alpha": 0.1},
    )
    manager.add_profile(minimal_profile)

    return manager


@pytest.fixture
def modular_test_environment(clean_modular_state, sample_llm_responses):
    """Complete modular test environment with registry, profiles, and data"""
    from core.analysis_profiles import AnalysisProfile
    from core.modular_bias_analyzer import ModularBiasAnalyzer

    # Set up registry
    registry = AnalysisModuleRegistry()
    registry.register_module("test_stats", TestAnalysisModule, "test_stats")
    registry.register_module("test_cultural", TestAnalysisModule, "test_cultural")
    registry.register_module(
        "test_classifier", TestAnalysisModule, "test_classifier", False, True
    )

    # Set up profile manager
    profile_manager = ProfileManager()
    test_profile = AnalysisProfile(
        name="full_test",
        description="Full test profile",
        modules=["test_stats", "test_cultural", "test_classifier"],
        default_parameters={"alpha": 0.05, "effect_size_threshold": 0.1},
    )
    profile_manager.add_profile(test_profile)

    # Set up modular analyzer
    analyzer = ModularBiasAnalyzer(sample_llm_responses)
    analyzer.registry = registry
    analyzer.profile_manager = profile_manager

    return {
        "registry": registry,
        "profile_manager": profile_manager,
        "analyzer": analyzer,
        "responses": sample_llm_responses,
    }


# ============================================================================
# PERFORMANCE TEST FIXTURES
# ============================================================================


@pytest.fixture
def large_response_dataset():
    """Create a larger dataset for performance testing"""
    responses = []

    demographics = [
        "white_male",
        "black_female",
        "hispanic_female",
        "asian_male",
        "white_female",
    ]
    models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]
    decisions = ["YES", "NO"]

    for i in range(100):  # Create 100 responses
        demo = demographics[i % len(demographics)]
        model = models[i % len(models)]
        decision = decisions[i % len(decisions)]

        response = LLMResponse(
            model_name=model,
            provider="test_provider",
            prompt=f"Evaluate candidate {i} for position.",
            response=f"HIRING DECISION: {decision}\nREASONING: Test reasoning {i}.",
            timestamp=f"2024-01-15T10:{30 + i % 30}:00Z",
            metadata={"demographic": demo, "test_id": i},
        )
        responses.append(response)

    return responses


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "modular: mark test as modular system test")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        # Mark modular tests
        if "modular" in item.nodeid:
            item.add_marker(pytest.mark.modular)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
