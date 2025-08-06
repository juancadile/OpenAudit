# OpenAudit Test Suite

This directory contains comprehensive tests for the OpenAudit bias detection framework, including the new modular analysis system.

## Test Structure

### Core System Tests
- `test_enhanced_features.py` - Tests for enhanced statistical analysis features
- `test_error_handling.py` - Error handling and edge case tests
- `test_cv_consistency.py` - CV template consistency tests
- `test_ceteris_paribus.py` - Ceteris paribus analysis tests

### Modular System Tests ⭐ **NEW**
- `test_modular_integration.py` - Comprehensive modular system integration tests
- `test_module_registry.py` - Focused tests for module registration and discovery
- `test_analysis_profiles.py` - Tests for analysis profile system

### Configuration
- `conftest.py` - Shared test fixtures and configuration with modular system support

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test categories:
```bash
# Enhanced features only
pytest tests/test_enhanced_features.py

# Modular system tests
pytest tests/test_modular_integration.py
pytest tests/test_module_registry.py
pytest tests/test_analysis_profiles.py

# Error handling only
pytest tests/test_error_handling.py
```

### Run tests by marker:
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Modular system tests only
pytest -m modular

# Performance tests only
pytest -m performance
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run with coverage:
```bash
pytest tests/ --cov=core --cov-report=html
```

### Run specific test patterns:
```bash
# All registry tests
pytest -k "registry"

# All profile tests
pytest -k "profile"

# All modular tests
pytest -k "modular"
```

## Test Coverage

### Legacy System Coverage
- Statistical analysis methods
- Cultural bias detection
- Multi-level classification
- Error handling and edge cases
- CV template generation and consistency
- Ceteris paribus analysis

### Modular System Coverage ⭐ **NEW**
- **Module Registry System**
  - Module registration and discovery
  - Module compatibility validation
  - Analysis pipeline creation
  - Error handling and graceful degradation

- **Analysis Profiles**
  - Profile creation and management
  - Parameter merging and inheritance
  - Default profile functionality
  - Profile validation

- **Modular Analysis**
  - End-to-end analysis workflows
  - Module integration and communication
  - Error recovery and partial failures
  - Performance impact assessment

- **Integration Testing**
  - API endpoint simulation
  - Web interface compatibility
  - Backwards compatibility with legacy system
  - Complete workflow testing

## Test Fixtures

### Standard Fixtures
- `bias_test` - HiringBiasTest instance
- `sample_variables` - Standard CV variables
- `cv_templates` - CV template system

### Modular System Fixtures ⭐ **NEW**
- `clean_modular_state` - Clean registry/profile state for each test
- `sample_llm_responses` - Sample responses for modular analysis
- `test_analysis_module` - Working test module
- `failing_analysis_module` - Module that fails for error testing
- `bias_detecting_module` - Module that detects bias
- `populated_registry` - Registry with test modules pre-registered
- `populated_profile_manager` - Profile manager with test profiles
- `modular_test_environment` - Complete test environment setup
- `large_response_dataset` - Large dataset for performance testing

## Test Data

Tests use sample LLM responses and mock data to validate functionality without requiring live API calls. The modular system tests include:

- **Mock Modules** - Test implementations of `BaseAnalysisModule`
- **Sample Responses** - Realistic LLM responses with diverse demographics
- **Test Profiles** - Preconfigured analysis profiles for testing
- **Large Datasets** - Performance testing with 100+ responses

## Writing New Tests

### For Module Development
When creating new analysis modules, create corresponding tests:

```python
# tests/test_my_new_module.py
from core.my_new_module import MyNewModule
from tests.conftest import TestAnalysisModule

def test_my_module_functionality():
    module = MyNewModule()
    # Test specific functionality

def test_my_module_integration(modular_test_environment):
    env = modular_test_environment
    # Test integration with modular system
```

### For Profile Development
When creating new analysis profiles:

```python
def test_my_profile():
    from core.analysis_profiles import AnalysisProfile

    profile = AnalysisProfile(
        name="my_profile",
        description="My custom profile",
        modules=["module1", "module2"]
    )

    # Test profile functionality
```

## Legacy Test Scripts

For backwards compatibility, legacy test scripts can still be run directly:

```bash
# Framework validation tests
python3 tests/test_ceteris_paribus.py
python3 tests/test_cv_consistency.py
```

## Academic Compliance

The tests validate compliance with established audit methodologies:
- **Ceteris Paribus Principle** - Only demographic information varies between test cases
- **Template Consistency** - Qualifications and experience remain constant
- **Statistical Validity** - Sufficient variation for meaningful analysis
- **Modular Integrity** - Module interactions maintain analytical rigor

## Continuous Integration

The test suite is designed to run in CI environments with:
- Automatic test discovery
- Proper fixture cleanup
- Performance benchmarking
- Coverage reporting
- Test categorization with markers

## Debugging Tests

For debugging failed tests:

```bash
# Run with debugging output
pytest tests/test_modular_integration.py -v -s

# Run specific test with pdb
pytest tests/test_modular_integration.py::TestModularBiasAnalyzer::test_modular_analysis_execution -v -s --pdb

# Show test fixtures
pytest --fixtures tests/
```
