"""
Pytest configuration and fixtures for OpenAudit tests
"""

import pytest

from core.bias_testing_framework import HiringBiasTest
from core.cv_templates import CVTemplates


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
