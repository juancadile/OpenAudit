# Development Guide

This guide covers the development workflow and tools for OpenAudit.

## Quick Start

```bash
# 1. Set up development environment
make setup

# 2. Configure API keys
cp .env.example .env
# Edit .env with your actual API keys

# 3. Run tests to verify setup
make test

# 4. Start the unified dashboard
make run-unified
```

## Development Environment

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setup

1. **Create virtual environment:**
   ```bash
   python -m venv openaudit_env
   source openaudit_env/bin/activate  # Linux/Mac
   # or
   openaudit_env\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   make install-dev
   ```

3. **Set up pre-commit hooks:**
   ```bash
   make pre-commit-install
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Development Workflow

### Code Quality

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security analysis
- **pytest**: Testing

### Common Commands

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security check
make security

# Run tests
make test

# Run tests with coverage
make test-verbose

# Complete development workflow
make dev

# Simulate CI pipeline
make ci
```

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit:

```bash
# Install hooks (run once)
make pre-commit-install

# Run hooks manually on all files
make pre-commit-run
```

## Testing

### Test Structure

```
tests/
├── conftest.py           # Pytest fixtures and configuration
├── test_cv_consistency.py # CV generation consistency tests
└── test_ceteris_paribus.py # Bias detection validity tests
```

### Running Tests

```bash
# Basic test run
make test

# Verbose with coverage
make test-verbose

# Fast tests (no coverage)
make test-fast

# Watch mode (requires pytest-watch)
make test-watch
```

### Writing Tests

- Use pytest fixtures from `conftest.py`
- Test functions should start with `test_`
- Use descriptive test names
- Include docstrings explaining what is being tested
- Use parametrized tests for multiple scenarios

Example:
```python
def test_cv_generation_consistency(cv_templates, sample_variables):
    """Test that CV generation is deterministic."""
    cv1 = cv_templates.generate_cv_content('software_engineer', sample_variables)
    cv2 = cv_templates.generate_cv_content('software_engineer', sample_variables)
    assert cv1 == cv2, "CV generation should be deterministic"
```

## Code Style

### Formatting Rules

- Line length: 88 characters (Black default)
- Use double quotes for strings
- Imports sorted by isort with Black profile
- Type hints encouraged but not strictly required

### Import Order

1. Standard library imports
2. Related third-party imports
3. Local application/library imports

### Docstrings

Use Google-style docstrings:

```python
def analyze_bias(results: List[Dict]) -> BiasAnalysis:
    """Analyze bias patterns in experiment results.

    Args:
        results: List of experiment results with demographic data.

    Returns:
        BiasAnalysis object containing detected bias patterns.

    Raises:
        ValueError: If results list is empty.
    """
```

## Application Architecture

### Core Components

- `core/bias_testing_framework.py`: Main bias testing logic
- `core/multi_llm_dispatcher.py`: LLM provider integrations
- `core/cv_templates.py`: CV/resume generation
- `core/response_analyzer.py`: Response analysis
- `core/logging_config.py`: Centralized logging configuration
- `core/model_manager.py`: AI model management
- `core/template_manager.py`: Template management system
- `unified_interface.py`: Main web application

### Logging System

OpenAudit uses a comprehensive logging system with multiple outputs:

- **Console logs**: Development-friendly formatted output
- **File logs**: Detailed logs in `logs/openaudit.log`
- **Error logs**: Dedicated error tracking in `logs/errors.log`
- **Experiment logs**: JSON-formatted experiment events in `logs/experiments.log`

#### Using Logging in Code

```python
from core.logging_config import get_logger, log_experiment_event

# Get a logger for your module
logger = get_logger(__name__)

# Log at different levels
logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")

# Log experiment events with structured data
log_experiment_event(
    event_type="model_response",
    experiment_id="exp_123",
    model_name="gpt-4",
    demographic="white_male",
    decision="YES"
)
```

#### Log Level Configuration

Set log level via environment variable:
```bash
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Testing Philosophy

OpenAudit implements rigorous testing for bias detection validity:

1. **Ceteris Paribus**: Only names vary between demographic groups
2. **Deterministic Generation**: Same inputs produce identical outputs
3. **Statistical Validity**: Proper statistical analysis of results

## Continuous Integration

GitHub Actions automatically run:

1. **Tests**: Across Python 3.8-3.11
2. **Code Quality**: Linting, formatting, type checking
3. **Security**: Vulnerability scanning
4. **Coverage**: Code coverage reporting

### Local CI Simulation

```bash
# Run the same checks as CI
make ci
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'core'**
- Ensure you're in the project root directory
- Activate your virtual environment

**Pre-commit hooks failing**
- Run `make format` to fix formatting issues
- Run `make lint` to see specific linting errors

**Tests failing due to API keys**
- Tests should use mock API keys (see GitHub Actions)
- Real API calls should be mocked in tests

**Coverage too low**
- Add tests for uncovered code paths
- Use `make test-verbose` to see coverage report

### Getting Help

1. Check this development guide
2. Review the main README.md
3. Look at existing tests for examples
4. Check GitHub Issues for known problems

## Contributing

### Before Submitting PRs

1. Run `make ci` to ensure all checks pass
2. Add tests for new functionality
3. Update documentation if needed
4. Ensure commit messages are descriptive

### Code Review Checklist

- [ ] Tests pass locally and in CI
- [ ] Code follows style guidelines
- [ ] New functionality has tests
- [ ] Documentation updated if needed
- [ ] No security vulnerabilities introduced
- [ ] Ceteris paribus maintained for bias testing

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md (if exists)
3. Run full test suite
4. Create tagged release
5. Update documentation as needed
