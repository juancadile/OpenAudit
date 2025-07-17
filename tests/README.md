# Tests

This folder contains test scripts for validating OpenAudit functionality and research methodologies.

## Test Scripts

### Framework Validation Tests
- **`test_ceteris_paribus.py`** - Tests the "all else equal" principle in CV generation
  ```bash
  python3 tests/test_ceteris_paribus.py
  ```

- **`test_cv_consistency.py`** - Validates CV template consistency across demographic variations
  ```bash
  python3 tests/test_cv_consistency.py
  ```

## Purpose

These tests ensure:

1. **Methodological Rigor** - CVs differ only by demographic signals, not qualifications
2. **Data Quality** - Generated data meets academic standards for bias testing
3. **Framework Reliability** - Core components function correctly across different scenarios

## Academic Compliance

The tests validate compliance with established audit methodologies:
- **Ceteris Paribus Principle** - Only demographic information varies between test cases
- **Template Consistency** - Qualifications and experience remain constant
- **Statistical Validity** - Sufficient variation for meaningful analysis

## Usage

Run all tests to validate the framework:

```bash
cd tests/
python3 test_ceteris_paribus.py
python3 test_cv_consistency.py
```

These tests should pass before conducting bias experiments to ensure methodological soundness. 