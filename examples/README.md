# OpenAudit External Modules

This directory contains examples and documentation for creating custom analysis modules for the OpenAudit modular system.

## Overview

The OpenAudit external module system allows you to create custom bias analysis modules that integrate seamlessly with the existing framework. These modules can implement specialized analysis techniques, domain-specific bias detection, or novel statistical approaches.

## Quick Start

### 1. Create a Module Template

```bash
# Create a new module template
python cli.py modules create my_custom_analyzer \
  --author "Your Name" \
  --description "My custom bias analysis module"
```

This creates:
- `external_modules/my_custom_analyzer.py` - Module implementation
- `external_modules/my_custom_analyzer.manifest.json` - Module metadata

### 2. Implement Your Analysis Logic

Edit the generated Python file to implement your custom analysis:

```python
def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    # Your custom bias analysis logic here
    return {
        "summary": {"bias_detected": False, ...},
        "detailed_results": {...},
        "key_findings": [...],
        "confidence_score": 0.8,
        "recommendations": [...],
        "metadata": {...}
    }
```

### 3. Register and Test

```bash
# Register your module
python cli.py modules register my_custom_analyzer external_modules/my_custom_analyzer.py

# Test it
python cli.py modules test my_custom_analyzer

# Use it in a profile
python cli.py profiles create my_profile --modules enhanced_statistics my_custom_analyzer
```

## Module Development Guide

### Required Interface

All external modules must inherit from `BaseAnalysisModule` and implement these methods:

```python
from core.base_analyzer import BaseAnalysisModule

class MyModule(BaseAnalysisModule):
    def get_module_info(self) -> Dict[str, Any]:
        """Return module metadata"""
        
    def validate_input(self, data: Any) -> Dict[str, Any]:
        """Validate input data format and requirements"""
        
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Perform the actual bias analysis"""
```

### Standard Return Format

The `analyze()` method must return a dictionary with this structure:

```python
{
    "summary": {
        "analysis_successful": bool,
        "bias_detected": bool,
        "total_responses_analyzed": int,
        # Add custom summary metrics
    },
    "detailed_results": {
        "parameters_used": dict,
        # Add detailed analysis results
    },
    "key_findings": [
        "List of key findings as strings"
    ],
    "confidence_score": float,  # 0.0 to 1.0
    "recommendations": [
        "List of actionable recommendations"
    ],
    "metadata": {
        "module": "module_name",
        "version": "1.0.0",
        "analysis_timestamp": "ISO format timestamp"
    }
}
```

### Input Data Format

Modules typically receive a pandas DataFrame with these columns:

- **`model`** - LLM model name (e.g., "gpt-4o")
- **`response`** - LLM response text
- **`demographic`** - Demographic group identifier
- **`prompt`** - Original prompt sent to the model
- **`timestamp`** - Response timestamp
- **Additional columns** may be present depending on the experiment

### Security Considerations

External modules are subject to security validation:

#### ✅ Allowed Imports
- Standard scientific libraries: `numpy`, `pandas`, `scipy`, `sklearn`
- Visualization: `matplotlib`, `seaborn`
- Standard library: `typing`, `datetime`, `json`, `logging`, `collections`
- OpenAudit core: `core.*`

#### ❌ Restricted Operations
- File system access (`open`, `os.remove`, etc.)
- Network operations (`requests`, `urllib`, etc.)
- Code execution (`eval`, `exec`, `subprocess`)
- System operations (`os.system`, `subprocess`)

#### Bypass Security (Use with Caution)
```bash
python cli.py modules register my_module path/to/module.py --no-security-check
```

## Example Modules

### 1. Custom Bias Detector (`custom_bias_detector.py`)

This example demonstrates:
- Language pattern analysis
- Demographic group comparison
- Custom bias scoring
- Statistical confidence calculation

**Features:**
- Detects qualification doubt patterns
- Identifies positive/negative assumptions
- Analyzes cultural stereotypes
- Provides actionable recommendations

**Usage:**
```bash
python cli.py modules register custom_bias_detector examples/custom_bias_detector.py
python cli.py profiles create language_analysis --modules enhanced_statistics custom_bias_detector
```

### 2. Creating Your Own Module

Here's a minimal template for a custom module:

```python
from typing import Dict, Any
import pandas as pd
from core.base_analyzer import BaseAnalysisModule

class MyAnalysisModule(BaseAnalysisModule):
    def get_module_info(self) -> Dict[str, Any]:
        return {
            "name": "my_analysis",
            "version": "1.0.0",
            "description": "My custom analysis",
            "dependencies": ["pandas"],
            "capabilities": ["bias_analysis"]
        }
    
    def validate_input(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, pd.DataFrame):
            return {"valid": False, "errors": ["Expected pandas DataFrame"]}
        return {"valid": True, "errors": []}
    
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        # Your analysis logic here
        bias_detected = False  # Replace with actual analysis
        
        return {
            "summary": {
                "analysis_successful": True,
                "bias_detected": bias_detected,
                "total_responses_analyzed": len(data)
            },
            "detailed_results": {
                "parameters_used": kwargs,
                "custom_metric": 0.5
            },
            "key_findings": ["Custom analysis completed"],
            "confidence_score": 0.8,
            "recommendations": ["Continue monitoring"],
            "metadata": {
                "module": "my_analysis",
                "version": "1.0.0"
            }
        }
```

## Advanced Features

### 1. Module Dependencies

Specify dependencies in your module info:

```python
def get_module_info(self) -> Dict[str, Any]:
    return {
        "dependencies": ["pandas", "numpy", "scipy"],
        # ... other info
    }
```

### 2. Parameter Validation

Validate analysis parameters:

```python
def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    # Validate parameters
    alpha = kwargs.get('alpha', 0.05)
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Continue with analysis...
```

### 3. Progress Logging

Use logging for debugging and progress tracking:

```python
import logging
logger = logging.getLogger(__name__)

def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    logger.info(f"Starting analysis with {len(data)} responses")
    # ... analysis code ...
    logger.info("Analysis completed successfully")
```

### 4. Error Handling

Implement robust error handling:

```python
def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    try:
        # Analysis logic
        return results
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "summary": {"analysis_successful": False, "error": str(e)},
            # ... minimal result structure
        }
```

## Module Lifecycle

### Development Workflow

1. **Create Template**
   ```bash
   python cli.py modules create my_module
   ```

2. **Implement Logic**
   Edit the generated Python file

3. **Test Locally**
   ```bash
   python cli.py modules register my_module external_modules/my_module.py
   python cli.py modules test my_module
   ```

4. **Integration Testing**
   ```bash
   python cli.py profiles create test_profile --modules my_module enhanced_statistics
   python cli.py test modular --verbose
   ```

5. **Production Use**
   ```bash
   python cli.py profiles create production_profile --modules my_module enhanced_statistics cultural_context
   ```

### Module Management

```bash
# List all modules (built-in + external)
python cli.py modules list

# List only external modules
python cli.py modules list-external

# Get detailed module info
python cli.py modules info my_module

# Test compatibility
python cli.py modules test my_module enhanced_statistics

# Load all modules from directory
python cli.py modules load-all
```

### Version Management

Update your module's version in `get_module_info()`:

```python
def get_module_info(self) -> Dict[str, Any]:
    return {
        "version": "1.1.0",  # Increment for changes
        # ... other info
    }
```

## Best Practices

### 1. Code Quality
- Use type hints for all function parameters and returns
- Add comprehensive docstrings
- Follow Python naming conventions
- Include unit tests when possible

### 2. Data Validation
- Always validate input data format
- Check for required columns
- Verify minimum sample sizes
- Handle missing or malformed data gracefully

### 3. Statistical Rigor
- Use appropriate statistical tests
- Report confidence intervals
- Validate assumptions (normality, independence, etc.)
- Account for multiple comparisons when applicable

### 4. Performance
- Optimize for large datasets (1000+ responses)
- Use vectorized operations with pandas/numpy
- Avoid nested loops when possible
- Consider memory usage for large analyses

### 5. Reproducibility
- Use random seeds when applicable
- Document parameter effects
- Provide clear methodology descriptions
- Version control your modules

### 6. Integration
- Follow the standard return format exactly
- Use consistent terminology with other modules
- Make recommendations actionable
- Provide confidence scores based on data quality

## Troubleshooting

### Common Issues

**Security Validation Fails**
```bash
# Check specific security issues
python cli.py modules register my_module path/to/module.py
# Review error messages and remove restricted operations
```

**Module Not Loading**
```bash
# Check Python syntax
python -m py_compile path/to/module.py

# Check imports
python -c "import sys; sys.path.append('.'); import my_module"
```

**Interface Validation Fails**
```bash
# Ensure all required methods are implemented
python cli.py modules info my_module
```

**Integration Issues**
```bash
# Test with minimal profile
python cli.py profiles create minimal_test --modules my_module
python cli.py modules test my_module
```

### Debug Mode

Run with debug logging:
```bash
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from cli import OpenAuditCLI
cli = OpenAuditCLI()
cli.run(['modules', 'register', 'my_module', 'path/to/module.py'])
"
```

## Contributing

### Sharing Modules

To share your module with the community:

1. Ensure it follows all best practices
2. Include comprehensive documentation
3. Add example usage and test cases
4. Submit as a pull request to the OpenAudit repository

### Module Repository

We maintain a repository of community-contributed modules:
- Specialized domain analyses (legal, medical, financial)
- Novel statistical approaches
- Integration with external tools
- Visualization modules

## Support

For help with module development:

1. Check this documentation and examples
2. Review the built-in module implementations in `core/`
3. Run `python cli.py --help` for CLI guidance
4. Open an issue on the OpenAudit GitHub repository

## API Reference

### BaseAnalysisModule

Abstract base class for all analysis modules.

**Required Methods:**
- `get_module_info() -> Dict[str, Any]`
- `validate_input(data: Any) -> Dict[str, Any]`
- `analyze(data: Any, **kwargs) -> Dict[str, Any]`

**Module Info Structure:**
```python
{
    "name": str,
    "version": str,
    "description": str,
    "author": str,              # Optional
    "dependencies": List[str],  # Optional
    "capabilities": List[str]   # Optional
}
```

**Validation Result Structure:**
```python
{
    "valid": bool,
    "errors": List[str]
}
```

### External Module Loader

**Functions:**
- `load_external_module(path, name, **kwargs)`
- `create_module_template(name, **kwargs)`
- `list_external_modules()`

**CLI Commands:**
- `modules create <name>` - Create template
- `modules register <name> <path>` - Register module
- `modules list-external` - List loaded modules
- `modules load-all` - Load all from directory 