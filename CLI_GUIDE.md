# OpenAudit CLI Guide

The OpenAudit Command Line Interface provides comprehensive access to the modular bias testing platform.

## Quick Start

```bash
# Start the web interface (same as before)
python cli.py start

# Get help
python cli.py --help

# Check system status
python cli.py status
```

## Commands Overview

### 🚀 Application Management

```bash
# Start web interface
python cli.py start                    # Default port 5100
python cli.py start --port 8080        # Custom port
python cli.py start --debug            # Debug mode

# System status
python cli.py status                   # Show system information
python cli.py version                  # Show version
```

### 🧩 Module Management

```bash
# List all available modules
python cli.py modules list

# Get detailed module information
python cli.py modules info enhanced_statistics

# Test module compatibility
python cli.py modules test enhanced_statistics cultural_context

# Register a new module (advanced)
python cli.py modules register my_module path/to/module.py
```

### 📝 Profile Management

```bash
# List available analysis profiles
python cli.py profiles list

# Show profile details
python cli.py profiles show comprehensive

# Create a custom profile
python cli.py profiles create my_analysis \
  --description "My custom analysis setup" \
  --modules enhanced_statistics cultural_context \
  --params '{"alpha": 0.01, "threshold": 0.2}'

# Validate a profile
python cli.py profiles validate my_analysis
```

### 📊 Analysis Commands

```bash
# Run analysis with a profile
python cli.py analyze run_123 --profile comprehensive

# Run with specific modules
python cli.py analyze run_123 --modules enhanced_statistics cultural_context

# Custom analysis parameters
python cli.py analyze run_123 \
  --profile standard \
  --alpha 0.01 \
  --effect-size-threshold 0.2 \
  --output results.json \
  --format json
```

### 🧪 Testing

```bash
# Run quick smoke tests
python cli.py test quick

# Run all modular system tests
python cli.py test modular

# Run full test suite
python cli.py test all

# Verbose test output
python cli.py test modular --verbose
```

## Module System

### Available Modules

The modular system includes these built-in analysis modules:

- **`enhanced_statistics`** - Advanced statistical analysis with effect sizes
- **`cultural_context`** - Cultural bias detection and context analysis  
- **`multi_level_classifier`** - Multi-level bias classification
- **`goal_conflict_analyzer`** - Goal conflict detection
- **`human_ai_alignment`** - Human-AI alignment analysis

### Module Information

Get detailed information about any module:

```bash
python cli.py modules info enhanced_statistics
```

Output includes:
- Module description and version
- Dependencies and capabilities
- Compatibility status

### Module Compatibility

Test if modules work together:

```bash
python cli.py modules test enhanced_statistics cultural_context multi_level_classifier
```

This checks:
- Individual module compatibility
- Group compatibility (modules working together)
- Dependency resolution

## Analysis Profiles

### Built-in Profiles

OpenAudit comes with several pre-configured profiles:

- **`quick`** - Fast analysis with essential modules
- **`standard`** - Balanced analysis with core modules
- **`comprehensive`** - Complete analysis with all modules
- **`statistical_focus`** - Statistics-heavy analysis
- **`cultural_focus`** - Cultural bias focused analysis

### Creating Custom Profiles

Create profiles tailored to your needs:

```bash
# Basic profile
python cli.py profiles create hiring_focus \
  --description "Focused on hiring bias detection" \
  --modules enhanced_statistics goal_conflict_analyzer

# Advanced profile with parameters
python cli.py profiles create research_grade \
  --description "Research-grade comprehensive analysis" \
  --modules enhanced_statistics cultural_context multi_level_classifier human_ai_alignment \
  --params '{"alpha": 0.01, "effect_size_threshold": 0.1, "cultural_sensitivity": 0.8}'
```

### Profile Parameters

Common profile parameters:

- `alpha` - Significance level (default: 0.05)
- `effect_size_threshold` - Minimum effect size for practical significance (default: 0.1)
- `confidence_level` - Confidence level for intervals (default: 0.95)
- `cultural_sensitivity` - Cultural analysis sensitivity (default: 0.5)
- `bias_threshold` - Bias detection threshold (default: 0.3)

## Analysis Workflow

### 1. Check System Status

```bash
python cli.py status
```

Verify modules and profiles are available.

### 2. Choose Analysis Approach

Option A - Use existing profile:
```bash
python cli.py profiles list
python cli.py profiles show comprehensive
```

Option B - Create custom profile:
```bash
python cli.py profiles create my_analysis --modules enhanced_statistics cultural_context
```

### 3. Run Analysis

```bash
# With profile
python cli.py analyze my_run_data --profile comprehensive --output results.json

# Custom parameters
python cli.py analyze my_run_data \
  --modules enhanced_statistics cultural_context \
  --alpha 0.01 \
  --format json \
  --output detailed_results.json
```

### 4. Review Results

Results include:
- Module-specific analysis results
- Unified assessment and recommendations
- Statistical significance tests
- Effect size calculations
- Cultural context analysis
- Bias classification and severity

## Advanced Usage

### Batch Analysis

For multiple runs, create a script:

```bash
#!/bin/bash
for run in run_001 run_002 run_003; do
    python cli.py analyze $run --profile comprehensive --output ${run}_results.json
done
```

### Development Workflow

When developing new modules:

```bash
# Test module registration
python cli.py modules register test_module path/to/test_module.py

# Test compatibility
python cli.py modules test test_module enhanced_statistics

# Create test profile
python cli.py profiles create test_profile --modules test_module enhanced_statistics

# Validate
python cli.py profiles validate test_profile

# Run tests
python cli.py test modular --verbose
```

### Debugging

For troubleshooting:

```bash
# Check system status
python cli.py status

# Validate specific profile
python cli.py profiles validate problematic_profile

# Test module compatibility
python cli.py modules test module1 module2

# Run in debug mode
python cli.py start --debug

# Verbose test output
python cli.py test quick --verbose
```

## Integration with Web Interface

The CLI complements the web interface:

1. **CLI**: Module management, profile creation, system administration
2. **Web UI**: Interactive analysis, visualization, experiment management

Start the web interface:
```bash
python cli.py start
```

Then access http://localhost:5100 for the full dashboard.

## Migration from Legacy

If you were using the old `run_openaudit.py` script:

**Old way:**
```bash
python run_openaudit.py
```

**New way (equivalent):**
```bash
python cli.py start
```

**New functionality:**
```bash
# All the module and profile management commands above
python cli.py modules list
python cli.py profiles create my_profile --modules enhanced_statistics
```

## Getting Help

```bash
# General help
python cli.py --help

# Command-specific help
python cli.py modules --help
python cli.py profiles --help
python cli.py analyze --help

# System status and diagnostics
python cli.py status
python cli.py test quick
```

## Best Practices

1. **Start with built-in profiles** before creating custom ones
2. **Test module compatibility** before creating profiles
3. **Use descriptive profile names** and descriptions
4. **Validate profiles** after creation
5. **Run quick tests** to verify system health
6. **Use version control** for custom profiles and configurations
7. **Document custom modules** and their capabilities

## Examples

### Research Workflow

```bash
# Setup
python cli.py status
python cli.py modules list

# Create research profile
python cli.py profiles create research_comprehensive \
  --description "Comprehensive research-grade analysis" \
  --modules enhanced_statistics cultural_context multi_level_classifier human_ai_alignment \
  --params '{"alpha": 0.01, "effect_size_threshold": 0.05}'

# Validate
python cli.py profiles validate research_comprehensive

# Run analysis
python cli.py analyze experiment_data \
  --profile research_comprehensive \
  --output research_results.json \
  --format json
```

### Production Monitoring

```bash
# Quick health check
python cli.py test quick

# System overview
python cli.py status

# Run standard analysis
python cli.py analyze production_data --profile standard --output monitoring_report.json
```

### Development Testing

```bash
# Test new module
python cli.py modules test my_new_module enhanced_statistics

# Create test profile
python cli.py profiles create dev_test --modules my_new_module enhanced_statistics

# Run modular tests
python cli.py test modular --verbose
``` 