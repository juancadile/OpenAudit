# Core Framework

This folder contains the core OpenAudit framework components that power bias detection and analysis.

## Components

### Core Analysis Engine
- **`bias_testing_framework.py`** - Main bias testing framework with statistical analysis
- **`response_analyzer.py`** - Analyzes LLM responses and categorizes by demographics
- **`reasoning_analyzer.py`** - Deep analysis of model reasoning patterns to understand bias mechanisms

### Model & Data Management  
- **`multi_llm_dispatcher.py`** - Handles multiple LLM provider integrations (OpenAI, etc.)
- **`cv_templates.py`** - Generates realistic CVs with controlled demographic variations

## Usage

These modules are imported by the main interface applications:

```python
from core.bias_testing_framework import HiringBiasTest, BiasDatasets
from core.response_analyzer import analyze_responses_by_demographics
from core.multi_llm_dispatcher import MultiLLMDispatcher
from core.cv_templates import CVTemplates
from core.reasoning_analyzer import analyze_reasoning_patterns
```

## Academic Foundation

The framework implements methodologies from:
- Metaxa et al. (2021) - Audit methodology for AI bias
- Bertrand & Mullainathan (2004) - Resume audit studies
- Marks & Karvonen (2025) - LLM fairness in realistic settings 