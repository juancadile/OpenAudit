# 🚀 Getting Started with OpenAudit

Welcome to OpenAudit! This guide will get you up and running with the world's leading open-source platform for LLM bias auditing in under 10 minutes.

## ✨ **What is OpenAudit?**

OpenAudit is a modular, extensible platform for detecting and analyzing bias in Large Language Models (LLMs). It provides:

- 🔍 **Comprehensive Bias Detection** - Statistical, cultural, and alignment analysis
- 🧩 **Modular Architecture** - Mix and match analysis modules
- 🌐 **Web Interface** - User-friendly dashboard for non-technical users
- 🖥️ **CLI Tools** - Command-line interface for power users
- 📊 **Research-Grade Analysis** - Publication-ready statistical methods
- 🔌 **Extensible** - Create custom analysis modules

## 📋 **Prerequisites**

- **Python 3.8+** (Python 3.9+ recommended)
- **8GB RAM** minimum (16GB recommended for large experiments)
- **Internet connection** (for model API calls)

## 🛠️ **Installation**

### Option 1: Install from PyPI (Recommended)
```bash
pip install openaudit
```

### Option 2: Install from Source
```bash
git clone https://github.com/openaudit/openaudit.git
cd openaudit
pip install -e .
```

### Option 3: Development Installation
```bash
git clone https://github.com/openaudit/openaudit.git
cd openaudit
pip install -r requirements-dev.txt
pip install -e .
```

## ⚡ **Quick Start: Your First Bias Audit**

### 1. Set Up API Keys

OpenAudit supports multiple AI providers. Set up at least one:

```bash
# OpenAI (recommended for getting started)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Other providers
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

### 2. Run Your First Audit

**Using the Web Interface** (Easiest):
```bash
openaudit start
# Opens http://localhost:5000
```

**Using the CLI** (Quick):
```bash
# Run a basic hiring bias audit
openaudit audit hiring --model gpt-4 --profile basic
```

**Using Python** (Programmatic):
```python
from openaudit import BiasAuditor

# Create auditor
auditor = BiasAuditor()

# Run hiring bias audit
results = auditor.audit_hiring_bias(
    model="gpt-4",
    profile="basic",
    demographics=["race", "gender"],
    iterations=10
)

# View results
print(f"Bias detected: {results['bias_detected']}")
print(f"Bias gap: {results['bias_gap']:.2%}")
```

## 🌐 **Using the Web Interface**

The web interface is the easiest way to get started:

### 1. Start the Server
```bash
openaudit start
# Or: python unified_interface.py
```

### 2. Navigate to Dashboard
Open your browser to http://localhost:5000

### 3. Run Live Experiment
1. Go to the **"Live Experiments"** tab
2. Select your model (e.g., "gpt-4")
3. Choose demographics to test
4. Click **"Start Experiment"**
5. Watch real-time bias detection!

### 4. Analyze Results
1. Go to **"Historical Analysis"** tab
2. Select your experiment run
3. Choose analysis profile:
   - **Basic**: Quick statistical analysis
   - **Standard**: Enhanced stats + classification
   - **Research Grade**: All modules for comprehensive analysis
4. Click **"Analyze"**

## 🖥️ **Using the CLI**

The CLI provides powerful automation capabilities:

### Basic Commands
```bash
# List available models
openaudit models list

# List analysis modules
openaudit modules list

# List analysis profiles
openaudit profiles list

# Run specific analysis
openaudit analyze run_20250127_143052 --profile research_grade

# Create custom module template
openaudit modules create my_custom_analyzer
```

### Audit Commands
```bash
# Quick hiring bias audit
openaudit audit hiring --model gpt-4 --iterations 20

# Custom audit with specific demographics
openaudit audit hiring \
  --model gpt-4 \
  --demographics race,gender,university \
  --cv-level borderline \
  --profile comprehensive

# Batch audit multiple models
openaudit audit hiring --models gpt-4,claude-3,gemini-pro
```

### Analysis Commands
```bash
# Analyze with custom modules
openaudit analyze latest --modules enhanced_stats,cultural_context

# Generate comprehensive report
openaudit report latest --format html --include-raw

# Export results
openaudit export latest --format csv --output results.csv
```

## 🧩 **Understanding Analysis Modules**

OpenAudit's modular architecture lets you choose the right analysis for your needs:

### Core Modules
- **Enhanced Statistics** - Advanced statistical testing with effect sizes
- **Cultural Context** - Cultural and linguistic bias detection  
- **Multi-Level Classifier** - Nuanced bias classification
- **Goal Conflict** - Anthropic's goal misalignment detection
- **Human-AI Alignment** - Compare AI vs human assessments

### Analysis Profiles
Pre-configured combinations of modules:

| Profile | Modules | Use Case | Time |
|---------|---------|----------|------|
| **basic** | Enhanced Statistics | Quick bias check | ~1 min |
| **standard** | Stats + Classifier | Standard audit | ~3 min |
| **cultural** | Stats + Cultural + Classifier | Cross-cultural testing | ~5 min |
| **research_grade** | All modules | Academic research | ~10 min |
| **comprehensive** | All modules + detailed analysis | Maximum depth | ~15 min |

## 📊 **Example: Hiring Bias Audit**

Let's walk through a complete hiring bias audit:

### Step 1: Create the Audit
```python
from openaudit import HiringBiasAuditor

# Initialize auditor
auditor = HiringBiasAuditor()

# Configure experiment
config = {
    "models": ["gpt-4"],
    "demographics": {
        "race": ["white", "black", "hispanic", "asian"],
        "gender": ["male", "female"]
    },
    "cv_levels": ["borderline"],
    "iterations": 20,
    "role": "software_engineer"
}

# Run audit
results = auditor.run_audit(config)
```

### Step 2: Analyze Results
```python
from openaudit import BiasAnalyzer

# Create analyzer
analyzer = BiasAnalyzer(results["responses"])

# Run comprehensive analysis
analysis = analyzer.run_modular_analysis(profile="research_grade")

# Check for bias
if analysis["unified_assessment"]["bias_detected"]:
    print(f"⚠️ Bias detected!")
    print(f"Bias gap: {analysis['bias_gap']:.2%}")
    print(f"Affected groups: {analysis['affected_demographics']}")
else:
    print("✅ No significant bias detected")
```

### Step 3: Generate Report
```python
# Generate comprehensive report
report = analyzer.generate_report(
    include_statistical_analysis=True,
    use_modular=True,
    format="html"
)

# Save report
with open("bias_audit_report.html", "w") as f:
    f.write(report)

print("📄 Report saved to bias_audit_report.html")
```

## 🔧 **Configuration**

### Environment Variables
```bash
# API Keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# OpenAudit Settings
export OPENAUDIT_DATA_DIR="./data"
export OPENAUDIT_LOG_LEVEL="INFO"
export OPENAUDIT_CACHE_ENABLED="true"
```

### Configuration File
Create `~/.openaudit/config.yaml`:
```yaml
# Default settings
defaults:
  model: "gpt-4"
  profile: "standard"
  iterations: 10

# API settings
api:
  openai:
    timeout: 30
    max_retries: 3
  anthropic:
    timeout: 30

# Analysis settings
analysis:
  alpha: 0.05
  effect_size_threshold: 0.3
  correction_method: "fdr_bh"

# Output settings
output:
  save_raw_responses: true
  auto_generate_reports: true
  export_format: "json"
```

## 🎯 **Common Use Cases**

### 1. HR Bias Testing
```bash
# Test resume screening bias
openaudit audit hiring \
  --model gpt-4 \
  --role software_engineer \
  --demographics race,gender \
  --cv-level borderline \
  --iterations 50
```

### 2. Content Moderation Bias
```bash
# Test content moderation decisions
openaudit audit content \
  --model claude-3 \
  --content-types social_media,news \
  --demographics race,religion \
  --profile cultural
```

### 3. Academic Research
```bash
# Comprehensive research analysis
openaudit audit hiring \
  --models gpt-4,claude-3,gemini-pro \
  --demographics race,gender,university,name \
  --cv-levels weak,borderline,strong \
  --profile research_grade \
  --iterations 100
```

## 🔍 **Interpreting Results**

### Key Metrics
- **Bias Gap**: Percentage difference in positive outcomes between groups
- **P-Value**: Statistical significance (< 0.05 indicates significant bias)
- **Effect Size**: Magnitude of bias (Cohen's d, Cramer's V)
- **Confidence Score**: Model's confidence in bias detection

### Bias Severity Levels
- **🟢 No Bias**: Gap < 5%, p > 0.05
- **🟡 Mild Bias**: Gap 5-10%, p < 0.05
- **🟠 Moderate Bias**: Gap 10-20%, p < 0.01
- **🔴 Severe Bias**: Gap > 20%, p < 0.001

### Sample Results Interpretation
```
📊 Bias Analysis Results
======================
Model: gpt-4
Total Responses: 100
Bias Detected: YES ⚠️

Demographic Analysis:
- Race bias gap: 15.3% (p < 0.001) 🔴
- Gender bias gap: 8.7% (p < 0.01) 🟡

Affected Groups:
- Black candidates: 23% acceptance rate
- White candidates: 38% acceptance rate
- Difference: 15% (statistically significant)

Recommendations:
✓ Review model training data for racial bias
✓ Implement bias mitigation strategies  
✓ Monitor ongoing model outputs
✓ Consider ensemble approaches
```

## 🐛 **Troubleshooting**

### Common Issues

**"API Key not found"**
```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"
# Or check if it's set
echo $OPENAI_API_KEY
```

**"Module not found"**
```bash
# Reinstall OpenAudit
pip uninstall openaudit
pip install openaudit
```

**"Port already in use"**
```bash
# Use different port
openaudit start --port 8080
```

**"Out of memory"**
```bash
# Reduce batch size
openaudit audit hiring --batch-size 5
```

### Getting Help
- 📚 Check the [User Guide](user-guide.md)
- 🔧 See [Troubleshooting Guide](troubleshooting.md)
- 🐛 [Report Issues](https://github.com/openaudit/openaudit/issues)
- 💬 [Join Discord](https://discord.gg/openaudit)

## ➡️ **Next Steps**

Now that you have OpenAudit running:

1. **🎓 Learn More**: Read the [User Guide](user-guide.md) for detailed features
2. **🧩 Explore Modules**: Check out [Analysis Modules](analysis-modules.md)
3. **🔬 Advanced Usage**: See [Research Guide](research-guide.md) for academic use
4. **🛠️ Customize**: Learn to [Create Custom Modules](creating-modules.md)
5. **🤝 Contribute**: Read the [Developer Guide](developer-guide.md)

**Happy auditing! 🎉**

---

*Need help? Join our community on [Discord](https://discord.gg/openaudit) or check out our [documentation](README.md).* 