# OpenAudit - LLM Bias Testing Framework

A **production-ready** framework for detecting algorithmic bias in Large Language Models (LLMs) using established audit study methodologies. Successfully detects real bias patterns and model inconsistencies.

## 🚨 Key Findings

**OpenAudit has detected significant bias in OpenAI models:**
- **25% hiring rate gap** between demographic groups
- **0% model consistency** across identical prompts  
- **Systematic discrimination** in hiring decisions

## 🎯 Key Features

- **Multi-LLM Testing**: Simultaneous testing across OpenAI model family (GPT-3.5, GPT-4o, GPT-4-turbo, GPT-4o-mini, GPT-4.1 series, o1/o3 reasoning models)
- **Demographic Bias Detection**: Systematic testing across race, gender, and other demographic factors
- **Model Consistency Analysis**: Compare responses across different models from same provider
- **Statistical Rigor**: Automated bias detection with significance testing (>5% moderate, >10% significant)
- **Web Dashboard**: Interactive visualization of bias patterns and results
- **Command Line Tools**: Quick analysis and batch processing capabilities

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys**:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

3. **Run bias experiment**:
   ```bash
   python3 bias_testing_framework.py
   ```

4. **Launch web dashboard**:
   ```bash
   python3 web_interface.py         # Historical results dashboard
   # Visit http://localhost:5002
   
   python3 live_experiment.py       # Live experiment monitor
   # Visit http://localhost:5003
   ```

## 📊 Web Interfaces

### Historical Results Dashboard (`http://localhost:5002`)
- **Run Selection**: Choose from historical experiment runs
- **Aggregation**: Combine multiple runs for larger sample sizes
- **Bias Visualization**: Interactive charts showing hiring rates by demographic
- **Model Comparison**: Compare consistency across different LLM models
- **Detailed Analysis**: Drill down into specific responses and reasoning

### Live Experiment Monitor (`http://localhost:5004`)
- **Real-time Prompt Visualization**: See exactly what prompts are being sent
- **Full CV/Resume Display**: View complete candidate profiles during experiments
- **Editable Prompt Templates**: Customize base prompts for different research scenarios
- **CV Qualification Customization**: Choose weak, borderline, or strong candidate profiles
- **Variable Highlighting**: Names and other variables highlighted in prompts
- **Live Response Stream**: Watch model responses arrive in real-time
- **Progress Tracking**: Monitor experiment progress and timing with accurate iteration counting
- **Experiment Control**: Stop running experiments anytime with dedicated stop button
- **Instant Bias Detection**: Live statistics and bias alerts during experiments

### 🎯 Prompt Customization Features
**The live interface now supports fully editable prompt templates for advanced research:**
- **Custom Research Scenarios**: Edit the base prompt to test different bias detection approaches
- **A/B Testing**: Compare results with and without bias-related language (e.g., "This is a controlled research study...")
- **Domain Adaptation**: Modify prompts for different industries (healthcare, finance, education)
- **Guardrail Testing**: Test how different prompt structures affect model compliance
- **Variable Flexibility**: Use {name}, {university}, {experience}, {address}, and {cv_content} variables in custom prompts

### 📄 Realistic CV/Resume Enhancement
**OpenAudit now includes comprehensive CV/resume generation for realistic hiring scenarios:**
- **Detailed Professional Profiles**: Full CVs with work experience, education, skills, and achievements
- **Role-Specific Templates**: Tailored CVs for software engineers, managers, and sales professionals
- **Dynamic Content Generation**: Each candidate gets unique but comparable qualifications
- **Customizable Qualification Levels**: Choose between weak, borderline, or strong candidate profiles
- **Bias Detection Optimization**: Identical qualification levels across demographic groups for fair comparison
- **Live CV Preview**: See exactly what CVs are being generated during experiments

**CV Qualification Levels:**
- **Weak Candidates**: 2.5-3.0 GPA, minimal experience, basic achievements
- **Borderline Candidates**: 3.0-3.4 GPA, modest experience, average achievements (default)
- **Strong Candidates**: 3.5-4.0 GPA, excellent experience, outstanding achievements

**CV Features Include:**
- Professional summaries and contact information
- Educational background with GPA and graduation details
- Work experience with specific metrics and achievements
- Technical skills and certifications
- Projects and accomplishments
- Realistic timelines and career progression
- Deterministic generation ensuring ceteris paribus (only name varies)

### Using the Interfaces

**Historical Dashboard:**
1. **Single Run Analysis**: Select one experiment run and click "Analyze Selected Run"
2. **Multi-Run Aggregation**: Select multiple runs and click "Aggregate Multiple Runs"
3. **View Results**: Interactive charts show bias patterns and model inconsistencies

**Live Monitor:**
1. **Configure Experiment**: Select job role, models, and iterations
2. **Edit Prompt Template**: Customize the base prompt sent to all models
3. **Customize CV Level**: Choose weak, borderline, or strong candidate qualifications
4. **Preview Sample CV**: View realistic resume content generated for candidates
5. **Configure Demographics**: Select and edit demographic groups and names
6. **Preview Prompts**: See exactly what will be sent before running
7. **Start Live Experiment**: Watch real-time prompt dispatch and responses
8. **Monitor CVs**: View complete candidate profiles during experiments
9. **Control Experiment**: Stop running experiments anytime if needed
10. **Monitor Bias**: Live statistics show bias patterns as they emerge

**Example Prompt Customization:**
```
Original: "IMPORTANT: This is a controlled research study conducted by an AI safety team..."
Modified: "You are a hiring manager. Please evaluate this candidate for a software engineering position..."
```
This allows testing whether mentioning "bias research" affects model responses.

**Impact of Realistic CVs on Bias Detection:**
- **Enhanced Credibility**: Full CVs make hiring decisions more realistic and meaningful
- **Controlled Variables**: All candidates have comparable qualifications across demographic groups
- **Detailed Context**: Rich professional profiles provide more realistic decision-making scenarios
- **Reduced Noise**: Comprehensive information reduces arbitrary decision-making factors
- **Better Insights**: More realistic scenarios lead to more actionable bias detection results
- **Customizable Difficulty**: Adjust candidate strength to avoid 100% hire rates
- **Live Transparency**: See exactly what qualifications are being evaluated

**Note:** The repository contains runs with different qualification levels:
- **Early runs**: Harvard University + startup experience (higher baseline rates)
- **Recent runs**: State University + limited experience (lower baseline rates)
- **Aggregation**: Combines all runs for larger sample sizes

## 🔍 Detailed Results

### Model Inconsistency (0% Agreement)
**Latest Run Results (Borderline Candidates):**
- **GPT-3.5-turbo**: 100% hire rate (too permissive)
- **GPT-4o**: 47% hire rate (realistic but biased)
- **GPT-4o-mini**: 0% hire rate (too restrictive)
- **GPT-4-turbo**: 0% hire rate (too restrictive)

**Note:** Earlier runs with stronger qualifications (Harvard + startup experience) show higher overall rates, but similar bias patterns.

### Expanded Model Support
**OpenAudit now supports testing across OpenAI's complete model lineup:**
- **GPT-3.5 Series**: gpt-3.5-turbo
- **GPT-4 Series**: gpt-4o, gpt-4o-mini, gpt-4-turbo
- **GPT-4.1 Series**: gpt-4.1-nano, gpt-4.1-mini, gpt-4.1 (latest generation)
- **Reasoning Models**: o1-preview, o1-mini, o1, o3-mini, o3

*Note: Some newer models may not be available via API yet. The system gracefully handles unavailable models and will show availability status during setup.*

## 🔧 Recent Improvements

### Live Experiment Enhancements
- **Fixed Iteration Counting**: Resolved double-counting bug that was multiplying iterations
- **Real-time CV Display**: Added full CV/resume visibility during experiments
- **Stop Experiment Control**: Added ability to stop long-running experiments
- **Prompt Processing Display**: Fixed "no prompt currently processing" issue
- **CV Qualification Levels**: Added weak/borderline/strong candidate options
- **Progress Tracking**: Accurate experiment progress with proper iteration counting

### CV System Improvements
- **Borderline Candidates**: Adjusted default CVs to create realistic hiring decisions
- **Deterministic Generation**: Ensures ceteris paribus (only name varies between candidates)
- **Live Preview**: Real-time CV preview with customization options
- **Qualification Control**: Prevent 100% hire rates with appropriate candidate strength

### Demographic Bias (25% Gap)
**Consistent across qualification levels:**
- **Black females**: 50.0% hire rate (highest)
- **White males**: 25.0% hire rate (lowest)
- **Other groups**: 31.2% - 37.5% hire rate

### Statistical Significance
- **Bias detection**: >10% gap triggers significant bias alert
- **Sample size**: 128 responses across 8 demographic groups
- **Reproducibility**: Consistent bias patterns across multiple runs and qualification levels

## 📁 Project Structure

```
OpenAudit/
├── bias_testing_framework.py    # Main bias testing framework
├── multi_llm_dispatcher.py      # Multi-LLM prompt dispatcher
├── response_analyzer.py         # Detailed response analysis
├── web_interface.py             # Flask web dashboard
├── templates/
│   └── index.html               # Web dashboard HTML
├── runs/                        # Experiment results storage
│   ├── hiring_bias_experiment_*.json
│   └── bias_analysis_report_*.txt
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🛠️ Advanced Usage

### Custom Test Cases

Modify `HiringBiasTest` class to create custom scenarios:

```python
test_case = BiasTestCase(
    template="Custom prompt with {variable}",
    variables={"variable": ["option1", "option2"]},
    domain="custom_domain"
)
```

### Adding New LLM Providers

Extend `MultiLLMDispatcher` to support additional providers:

```python
self.models["new_model"] = ChatNewProvider(
    model="model-name",
    temperature=0
)
```

### Analyzing Results

Use `response_analyzer.py` for detailed analysis:

```python
python3 response_analyzer.py
```

## 🔍 Troubleshooting

### Common Issues

**Getting 100% hire rates:**
- Use "Customize CV Level" to select "Weak" or "Borderline" candidates
- Default borderline candidates should produce realistic hiring decisions
- Preview CVs to ensure appropriate qualification levels

**Experiment shows doubled iterations:**
- This was a bug that has been fixed
- Ensure you're using the latest version of the code

**"No prompt currently processing" message:**
- This display issue has been resolved
- The interface now shows real-time prompt processing and CV content

**Need to stop a running experiment:**
- Use the red "Stop Experiment" button that appears during experiments
- Experiments can be stopped gracefully with partial results

**Can't see what CVs are being generated:**
- Use "Preview Sample CV" to see example CVs
- During experiments, CV content is displayed in real-time
- Use "Customize CV Level" to control qualification strength

## 📈 Methodology

Based on established algorithm audit practices (Metaxa et al., 2021):

1. **External Evaluation**: Study algorithms without internal access
2. **Systematic Testing**: Randomized controlled experiments across demographics
3. **Statistical Rigor**: Aggregate analysis to detect disparate impact
4. **Reproducibility**: Consistent methodology across different test runs

## 🔬 Research Applications

- **AI Safety Research**: Identify potential biases in production LLMs
- **Fairness Testing**: Systematic evaluation of AI systems for discrimination
- **Model Comparison**: Compare bias patterns across different AI providers
- **Regulatory Compliance**: Generate evidence for bias detection requirements

## 🤝 Contributing

This is an open-source project for AI safety research. Key areas for contribution:

- Additional bias testing domains (lending, healthcare, education)
- Support for more LLM providers
- Enhanced statistical analysis methods
- Improved visualization and reporting

## 📄 License

GPL-3.0 License - see LICENSE file for details.

## 🙏 Acknowledgments

- Based on algorithm audit methodology from Metaxa et al. (2021)
- Inspired by classic discrimination audits (Bertrand & Mullainathan, 2004)
- Built for the AI safety research community