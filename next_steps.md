## 🛠️ OpenAudit: Roadmap to v1.0 — Become the Standard for LLM Bias Audits

This document outlines the short- and mid-term priorities to position OpenAudit as the leading open-source framework for LLM auditing and fairness evaluation.

---

### ✅ GOAL

**Make OpenAudit the `lm-eval-harness` of LLM bias/fairness auditing.**
Modular, reproducible, extensible, and backed by real research + interactive UIs.

---

### 🧩 MILESTONE 1: Core Modularization (Week 1–2)

Refactor the system to support modular, extensible components like Eval Harness.

#### 📁 `audits/`

* [ ] Each audit type (e.g. hiring\_bias, loan\_bias, college\_admit) is its own folder with:

  * `template.yaml` (prompt structure)
  * `config.json` (demographics, CV strength, test cases)
  * `analyzer.py` (bias logic if custom)

#### 📁 `models/`

* [ ] Standardize wrapper classes:

  * [ ] `OpenAIModel`, `AnthropicModel`, `HuggingFaceModel`, `LocalInferenceServer`
  * [ ] Ensure parity of args: temp, top\_p, system prompt, etc.
  * [ ] Add availability/health check endpoint

#### 📁 `cv_templates/`

* [ ] Move resume generation logic into separate templates by domain (e.g. SWE, healthcare)
* [ ] Support levels: `weak`, `borderline`, `strong`
* [ ] Make this deterministic with seeding + hashing

---

### 📊 MILESTONE 2: Metrics & Analysis (Week 3–4)

#### ✅ Statistical Analysis

* [ ] Bias gap % calculation per demographic
* [ ] p-value + significance level thresholds (>5% moderate, >10% significant)
* [ ] Model consistency scoring (same input, diff output)

#### ✅ Reasoning Analyzer

* [ ] Token-level diff of rationales between accepted vs rejected
* [ ] Highlight *rationalization artifacts* (e.g. stricter language used on rejected group)
* [ ] Output HTML or Markdown file summarizing patterns

---

### 🧪 MILESTONE 3: Audit Templates Library (Week 4–5)

#### 🧩 `templates/` folder

* [ ] Add YAML prompt templates replicating known research:

  * [x] `paper_binary_basic_legal.yaml`
  * [x] `paper_binary_tamkin_warning.yaml`
  * [x] `paper_cot_tamkin_warning.yaml`
  * [ ] `company_context/meta.yaml`, `company_context/gm.yaml`
* [ ] Support placeholder variables:

  * `{name}`, `{cv_content}`, `{university}`, `{gender}`, `{race}`, etc.

#### 🔁 Prompt testing mode:

* [ ] CLI: `python3 test_template.py --template paper_binary_tamkin_warning.yaml`
* [ ] Output 3 filled examples with randomized names + preview

---

### 🌐 MILESTONE 4: Frontend Improvements (Week 6–7)

#### 🧪 Live Experiment UI

* [ ] Display real-time response stream by demographic
* [ ] Visual indicator of bias emergence
* [ ] Progress bar + ETA
* [ ] Download button for JSON results & CSV export

#### 📊 Historical Dashboard

* [ ] Add filters: by date, model, demographic, template
* [ ] Aggregation UI: drag & drop multiple runs → view summary
* [ ] Add bias gap charts (bar, heatmap)

---

### 📦 MILESTONE 5: Release v1.0 Package (Week 8)

#### 📦 Packaging & Docs

* [ ] `setup.py` + `requirements.txt` for `pip install openaudit[dashboard]`
* [ ] `docs/` folder with:

  * [ ] Getting Started (CLI + Web)
  * [ ] Adding new audit
  * [ ] Supported models
  * [ ] Bias interpretation guide
* [ ] Add example notebook (`notebooks/hiring_bias_demo.ipynb`)

#### 🧪 Command-Line CLI

```bash
# Run a full audit
python3 run_audit.py --template hiring_bias --model openai --variant borderline --output results.json

# Visualize
python3 start_unified.py
```

---

### 🎯 FUTURE PHASE: Bias Leaderboard (Optional)

* [ ] Define standard audit battery (e.g. 3 tasks x 8 demographics x 3 CV strengths)
* [ ] Enable pushing results to HF hub or public dashboard
* [ ] Community model submissions

---

### 🧠 Attribution & Vision

OpenAudit draws inspiration from:

* `lm-evaluation-harness` (for structure)
* Algorithmic audits from \[Metaxa et al. 2021], \[Marks & Karvonen 2025]
* Resume audit studies from Bertrand & Mullainathan (2004)

We aim to become **the default infra for LLM fairness testing** in research, enterprise, and regulation.