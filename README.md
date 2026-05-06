# 🧩 Risk Evaluation Toolkit

This repository contains the code and data for evaluating reasoning-level risks in scientific content generation, as described in our NeurIPS submission.

It includes:
- A risk evaluation pipeline (1-step and 2-step)
- A taxonomy of reasoning risks
- A benchmark dataset
- Scripts to reproduce experimental results

---

## 🚀 Overview

The toolkit supports two evaluation strategies:

### 1. One-Step Risk Assessment
- Directly evaluates whether a given risk is present in a model output
- Uses a single LLM call per risk

### 2. Two-Step Risk Assessment
- Generates structured evaluation criteria from the taxonomy
- Evaluates criteria using a DAG-style aggregation
- Produces more structured and robust risk judgments

---

## 🔍 For Reviewers

To reproduce the main experimental results, run:

```bash
python scripts/test_harness_1_step.py
python scripts/test_harness_2_step.py
```

These scripts:
- Load the benchmark dataset
- Run the respective risk evaluation method (1-step or 2-step)
- Compute precision, recall, and per-risk metrics
- Print results directly to the terminal

Model selection is controlled inside each script. OpenAI models are selected by passing a model name as a string (for example, `"gpt-4.1-mini"`), while Watsonx and RITS models are selected using the corresponding enum values. The scripts include commented examples showing how to switch between providers and models.

No additional pipeline setup is required beyond installing dependencies and setting the relevant provider credentials.

---

## 📊 Benchmark Dataset

The benchmark is provided as a JSONL file:

```
data/benchmark/scientific_risk_benchmark.jsonl
```

Each instance contains:
- question: contextual prompt
- paragraph: generated scientific content
- risk_reason: injected reasoning risk (or none)
- risk_id (or none)

The dataset includes:
- 45 risk-free reference instances
- 252 corrupted instances with a single injected risk

---

## 🧠 Model Providers

The toolkit supports three model backends:

- OpenAI (default if model name is a string)
- Watsonx
- RITS

Model selection is handled by the type of model identifier passed to the evaluator:

- If the model is provided as a string, it is treated as an OpenAI model name.
- If the model is provided as a `WatsonxModels` enum value, the Watsonx backend is used.
- If the model is provided as a `RITSModels` enum value, the RITS backend is used.

Examples:

```python
model = "gpt-4.1-mini"                       # OpenAI
model = "gpt-4.1-nano"                       # OpenAI
model = WatsonxModels.GRANITE_4H_SMALL       # Watsonx
model = WatsonxModels.LLAMA_3_3_70B_INSTRUCT # Watsonx
model = RITSModels.LLAMA_3_3_70B_INSTRUCT    # RITS
```

### Configuration

Set credentials in a .env file:

```
RITS_API_KEY=<your_rits_api_key>
RITS_BASE_URL=<rits_base_url>

WATSONX_APIKEY=<your_watsonx_api_key>
WATSONX_URL=<watsonx_base_url>
WATSONX_PROJECT_ID=<watsonx_project_id>

OPENAI_API_KEY=<your_openai_api_key>
```


---

## ⚙️ Environment Setup

This project uses [**uv**](https://github.com/astral-sh/uv) for dependency management with **Python 3.12**.

### 1. Create environment
```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 2. Install dependencies
```bash
uv pip install -e .
```

### 3. Additional dependencies
Then install `langchain-ibm` separately.
```bash
uv pip install langchain-ibm==0.3.19
```

This downgrades `langchain-core`, so you must then install this updated version.
```bash
uv pip install langchain-core==1.0.0
```

---

## 🧩 Risk Taxonomy

Taxonomies are stored in:

```
risk/builders/risk_taxonomies/*.yaml
```

Each risk must define:
- id
- description

Custom taxonomies can be added by:
1. Creating a YAML file
2. Registering it in:
   - `risk/builders/risk.py`
   - `risk/builders/risk_report.py`

---

## 📂 Project Structure
```
risk/
├── builders/
│   ├── risk.py
│   ├── risk_report.py
│   ├── risk_taxonomies/
├── configs/
│   ├── project.py
│   ├── models.py
│   ├── prompts.py
├── tools/
│   ├── pipeline.py
├── utils.py

data/
├── benchmark/
│   ├── scientific_risk_benchmark.yaml
│   └── scientific-risk-croissant.jsonld
├── inputs/
│   ├── inputs_with_risks_full.yaml
│   └── inputs_no_risks_full.yaml
├── outputs/   # generated at runtime

scripts/
├── test_harness_1_step.py
├── test_harness_2_step.py
```
---

## 🤝 Notes

- The risk report component is optional and not required for reproducing results
- The primary evaluation is performed via the test harness scripts
- All benchmark instances are SME-authored or SME-validated

---

## 📌 Summary

This repository enables:
- Controlled evaluation of reasoning-level risks
- Comparison of structured vs direct risk detection methods
- Reproducibility of all experimental results reported in the paper