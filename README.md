# Healthcare Readmission Risk Platform
### Capstone Project 7 — Industry-Integrated AI Systems Synthesis

---

## Overview

This repository contains the complete deliverables for the final capstone synthesis project.
The system predicts 30-day hospital readmission risk by integrating five prior capstone
projects into a single, cohesive AI pipeline.

**Industry:** Healthcare  
**Problem:** Unplanned 30-day readmissions affect ~15% of Medicare patients and cost
the US healthcare system over $26 billion annually.  
**Solution:** A five-layer AI platform — EDA → ML → DL → Generative AI → Agentic Triage

---

## Repository Structure

```
industry_ai_synthesis/
├── notebooks/
│   └── integrated_healthcare_ai_system.ipynb   ← MAIN ARTIFACT (start here)
├── src/
│   ├── readmission_pipeline.py                  ← Reusable Python module
│   └── architecture_diagram.py                  ← Diagram generator
├── diagrams/
│   ├── system_architecture.png                  ← System overview
│   ├── eda_distributions.png                    ← Layer 1 output
│   ├── ml_results.png                           ← Layer 2 output
│   ├── dl_loss_curve.png                        ← Layer 3 output
│   ├── ensemble_confusion_matrix.png            ← Ensemble evaluation
│   └── triage_results.png                       ← Layer 5 output
├── docs/
│   └── Reflective_Synthesis_Paper.docx          ← 1,500–2,000 word paper
├── requirements.txt
└── README.md
```

---

## System Architecture

```
EHR Patient Data
      │
      ▼
Layer 1 — EDA & Feature Engineering      ← Project 1
      │
      ▼
Layer 2 — ML Classifiers (RF / LR / GB)  ← Project 2
      │
      ▼
Layer 3 — Deep Learning MLP              ← Project 3
      │
      ▼  (Weighted Ensemble: 60% ML + 40% DL)
      │
      ▼
Layer 4 — GenAI Patient Summary          ← Project 5
      │
      ▼
Layer 5 — Agentic Triage & Dispatch      ← Project 6
      │
      ▼
Care Team Dashboard
```

---

## Prior Project Integration

| Prior Project | Contribution |
|---|---|
| **Project 1 — Data & Statistical Reasoning** | EDA pipeline, feature distribution analysis, composite risk_index |
| **Project 2 — Machine Learning** | Logistic Regression, Random Forest, Gradient Boosting classifiers |
| **Project 3 — Deep Learning** | MLP readmission risk network, training loop, loss diagnostics |
| **Project 5 — Generative AI** | Plain-language patient risk summaries (template + LLM hook) |
| **Project 6 — Agentic AI** | Tool-calling orchestration loop over discharge queue |

---

## Setup and Execution

### Prerequisites

- Python 3.9+
- VS Code with the Jupyter extension (or JupyterLab / Anaconda)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the main notebook

Open `notebooks/integrated_healthcare_ai_system.ipynb` in VS Code and run all cells.

The notebook is self-contained. All data is generated synthetically — no external
data files or API keys are required.

### Run the module smoke-test

```bash
python src/readmission_pipeline.py
```

### Regenerate diagrams

```bash
python src/architecture_diagram.py
```

---

## Evaluation Results (held-out test set)

| Model | AUC |
|---|---|
| Logistic Regression | ~0.74 |
| Random Forest | ~0.77 |
| Gradient Boosting | ~0.76 |
| Deep Learning MLP | ~0.75 |
| **Ensemble (60% ML + 40% DL)** | **~0.79** |

---

## Ethical Considerations

- **Algorithmic bias**: Disparate-impact audit required across insurance / demographic groups before clinical deployment.
- **Explainability**: SHAP values planned for production version.
- **Accountability**: Human-in-loop clinician confirmation required for all AI-generated care actions.
- **Privacy**: All data must be de-identified and governed under HIPAA before clinical use.

---

## Submission Checklist

- [x] Integrated AI system artifact (notebook + Python module)
- [x] Reflective Synthesis Paper (`docs/Reflective_Synthesis_Paper.docx`)
- [x] System architecture diagram (`diagrams/system_architecture.png`)
- [x] Supporting diagrams (EDA, ROC curves, triage results)
- [x] `requirements.txt`
- [x] Mentor presentation preparation (see paper Section 7)

---

## References

- Jencks et al. (2009). *Rehospitalizations among patients in the Medicare fee-for-service program.* NEJM, 360(14).
- Barocas, Hardt, & Narayanan (2019). *Fairness and machine learning.* fairmlbook.org.
- Johnson et al. (2016). *MIMIC-III, a freely accessible critical care database.* Scientific Data, 3.
- Rajpurkar et al. (2022). *AI in health and medicine.* Nature Medicine, 28(1).
