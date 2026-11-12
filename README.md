# Customer Churn AI

![CI](https://github.com/aishvadla/customer-churn-ai-platform/actions/workflows/ci.yaml/badge.svg)

A **production-style ML pipeline** that predicts which telecom customers are likely to churn, built with modular components, train/val/test discipline, shared training and serving logic, and CI-tested code.

> The goal is a churn model a retention team could act on: rank customers by risk, explain why each one is at risk, and serve predictions through an API. Explanations (SHAP), LLM-written retention notes and the API are in progress; see [Roadmap](#roadmap).

---

## Results

Evaluated once on a held-out test set of 705 customers (187 churners, 518 non-churners).

| Metric | Score |
|--------|-------|
| Test ROC-AUC | **0.865** |
| Test PR-AUC | **0.684** (random baseline ≈ 0.27) |
| Churn recall @ 0.5 | **0.81** |
| Churn precision @ 0.5 | 0.56 |
| Accuracy | 0.78 (majority-class baseline: 0.73) |

**In business terms:** flagging 38% of customers catches 81% of churners, and 56% of flagged customers actually churn.

**Confusion matrix (threshold 0.5):**

|  | Predicted: stay | Predicted: churn |
|---|---|---|
| **Actually stayed** | 399 | 119 (false alarms) |
| **Actually churned** | 35 (missed) | 152 (caught) |

### Model comparison (validation ROC-AUC)

| Model | Val ROC-AUC |
|-------|-------------|
| Logistic regression (baseline) | 0.865 |
| Random forest (tuned, 30-iteration randomized search, 5-fold CV) | **0.867** |

The tuned random forest only matched the logistic regression baseline: the gap is well within validation noise. Most of the churn signal in this dataset is close to linear (contract type, tenure, charges). The random forest was kept for exact, fast SHAP tree explanations; in production, the simpler model would be a reasonable choice.

### Choosing the threshold

`class_weight="balanced"` favours catching churners over avoiding false alarms. The threshold is configurable in `ModelEvaluatorConfig`: raising it flags fewer customers with higher precision, lowering it catches more churners. The right value depends on cost: if a retention offer costs far less than a lost customer, high recall is worth the false alarms.

---

## Architecture

```
Raw Data (IBM Telco Customer Churn)
        │
        ▼
┌─────────────────┐
│  Data Ingestion │  Reads raw CSV, three-way split (train/val/test)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Data Preprocessing  │  clean(): shared with inference (ID drop, type fixes,
│                      │  "No internet/phone service" → "No")
│                      │  ColumnTransformer: StandardScaler + binary encoding
│                      │  + OneHotEncoder, fitted on train only
│                      │  Saves preprocessor.pkl to artifacts
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │  Logistic regression baseline vs. tuned random forest
│                 │  (RandomizedSearchCV, scoring=roc_auc)
│                 │  Selects best on validation, saves model.pkl
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Model Evaluate  │  ROC-AUC, PR-AUC, confusion matrix, per-class report
│                  │  on the test set, used once. Saves metrics.json
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│ Prediction Pipeline  │  Raw customer record → clean() → preprocessor
│                      │  → churn probability + decision
└──────────────────────┘
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | Scikit-learn (LogisticRegression, RandomForestClassifier, RandomizedSearchCV) |
| Data Processing | Pandas, NumPy |
| Testing | Pytest with session-scoped fixtures |
| Code Quality | Black (pre-commit) |
| Environment | uv, pyproject.toml |
| CI/CD | GitHub Actions |

---

## Project Structure

```
churn-ai/
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
├── src/
│   ├── data/
│   │   ├── ingestion.py           # Read raw data, train/val/test split
│   │   └── preprocessing.py       # clean() + ColumnTransformer pipeline
│   ├── models/
│   │   ├── model_registry.py      # Model definitions (LR, RF, PyTorch net)
│   │   ├── train.py               # Baseline + tuned RF, select on val ROC-AUC
│   │   └── evaluate.py            # Test-set metrics → metrics.json
│   ├── pipeline/
│   │   ├── training_pipeline.py   # End-to-end training orchestration
│   │   └── prediction_pipeline.py # Single-customer inference
│   └── utils/
│       ├── logger.py              # Structured logging
│       ├── exception.py           # Custom exception handling
│       └── helper.py              # save/load object utilities
├── tests/
│   ├── conftest.py                # Session-scoped pipeline fixture
│   ├── test_data_ingestion.py
│   ├── test_data_preprocessing.py
│   ├── test_model_trainer.py
│   ├── test_evaluate.py
│   └── test_prediction_pipeline.py
├── .github/workflows/ci.yaml      # Lint + test on every push
└── pyproject.toml                 # Dependencies + build config (uv)
```

---

## Engineering Highlights

**No train/serve skew.** Raw-data cleaning lives in one `clean()` function, called by both the training pipeline and the prediction pipeline. The fitted `ColumnTransformer` is saved and reused at inference, so a new customer goes through exactly the transformations the model was trained on.

**Proper train/val/test separation.** The preprocessor is fitted on train only. Hyperparameters are tuned with cross-validation on train, models are compared on validation, and the test set is used once, for final reporting.

**Honest model selection.** A logistic regression baseline is trained first, and every tuned model is compared against it. The comparison is reported even when the complex model barely wins.

**Imbalance-aware evaluation.** With ~27% churners, accuracy is misleading (always predicting "stay" scores 73%). The pipeline reports ROC-AUC, PR-AUC and per-class precision/recall, and the threshold is configurable.

**CI-safe tests.** Tests train small models (`n_iter=2, cv=2`) into pytest's `tmp_path`, so they run in ~13 seconds, never overwrite real artifacts, and pass on a fresh CI checkout with no saved model. Thresholds (`ROC-AUC > 0.75`) catch silent regressions, not just crashes.

**Behavioural test.** The prediction test checks direction, not just output type: a new month-to-month fiber customer paying by electronic check must score higher churn risk than a 70-month, two-year-contract customer.

**Model registry.** Model definitions live in `model_registry.py`, separate from training logic, so configurations change in one place.

---

## Quick Start

**1. Clone and install:**
```bash
git clone https://github.com/aishvadla/customer-churn-ai-platform.git
cd customer-churn-ai-platform
uv sync
```

**2. Train and evaluate:**
```bash
uv run python -m src.pipeline.training_pipeline
```
Writes `preprocessor.pkl`, `model.pkl` and `metrics.json` to `artifacts/`.

**3. Run tests:**
```bash
uv run pytest -v
```

**4. Predict for one customer:**
```python
from src.pipeline.prediction_pipeline import PredictionPipeline

PredictionPipeline().predict({
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": "85.0",
})
# {"churn_probability": ..., "will_churn": True}
```

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 19 features, binary target `Churn` (~27% positive).

| Feature group | Columns | Preprocessing |
|---------------|---------|---------------|
| Numeric | tenure, MonthlyCharges, TotalCharges | Mean imputation + StandardScaler |
| Binary | Partner, Dependents, PhoneService, PaperlessBilling, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | "No internet/phone service" → "No", then No/Yes → 0/1 |
| Nominal | gender, InternetService, Contract, PaymentMethod | OneHotEncoder (drop first) |
| Already 0/1 | SeniorCitizen | Passed through |

---

## Roadmap

- [ ] **SHAP explanations:** top reasons behind each customer's churn risk
- [ ] **LLM retention notes:** turn SHAP reasons into a plain-English note for the retention team (Anthropic API)
- [ ] **FastAPI service:** `POST /predict` returning probability, decision and explanation; `GET /health`
- [ ] Compare against a PyTorch feedforward network (defined in `model_registry.py`)
- [ ] Feature engineering (e.g. number of services, charges per month of tenure)
- [ ] Cost-based threshold selection

---

## Author

**Aishwarya Vadlamudi**, AI Software Architect transitioning to ML Engineering
[GitHub](https://github.com/aishvadla) · [LinkedIn](https://www.linkedin.com/in/aishwaryavadlamudi/)