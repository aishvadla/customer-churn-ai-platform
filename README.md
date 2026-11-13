# Customer Churn AI

![CI](https://github.com/aishvadla/customer-churn-ai-platform/actions/workflows/main.yaml/badge.svg)

An **end-to-end churn prediction service**: it predicts which telecom customers are likely to leave, explains *why* with SHAP, and writes a short retention note for the customer success team with an LLM, served through a FastAPI REST API.

```
Customer record ──► Validation ──► Churn model ──► SHAP reasons ──► LLM retention note
                    (Pydantic)     (scikit-learn)   (top 3 drivers)   (Claude, grounded)
```

---

## Example

**Request** (`POST /predict`): a customer one month in, on a month-to-month fiber plan, paying by electronic check.

**Response:**
```json
{
  "churn_probability": 0.855,
  "will_churn": true,
  "reasons": [
    {"feature": "tenure", "value": 1, "impact": 0.964, "effect": "increases churn risk"},
    {"feature": "Contract", "value": "Month-to-month", "impact": 0.501, "effect": "increases churn risk"},
    {"feature": "InternetService", "value": "Fiber optic", "impact": 0.459, "effect": "increases churn risk"}
  ],
  "retention_note": "This customer has a high churn risk of 86%, mainly because they have been a customer for only 1 month and are on a month-to-month contract, which are both associated with leaving sooner. Their fiber optic internet service is also linked to higher churn risk. The most useful action is to offer an incentive to move to a longer-term contract, such as a discounted rate for committing to 12 months, and to make early contact to confirm they are satisfied with the service."
}
```

---

## Results

Evaluated once on a held-out test set of 705 customers (187 churners, 518 non-churners).

| Metric | Score |
|--------|-------|
| Test ROC-AUC | **0.856** |
| Test PR-AUC | **0.641** (random baseline ≈ 0.27) |
| Churn recall @ 0.5 | **0.82** |
| Churn precision @ 0.5 | 0.53 |
| Accuracy | 0.76 (majority-class baseline: 0.73) |

**In business terms:** flagging 41% of customers catches 82% of churners, and 53% of flagged customers actually churn.

| | Predicted: stay | Predicted: churn |
|---|---|---|
| **Actually stayed** | 385 | 133 (false alarms) |
| **Actually churned** | 34 (missed) | 153 (caught) |

### Model selection

| Model | Val ROC-AUC |
|-------|-------------|
| **Logistic regression** (selected) | 0.862 |
| Random forest (30-iteration randomized search, 5-fold CV) | 0.865 |

The tuned random forest didn't meaningfully beat the baseline. The training step applies an explicit rule: **prefer the simpler model unless the complex one wins by more than 0.01 ROC-AUC**. Without it, the selected model flipped between runs on differences smaller than validation noise. Logistic regression is stable, fast and easy to explain.

### Choosing the threshold

`class_weight="balanced"` favours catching churners over avoiding false alarms. The threshold is configurable (`ModelEvaluatorConfig.threshold`): raising it flags fewer customers with higher precision; lowering it catches more churners. The right value depends on cost: if a retention offer costs far less than a lost customer, high recall is worth the false alarms.

---

## What drives churn

EDA ([`notebooks/eda.ipynb`](notebooks/eda.ipynb), training split only) and the model's SHAP explanations agree:

| Driver | Churn rate |
|---|---|
| First 6 months as a customer | **53%** (vs under 10% after four years) |
| Month-to-month contract | **42.7%** (vs 11.2% one-year, 2.9% two-year) |
| Fiber optic internet | **41.3%** (vs 19.2% DSL) |
| Paying by electronic check | **44.8%** (vs 15.7–16.9% automatic payments) |

**Highest-risk profile:** a new customer on a month-to-month fiber plan, paying by electronic check.

---

## Architecture

```
Raw Data (IBM Telco Customer Churn)
        │
        ▼
┌─────────────────┐
│  Data Ingestion │  Reads raw CSV, three-way split (train/val/test)
└────────┬────────┘
         ▼
┌──────────────────────┐
│  Data Preprocessing  │  One sklearn ColumnTransformer, saved as preprocessor.pkl:
│                      │  type conversion, "No internet/phone service" → "No",
│                      │  imputation, scaling, binary + one-hot encoding.
│                      │  Fitted on train only.
└────────┬─────────────┘
         ▼
┌─────────────────┐
│  Model Training │  Logistic regression baseline vs. tuned random forest,
│                 │  simplicity rule, saves model.pkl + SHAP background sample
└────────┬────────┘
         ▼
┌──────────────────┐
│  Model Evaluate  │  ROC-AUC, PR-AUC, confusion matrix, per-class report
│                  │  on the test set, used once → metrics.json
└────────┬─────────┘
         ▼
┌──────────────────────────────────────────────────────────┐
│  Prediction Pipeline                                     │
│  raw record → preprocessor → model → probability          │
│            → ChurnExplainer (SHAP) → top 3 reasons         │
│            → RetentionNoteGenerator (LLM) → note if at risk│
└────────┬─────────────────────────────────────────────────┘
         ▼
┌──────────────────────┐
│  FastAPI REST API    │  POST /predict, GET /health, interactive /docs
└──────────────────────┘
```

---

## Engineering Highlights

**One preprocessing artifact, no train/serve skew.** All cleaning (type conversion, category normalisation) lives inside the saved `ColumnTransformer` via `FunctionTransformer` steps. Training and serving apply exactly the same transformations, and inference can't forget a step.

**Explanations grounded in the model.** SHAP values are grouped back from one-hot columns to original features (SHAP values are additive, so `Contract_One year` + `Contract_Two year` = the effect of `Contract`), ranked by absolute impact, and returned with the customer's real values. `shap.Explainer` keeps the explainer model-agnostic.

**LLM grounded in SHAP, not guessing.** The LLM receives only the probability and the top three reasons in plain English (with units, e.g. "1 months"). The system prompt restricts it to those facts, asks for "associated with" rather than causal language, and asks for one action tied to the top reason. Raw model internals (log-odds) are never sent to the LLM.

**Graceful degradation.** If the API key is missing or the LLM call fails, `/predict` still returns the probability and reasons with `retention_note: null`. The ML result never depends on an external service. Notes are generated only for at-risk customers, saving cost and latency.

**Robust LLM parsing.** Responses are parsed by block type, not position, so thinking blocks before the text don't break parsing. Token usage and `stop_reason` are logged on every call.

**Validation at the edge.** Pydantic `Literal` types restrict each category to the values the model was trained on, and numeric ranges reject inputs outside the training distribution (e.g. a $0 monthly bill). Invalid requests get a 422 before reaching the model.

**Data issues found through explainability.** SHAP plots revealed unhandled "No internet service" categories encoded as `-1`; fixing them is covered by a regression test. SHAP also showed `TotalCharges` (correlation 0.83 with tenure) producing counter-intuitive explanations, so it was dropped, at a cost of 0.001 test ROC-AUC.

**CI-safe tests.** Tests train small models into pytest's `tmp_path`, mock the LLM API, and run on a fresh CI checkout with no saved artifacts or API key. They check behaviour, not just types: a high-risk customer must outscore a low-risk one, and the top reasons must point in the right direction.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | scikit-learn (LogisticRegression, RandomForestClassifier, RandomizedSearchCV) |
| Explainability | SHAP |
| LLM | Anthropic API (Claude Haiku) |
| API Serving | FastAPI, Pydantic, Uvicorn |
| Data | Pandas, NumPy, Matplotlib |
| Testing | Pytest (session-scoped fixtures, mocked LLM client) |
| Code Quality | Black (pre-commit) |
| Environment | uv, pyproject.toml |
| CI/CD | GitHub Actions |

---

## Project Structure

```
churn-ai/
├── app/
│   └── app.py                     # FastAPI service
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
├── src/
│   ├── data/
│   │   ├── ingestion.py           # Read raw data, train/val/test split
│   │   └── preprocessing.py       # ColumnTransformer with cleaning steps
│   ├── models/
│   │   ├── model_registry.py      # Model definitions
│   │   ├── train.py               # Baseline + tuned RF, simplicity rule
│   │   ├── evaluate.py            # Test-set metrics → metrics.json
│   │   └── explain.py             # ChurnExplainer: model-agnostic SHAP reasons
│   ├── llm/
│   │   └── retention_note.py      # Prompt building + LLM call
│   ├── pipeline/
│   │   ├── training_pipeline.py   # End-to-end training orchestration
│   │   └── prediction_pipeline.py # Predict + explain + note
│   └── utils/                     # Logging, exceptions, save/load helpers
├── tests/                         # Data, model, explanation, LLM and pipeline tests
├── main.py                        # Training entry point
├── .github/workflows/ci.yaml      # Lint + test on every push
└── pyproject.toml                 # Dependencies + build config (uv)
```

---

## Quick Start

**1. Clone and install:**
```bash
git clone https://github.com/aishvadla/customer-churn-ai-platform.git
cd customer-churn-ai-platform
uv sync
```

**2. Add your Anthropic API key** (optional: without it, predictions work and `retention_note` is `null`):
```bash
echo "ANTHROPIC_API_KEY=your-key" > .env
```

**3. Train and evaluate:**
```bash
uv run python main.py
```

**4. Start the API:**
```bash
uv run uvicorn app.app:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs

**5. Make a prediction:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
       "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
       "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
       "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
       "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0
     }'
```

**6. Run tests:**
```bash
uv run pytest -v
```

---

## API Reference

| Endpoint | Description |
|---|---|
| `POST /predict` | Churn probability, decision, top 3 SHAP reasons, and a retention note for at-risk customers |
| `GET /health` | Service health check |
| `GET /docs` | Interactive API documentation |

Invalid input (unknown category, out-of-range value) returns **422** with the offending field.

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, binary target `Churn` (26.5% positive).

| Feature group | Columns | Preprocessing |
|---------------|---------|---------------|
| Numeric | tenure, MonthlyCharges | to_numeric, mean imputation, StandardScaler |
| Binary | Partner, Dependents, PhoneService, PaperlessBilling, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | "No internet/phone service" → "No", then No/Yes → 0/1 |
| Nominal | gender, InternetService, Contract, PaymentMethod | OneHotEncoder (drop first) |
| Passthrough | SeniorCitizen | Already 0/1 |
| Dropped | customerID, TotalCharges | ID; redundant with tenure (r = 0.83) |

---

## Next Steps

- **Out-of-distribution handling:** flag (not just reject) unusual inputs with warnings; monitor live input drift against training data; retrain when patterns change (e.g. new promotions)
- **LLM evaluation:** score notes across 20–30 customers for grounding, causal language and action quality
- **Business-specific offers:** give the LLM the company's actual retention offers to choose from
- **API tests** with FastAPI's `TestClient`
- **Deployment:** Docker image and a hosted endpoint, with the API key from a secret manager
- Cost-based threshold selection
- Feature engineering (e.g. number of services)

---

## Author

**Aishwarya Vadlamudi**, AI Software Architect transitioning to ML Engineering
[GitHub](https://github.com/aishvadla) · [LinkedIn](https://www.linkedin.com/in/aishwaryavadlamudi/)