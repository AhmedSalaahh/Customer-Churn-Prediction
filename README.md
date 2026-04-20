# Customer Churn Predictor

Production-ready ML pipeline for telco customer churn prediction.  
Modularised from a research notebook – designed to grow stage-by-stage into a fully deployed service.

---

## Project layout

```
churn_predictor/
├── configs/
│   └── config.yaml          # Single source of truth for all parameters
│
├── src/
│   ├── data/
│   │   └── preprocessing.py  # Load → clean → split
│   ├── features/
│   │   ├── engineering.py    # Domain feature creation
│   │   └── pipeline.py       # sklearn ColumnTransformer + SMOTE
│   ├── models/
│   │   ├── training.py       # Build, train, calibrate, MLflow-log
│   │   └── registry.py       # Select best, save/load bundle
│   └── evaluation/
│       └── metrics.py        # Metrics, SHAP, business simulations
│
├── scripts/
│   ├── train.py              # CLI – full training pipeline
│   └── predict.py            # CLI – batch scoring
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_engineering.py
│   ├── test_pipeline.py
│   └── test_training_integration.py
│
├── requirements.txt
├── Makefile
└── README.md
```

---

## Quick start

```bash
# 1. Install dependencies
make install

# 2. Train all models (MLflow tracked)
make train

# 3. Inspect experiments in the UI
make mlflow-ui          # open http://localhost:5000

# 4. Batch predict
make predict INPUT=data/new_customers.csv

# 5. Run the full test suite
make test
```

---

## Configuration

All tunable values live in `configs/config.yaml` – no magic numbers in code.

| Section | Key parameters |
|---------|----------------|
| `data` | raw path, test split, random seed |
| `features` | numeric/categorical lists, service cols, contract risk map |
| `models` | per-model hyperparameters |
| `calibration` | method, CV folds |
| `business` | retention cost, revenue/customer, campaign budget |

---

## MLflow tracking

Each of the three models (Logistic Regression, Random Forest, XGBoost) is trained
inside its own `mlflow.start_run()`. Logged per run:

* **Params**: model type, hyper-parameters, calibration settings  
* **Metrics**: ROC-AUC, PR-AUC  
* **Artefacts**: calibrated sklearn model (pickle)

The best model bundle (`model + preprocessor + feature_names`) is also saved
locally to `outputs/best_model.pkl` for fast inference.

---

## Testing

| Test file | What it covers |
|-----------|---------------|
| `test_preprocessing.py` | Column drops, type coercion, binary target, stratified split |
| `test_engineering.py` | Each feature function + `build_features` orchestrator |
| `test_pipeline.py` | ColumnTransformer output shape, SMOTE balance, feature name alignment |
| `test_training_integration.py` | End-to-end: synthetic data → trained models → MLflow runs → saved artefact → reload → predict |

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Roadmap – upcoming stages

| Stage | Description |
|-------|-------------|
| **Stage 2** | **FastAPI** – `/predict` and `/health` HTTP endpoints wrapping the model bundle |
| **Stage 3** | **Docker** – Containerise the API with a multi-stage Dockerfile |
| **Stage 4** | **CI/CD** – GitHub Actions: lint → test → build image → push to ECR |
| **Stage 5** | **AWS** – ECS Fargate deployment with ALB, auto-scaling, and CloudWatch |

Each stage builds directly on the artefacts produced here, so nothing needs to be refactored.

---

## Key design decisions

* **Single config file** – `configs/config.yaml` is the only place to change data paths,
  model params, or business assumptions.
* **Artefact bundle** – saving `(model, preprocessor, feature_names)` together means
  the API/batch jobs need zero extra config to reproduce inference.
* **SMOTE on processed data** – applied *after* the sklearn pipeline is fitted,
  so the test set is never touched by oversampling.
* **Calibrated probabilities** – all three models are wrapped with
  `CalibratedClassifierCV` for reliable churn probability outputs used in
  business simulations.
