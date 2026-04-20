# ──────────────────────────────────────────────────────────────────────────
# Churn Predictor – developer shortcuts
# ──────────────────────────────────────────────────────────────────────────
.PHONY: install train predict test lint mlflow-ui clean

install:
	pip install -r requirements.txt

train:
	python scripts/train.py --config configs/config.yaml

predict:
	python scripts/predict.py \
		--model outputs/best_model.pkl \
		--input $(INPUT) \
		--output outputs/predictions.csv

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf outputs/ .pytest_cache/ .ruff_cache/
