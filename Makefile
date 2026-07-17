PY ?= python3

install:
	pip install -e ".[dev]"

generate-data:
	$(PY) scripts/generate_data.py

generate-data-small:
	$(PY) scripts/generate_data.py --small

init-db:
	$(PY) scripts/initialize_database.py

features:
	$(PY) scripts/build_features.py

train:
	$(PY) scripts/train_all_models.py

evaluate:
	$(PY) scripts/evaluate_all_models.py

pipeline:
	$(PY) scripts/run_pipeline.py

pipeline-small:
	$(PY) scripts/run_pipeline.py --small

api:
	uvicorn driveintent.api.main:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/driveintent/dashboard/app.py --server.port 8501

test:
	pytest -q

lint:
	ruff check src scripts tests

format:
	black src scripts tests

smoke-test:
	$(PY) scripts/smoke_test.py

clean:
	rm -rf data/raw/*.parquet data/processed/*.parquet data/database/*.duckdb artifacts/*/*

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
