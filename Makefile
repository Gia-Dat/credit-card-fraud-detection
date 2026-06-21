.PHONY: lint train infra-up infra-down clean

# 1. Run automated code quality checks
lint:
	uv run pre-commit run --all-files

# 2. Trigger the machine learning training pipeline
train:
	uv run src/train.py

# Test suite validation
test:
	uv run pytest -v

# 3. Spin up local cloud infrastructure via Terraform
infra-up:
	cd terraform && terraform apply -auto-approve

# 4. Tear down local cloud resources cleanly
infra-down:
	cd terraform && terraform destroy -auto-approve

# 5. Clean up temporary cache and runtime directories
clean:
	rm -rf .pytest_cache .config .ruff_cache htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +

serve:
	uv run uvicorn src.app:app --reload --host 127.0.0.1 --port 8000

docker-build:
	docker build -t fraud-api:latest .

docker-run:
	docker run -d -p 8000:8000 --name transaction-api-server fraud-api:latest

serve-monitor:
	uv run uvicorn src.monitor:monitor_app --reload --host 127.0.0.1 --port 8050

# Pull and launch a pre-configured Grafana monitoring panel dashboard
grafana-up:
	docker run -d -p 3000:3000 --name telemetry-grafana grafana/grafana:latest

prometheus-up:
	docker run -d -p 9090:9090 -v "$(PWD)/prometheus.yml:/etc/prometheus/prometheus.yml" --name telemetry-prometheus prom/prometheus:latest

# Launch Grafana for live visualization panels
grafana-up:
	docker run -d -p 3000:3000 --name telemetry-grafana grafana/grafana:latest