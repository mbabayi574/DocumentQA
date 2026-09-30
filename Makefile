# LLM-free extractive document QA. Source of truth for phases: plan.md §7.
UV := uv run

.PHONY: check test run smoke eval calibrate rebuild sync

sync: ## install dependencies (uv.lock is committed)
	uv sync

check: ## gate: ruff lint + format + mypy + pytest
	$(UV) ruff check src tests
	$(UV) ruff format --check src tests
	$(UV) mypy src
	$(UV) pytest

test: ## tests only
	$(UV) pytest

run: ## single worker is required: one process owns data/ and the Chroma dir (plan.md L2)
	$(UV) uvicorn qasystem.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1

smoke: ## add -> query -> edit -> query -> delete -> query (script lands in P8)
	bash scripts/smoke_test.sh

eval: ## evaluation runner (lands in P9)
	$(UV) python -m qasystem.cli eval

calibrate: ## threshold calibration, writes config/thresholds.json (lands in P9)
	$(UV) python -m qasystem.cli calibrate

rebuild: ## re-upsert the dense index from SQLite's embedding cache, 0 API calls (lands in P5)
	$(UV) python -m qasystem.cli rebuild