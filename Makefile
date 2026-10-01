# LLM-free extractive document QA. Source of truth for phases: plan.md §7.
UV := uv run

.PHONY: check test run sync

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
