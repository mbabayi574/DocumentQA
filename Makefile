# LLM-free extractive document QA. Source of truth for phases: plan.md §7.
UV := uv run

.PHONY: check check-live live live-eval test run sync

sync: ## install dependencies (uv.lock is committed)
	uv sync

check: ## gate: ruff lint + format + mypy + pytest
	$(UV) ruff check src tests
	$(UV) ruff format --check src tests
	$(UV) mypy src
	$(UV) pytest

live: ## the same tests against the REAL provider. Every phase runs this, not just `check`.
	RUN_LIVE=1 $(UV) pytest tests/integration -q

check-live: check live ## offline gate and live gate: the full definition of done

live-eval: ## run the eval corpus against the real provider (P9)
	RUN_LIVE=1 $(UV) python -m qasystem.cli eval

test: ## tests only
	$(UV) pytest

run: ## single worker is required: one process owns data/ and the Chroma dir (plan.md L2)
	$(UV) uvicorn qasystem.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1
