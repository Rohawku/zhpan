# zhpan Makefile

.PHONY: help install dev-install build-prompts generate judge analyze benchmark demo budget clean lint format test

help:
	@echo "zhpan commands:"
	@echo ""
	@echo "  make install        — install runtime deps via uv"
	@echo "  make dev-install    — install runtime + dev deps"
	@echo ""
	@echo "  make build-prompts  — write curated 40-item Chinese prompt set"
	@echo "  make generate       — run generation pipeline (configs/v0.1.yaml)"
	@echo "  make judge          — run judge pipeline on latest generations"
	@echo "  make analyze        — compute bias matrix + leaderboard JSON"
	@echo "  make benchmark      — generate + judge + analyze (real APIs, ~\$$30)"
	@echo ""
	@echo "  make demo           — full pipeline using mock vendors (offline, no \$$)"
	@echo "  make budget         — dry-run estimate of generation cost"
	@echo ""
	@echo "  make lint           — ruff check"
	@echo "  make format         — ruff format"
	@echo "  make test           — pytest"
	@echo "  make clean          — remove cache, .pyc, build artifacts"

install:
	uv sync

dev-install:
	uv sync --extra dev

build-prompts:
	uv run python -m zhpan.scripts.build_alignbench --n 150 --out data/prompts/v0.3.jsonl

generate:
	uv run python -m zhpan.scripts.run_generate --config configs/v0.3.yaml

judge:
	uv run python -m zhpan.scripts.run_judge --config configs/v0.3.yaml

analyze:
	uv run python -m zhpan.scripts.analyze --config configs/v0.3.yaml

benchmark: build-prompts generate judge analyze

# ─── Demo (offline, mock vendors) ──────────────────────────
demo:
	@uv run python -m zhpan.scripts.build_prompts --out data/prompts/demo.jsonl
	@uv run python -m zhpan.scripts.run_generate --config configs/demo.yaml
	@uv run python -m zhpan.scripts.run_judge --config configs/demo.yaml
	@uv run python -m zhpan.scripts.analyze --config configs/demo.yaml

budget:
	uv run python -m zhpan.scripts.run_generate --config configs/v0.3.yaml --dry-run

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

test:
	uv run pytest

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
