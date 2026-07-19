# Makefile for Adaptive Portfolio Analytics Platform

.PHONY: help install install-dev verify test test-cov lint format format-check clean run-dashboard smoke

help:
	@echo "Adaptive Portfolio Analytics Platform"
	@echo "===================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install with development dependencies"
	@echo "  make verify        Verify structure and core dependencies"
	@echo "  make test          Run the test suite"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run Ruff lint checks"
	@echo "  make format        Format Python files with Ruff"
	@echo "  make format-check  Check Ruff formatting"
	@echo "  make clean         Remove build artifacts and cache"
	@echo "  make run-dashboard Run Streamlit dashboard"
	@echo "  make smoke         Run the final release smoke test"

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements.txt
	python -m pip install -e ".[dev,docs]"

verify:
	python verify_setup.py

test:
	python -m pytest -q

test-cov:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint:
	python -m ruff check .

format:
	python -m ruff format .

format-check:
	python -m ruff format --check .

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.mypy_cache' -delete
	find . -type d -name '*.egg-info' -delete
	rm -rf build/ dist/ htmlcov/

run-dashboard:
	streamlit run src/dashboard/app.py

smoke:
	python scripts/final_smoke_test.py

.DEFAULT_GOAL := help
