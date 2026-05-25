# Makefile for Adaptive Portfolio Analytics Platform

.PHONY: help install install-dev test test-cov lint format clean run-dashboard docs

help:
	@echo "Adaptive Portfolio Analytics Platform"
	@echo "===================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install with development dependencies"
	@echo "  make test          Run unit tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run code linting (flake8, mypy)"
	@echo "  make format        Format code (black, isort)"
	@echo "  make clean         Remove build artifacts and cache"
	@echo "  make run-dashboard Run Streamlit dashboard"
	@echo "  make docs          Build documentation"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,docs]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src tests --max-line-length=100
	mypy src --ignore-missing-imports

format:
	black src tests
	isort src tests

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.mypy_cache' -delete
	find . -type d -name '*.egg-info' -delete
	rm -rf build/ dist/ htmlcov/

run-dashboard:
	streamlit run src/dashboard/app.py

docs:
	cd docs && make html

.DEFAULT_GOAL := help
