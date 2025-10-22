SHELL := /bin/bash

# Python version
PYTHON_VERSION := 3.11

# Help target
.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Clean target
.PHONY: clean
clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .ruff_cache/
	@rm -rf .mypy_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf ./runs
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

# Development environment setup
.PHONY: install
install: ## Install package and dependencies
	@echo "Installing package with uv..."
	@uv sync

.PHONY: install-dev
install-dev: ## Install package with development dependencies
	@echo "Installing package with development dependencies..."
	@uv sync --extra dev
	@uv run pre-commit install

.PHONY: update
update: ## Update dependencies
	@echo "Updating dependencies..."
	@uv sync --upgrade

# Code quality
.PHONY: format
format: ## Format code with ruff
	@echo "Formatting code..."
	@uv run ruff format .
	@uv run ruff check --fix .

.PHONY: lint
lint: ## Lint code with ruff
	@echo "Linting code..."
	@uv run ruff check .

.PHONY: type-check
type-check: ## Type check with mypy
	@echo "Type checking..."
	@uv run mypy .

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks
	@echo "Running pre-commit hooks..."
	@uv run pre-commit run --all-files

# Build and distribution
.PHONY: build
build: ## Build package
	@echo "Building package..."
	@uv build

.PHONY: publish-test
publish-test: build ## Publish to test PyPI
	@echo "Publishing to test PyPI..."
	@uv run twine upload --repository testpypi dist/*

.PHONY: publish
publish: build ## Publish to PyPI
	@echo "Publishing to PyPI..."
	@uv run twine upload dist/*

# Documentation
.PHONY: docs
docs: ## Build documentation
	@echo "Building documentation..."
	@echo "Documentation build not implemented yet"

# Development workflow
.PHONY: dev-setup
dev-setup: clean install-dev ## Complete development setup
	@echo "Development environment ready!"

.PHONY: check
check: format lint type-check test ## Run all checks
	@echo "All checks completed!"

functions-api-client: ## requires serving from FunctionsAPI already active
	curl http://localhost:8000/generate-openapi
	npm install @openapitools/openapi-generator-cli -g
	openapi-generator-cli generate \
		-i openapi.json \
		-g python \
		-o ./functions-api-client \
	    --additional-properties=packageName=openapi_client
	sudo chown -R ordonez:ordonez functions-api-client
	uv add ./functions-api-client
