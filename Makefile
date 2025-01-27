SHELL := /bin/bash

# Directory for virtual environment
VENV_DIR := ".pyenv"

# Clean target
.PHONY: clean
clean:
	@rm -rf $(VENV_DIR)
	@rm -rf ./runs

# Virtual environment target
.PHONY: pyenv
pyenv: clean
	@python -m venv $(VENV_DIR)
	@. ./$(VENV_DIR)/bin/activate && pip install -r requirements.txt
	@. ./$(VENV_DIR)/bin/activate && pip install -r requirements-dev.txt
	@. ./$(VENV_DIR)/bin/activate && pre-commit install
# in the future, for deployment, remove requirements_dev and pre-commit hooks
