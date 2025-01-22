SHELL := /bin/bash

# Directory for virtual environment
VENV_DIR := ".pyenv"

# Clean target
.PHONY: clean
clean:
	@rm -rf $(VENV_DIR)
	
# Virtual environment target
.PHONY: pyenv
pyenv: clean
	@python -m venv $(VENV_DIR)
	@. ./$(VENV_DIR)/bin/activate && pip install -r requirements.txt