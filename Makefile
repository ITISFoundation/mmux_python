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
# the apt install might need sudo? But that is only for latex in Matplotlib so not critical, could be removed.
pyenv: clean
	@apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
	@python -m venv $(VENV_DIR)
	@. ./$(VENV_DIR)/bin/activate && pip install -r requirements.txt

.PHONY: pyenv-dev
pyenv-dev: pyenv
	@. ./$(VENV_DIR)/bin/activate && pip install -r requirements-dev.txt
	@. ./$(VENV_DIR)/bin/activate && pre-commit install
# in the future, for deployment, remove requirements_dev and pre-commit hooks
