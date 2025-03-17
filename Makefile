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

.PHONY: pyenv-dev
pyenv-dev: pyenv
	@. ./$(VENV_DIR)/bin/activate && pip install -r requirements-dev.txt
	@. ./$(VENV_DIR)/bin/activate && pre-commit install
# in the future, for deployment, remove requirements_dev and pre-commit hooks

functions-api-client: ## requires serving from FunctionsAPI already active
	curl http://localhost:8000/generate-openapi
	npm install @openapitools/openapi-generator-cli -g
	openapi-generator-cli generate \
		-i openapi.json \
		-g python \
		-o ./functions-api-client \
	    --additional-properties=packageName=openapi_client
	sudo chown -R ordonez:ordonez functions-api-client
	@. ./$(VENV_DIR)/bin/activate && pip install ./functions-api-client
