#################################################################################
# GLOBALS                                                                       #
#################################################################################

CONDA_ENV_NAME = ppera-env
SRC_DIR = ppera
PYTHON_INTERPRETER = python
MLFLOW_HOST = 127.0.0.1
MLFLOW_PORT = 8080

# Shell configuration for better error handling
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# Let the Makefile know that these are not actual files to be built
.PHONY: help install setup requirements lint format clean test \
		download-datasets kaggle-setup-help verify-datasets \
		run-all run-interactive run-mlflow stop-mlflow check-env

#################################################################################
# HELP                                                                          #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
lines = '\n'.join([line for line in sys.stdin])
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines)
print('PPERA - Personalization, Privacy and Explainability of Recommendation Algorithms')
print('=' * 80)
print('\nAvailable commands:\n')
for target, description in matches:
	print(f'{target:25} {description}')
print('\nFor more information, visit: https://github.com/your-repo/ppera')
endef
export PRINT_HELP_PYSCRIPT

## Show this help message
help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

#################################################################################
# ENVIRONMENT SETUP                                                            #
#################################################################################

## Create conda environment from environment.yml and install PyTorch
install:
	@echo "=== Setting up PPERA Environment ==="
	@echo "Step 1/3: Creating conda environment '$(CONDA_ENV_NAME)'..."
	@if conda env list | grep -q "^$(CONDA_ENV_NAME) "; then \
		echo "Environment '$(CONDA_ENV_NAME)' already exists. Use 'make requirements' to update."; \
	else \
		conda env create -f environment.yml --name $(CONDA_ENV_NAME); \
	fi
	@echo ""
	@echo "Step 2/3: Installing PyTorch for CPU..."
	@conda run -n $(CONDA_ENV_NAME) pip install torch torchvision torchaudio \
		--index-url https://download.pytorch.org/whl/cpu
	@echo ""
	@echo "Step 3/3: Installing project package..."
	@conda run -n $(CONDA_ENV_NAME) pip install -e .
	@echo ""
	@echo "✅ Environment setup complete!"
	@echo "Next steps:"
	@echo "  1. conda activate $(CONDA_ENV_NAME)"
	@echo "  2. make verify-datasets"
	@echo "  3. make run-all"

## Install project package in editable mode (run after activating environment)
setup:
	@echo ">>> Installing project package '$(SRC_DIR)' in editable mode..."
	@$(PYTHON_INTERPRETER) -m pip install -e .
	@echo "✅ Project package installed successfully!"

## Update conda environment from environment.yml
requirements:
	@echo ">>> Updating conda environment '$(CONDA_ENV_NAME)'..."
	@conda env update --name $(CONDA_ENV_NAME) --file environment.yml --prune
	@echo "✅ Environment updated successfully!"

## Check if conda environment exists and is properly configured
check-env:
	@echo ">>> Checking environment setup..."
	@if ! conda env list | grep -q "^$(CONDA_ENV_NAME) "; then \
		echo "❌ Conda environment '$(CONDA_ENV_NAME)' not found."; \
		echo "Run 'make install' to create it."; \
		exit 1; \
	fi
	@echo "✅ Conda environment '$(CONDA_ENV_NAME)' found."
	@if ! conda run -n $(CONDA_ENV_NAME) python -c "import $(SRC_DIR)" 2>/dev/null; then \
		echo "❌ Project package not installed or importable."; \
		echo "Run 'make setup' or 'conda activate $(CONDA_ENV_NAME) && make setup'"; \
		exit 1; \
	fi
	@echo "✅ Project package is importable."

#################################################################################
# CODE QUALITY                                                                 #
#################################################################################

## Run linting and format checking with ruff
lint:
	@echo ">>> Running code quality checks..."
	@conda run -n $(CONDA_ENV_NAME) ruff check $(SRC_DIR)
	@echo "✅ Linting complete!"

## Format code and apply safe fixes with ruff
format:
	@echo ">>> Formatting code and applying safe fixes..."
	@conda run -n $(CONDA_ENV_NAME) ruff format $(SRC_DIR)
	@conda run -n $(CONDA_ENV_NAME) ruff check $(SRC_DIR) --fix
	@echo "✅ Code formatting complete!"

## Run tests (placeholder for future test implementation)
test:
	@echo ">>> Running tests..."
	@conda run -n $(CONDA_ENV_NAME) python -m pytest tests/ -v || echo "No tests found. Create tests/ directory with test files."

## Clean up compiled Python files and caches
clean:
	@echo ">>> Cleaning up project files..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache build/ dist/
	@echo "✅ Cleanup complete!"

#################################################################################
# DATASET MANAGEMENT                                                           #
#################################################################################

## Download all required datasets from Kaggle
download-datasets: check-env
	@echo ">>> Downloading datasets from Kaggle..."
	@conda run -n $(CONDA_ENV_NAME) --no-capture-output python $(SRC_DIR)/datasets_downloader.py
	@echo "✅ Dataset download complete!"

## Show Kaggle API setup instructions
kaggle-setup-help:
	@echo ">>> Kaggle API Setup Instructions <<<"
	@echo ""
	@echo "1. Create a Kaggle account at https://www.kaggle.com"
	@echo "2. Go to Account Settings (click on your profile picture)"
	@echo "3. Scroll down to 'API' section and click 'Create New API Token'"
	@echo "4. Download the kaggle.json file"
	@echo "5. Place it in one of these locations:"
	@echo "   - ~/.kaggle/kaggle.json (recommended)"
	@echo "   - /path/to/this/project/kaggle.json"
	@echo "6. Set permissions: chmod 600 ~/.kaggle/kaggle.json"
	@echo ""
	@echo "Then run: make download-datasets"

## Verify all datasets are downloaded and complete
verify-datasets:
	@echo ">>> Verifying dataset downloads..."
	@failed=0; \
	total=0; \
	for dataset in AmazonSales MovieLens PostRecommendations; do \
		total=$$((total + 1)); \
		echo -n "Checking datasets/$$dataset/... "; \
		if [ -d "datasets/$$dataset" ] && [ "$$(find datasets/$$dataset -name "*.csv" 2>/dev/null | wc -l)" -gt 0 ]; then \
			echo "✅ OK"; \
		else \
			echo "❌ MISSING"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	if [ $$failed -eq 0 ]; then \
		echo "✅ All $$total datasets verified successfully!"; \
	else \
		echo "❌ $$failed/$$total dataset(s) missing or incomplete."; \
		echo "Solutions:"; \
		echo "  1. Run 'make kaggle-setup-help' for Kaggle API setup"; \
		echo "  2. Run 'make download-datasets' to download missing datasets"; \
		exit 1; \
	fi

#################################################################################
# MLFLOW MANAGEMENT                                                            #
#################################################################################

## Start MLflow server in background
run-mlflow: check-env
	@echo ">>> Starting MLflow server..."
	@if lsof -Pi :$(MLFLOW_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "❌ Port $(MLFLOW_PORT) is already in use. Stop existing MLflow server first."; \
		exit 1; \
	fi
	@conda run -n $(CONDA_ENV_NAME) mlflow server \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns & \
	echo ">>> Waiting for MLflow server to start..." && \
	timeout=30; \
	while [ $$timeout -gt 0 ] && ! curl -s --fail http://$(MLFLOW_HOST):$(MLFLOW_PORT) >/dev/null 2>&1; do \
		sleep 1; \
		timeout=$$((timeout - 1)); \
	done; \
	if [ $$timeout -eq 0 ]; then \
		echo "❌ MLflow server failed to start within 30 seconds"; \
		exit 1; \
	fi; \
	MLFLOW_PID=$$(lsof -t -i:$(MLFLOW_PORT) | head -n 1); \
	echo "✅ MLflow server started (PID: $$MLFLOW_PID)"; \
	echo "🌐 Access MLflow at: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
	echo ""; \
	echo "Press [Enter] to stop the server..."; \
	read dummy < /dev/tty; \
	kill $$MLFLOW_PID && echo "✅ MLflow server stopped"

## Stop any running MLflow servers
stop-mlflow:
	@echo ">>> Stopping MLflow servers..."
	@if MLFLOW_PID=$$(lsof -t -i:$(MLFLOW_PORT) 2>/dev/null | head -n 1); then \
		kill $$MLFLOW_PID && echo "✅ MLflow server (PID: $$MLFLOW_PID) stopped"; \
	else \
		echo "ℹ️ No MLflow server running on port $(MLFLOW_PORT)"; \
	fi

#################################################################################
# MAIN WORKFLOWS                                                               #
#################################################################################

## Run complete pipeline: download data, start MLflow, execute experiments
run-all: check-env verify-datasets
	@echo "=== Running Complete PPERA Pipeline ==="
	@echo ""
	@echo ">>> Starting MLflow server..."
	@if lsof -Pi :$(MLFLOW_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "ℹ️ MLflow server already running on port $(MLFLOW_PORT)"; \
		EXTERNAL_MLFLOW=true; \
	else \
		conda run -n $(CONDA_ENV_NAME) mlflow server \
			--host $(MLFLOW_HOST) \
			--port $(MLFLOW_PORT) \
			--backend-store-uri sqlite:///mlflow.db \
			--default-artifact-root ./mlruns & \
		echo ">>> Waiting for MLflow server to start..."; \
		timeout=30; \
		while [ $$timeout -gt 0 ] && ! curl -s --fail http://$(MLFLOW_HOST):$(MLFLOW_PORT) >/dev/null 2>&1; do \
			sleep 1; \
			timeout=$$((timeout - 1)); \
		done; \
		if [ $$timeout -eq 0 ]; then \
			echo "❌ MLflow server failed to start"; \
			exit 1; \
		fi; \
		MLFLOW_PID=$$(lsof -t -i:$(MLFLOW_PORT) | head -n 1); \
		trap "echo '>>> Shutting down MLflow server...'; kill $$MLFLOW_PID 2>/dev/null || true" EXIT; \
		echo "✅ MLflow server started (PID: $$MLFLOW_PID)"; \
	fi; \
	echo "🌐 MLflow UI: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
	echo ""; \
	echo ">>> Running main experiments..."; \
	conda run -n $(CONDA_ENV_NAME) --no-capture-output $(PYTHON_INTERPRETER) -u -m $(SRC_DIR).main; \
	echo ""; \
	echo "✅ Pipeline execution complete!"

## Run pipeline interactively (MLflow stays open for inspection)
run-interactive: check-env verify-datasets
	@echo "=== Running PPERA Pipeline (Interactive Mode) ==="
	@echo ""
	@echo ">>> Starting MLflow server..."
	@conda run -n $(CONDA_ENV_NAME) mlflow server \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns & \
	echo ">>> Waiting for MLflow server to start..." && \
	timeout=30; \
	while [ $$timeout -gt 0 ] && ! curl -s --fail http://$(MLFLOW_HOST):$(MLFLOW_PORT) >/dev/null 2>&1; do \
		sleep 1; \
		timeout=$$((timeout - 1)); \
	done; \
	if [ $$timeout -eq 0 ]; then \
		echo "❌ MLflow server failed to start"; \
		exit 1; \
	fi; \
	MLFLOW_PID=$$(lsof -t -i:$(MLFLOW_PORT) | head -n 1); \
	echo "✅ MLflow server started (PID: $$MLFLOW_PID)"; \
	echo "🌐 Access MLflow at: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
	echo ""; \
	echo ">>> Running main experiments..."; \
	conda run -n $(CONDA_ENV_NAME) --no-capture-output $(PYTHON_INTERPRETER) -u -m $(SRC_DIR).main; \
	echo ""; \
	echo "✅ Experiments complete! MLflow server is still running."; \
	echo "🔍 Review results at: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
	echo ""; \
	echo "Press [Enter] to stop MLflow server and exit..."; \
	read dummy < /dev/tty; \
	kill $$MLFLOW_PID && echo "✅ MLflow server stopped"

## Quick start: install environment and run complete pipeline
quickstart:
	@echo "=== PPERA Quick Start ==="
	@echo "This will set up everything and run the complete pipeline."
	@echo ""
	@make install
	@echo ""
	@make download-datasets
	@echo ""
	@make run-all

#################################################################################
# DEVELOPMENT HELPERS                                                          #
#################################################################################

## Show project status and diagnostics
status:
	@echo "=== PPERA Project Status ==="
	@echo ""
	@echo "Environment:"
	@if conda env list | grep -q "^$(CONDA_ENV_NAME) "; then \
		echo "  ✅ Conda environment: $(CONDA_ENV_NAME)"; \
	else \
		echo "  ❌ Conda environment: $(CONDA_ENV_NAME) (not found)"; \
	fi
	@echo ""
	@echo "Datasets:"
	@for dataset in AmazonSales MovieLens PostRecommendations; do \
		if [ -d "datasets/$$dataset" ] && [ "$$(find datasets/$$dataset -name "*.csv" 2>/dev/null | wc -l)" -gt 0 ]; then \
			echo "  ✅ $$dataset"; \
		else \
			echo "  ❌ $$dataset (missing or empty)"; \
		fi; \
	done
	@echo ""
	@echo "Services:"
	@if lsof -Pi :$(MLFLOW_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "  ✅ MLflow server (running on port $(MLFLOW_PORT))"; \
	else \
		echo "  ⭕ MLflow server (not running)"; \
	fi

## Remove conda environment completely
uninstall:
	@echo ">>> Removing conda environment '$(CONDA_ENV_NAME)'..."
	@conda env remove -n $(CONDA_ENV_NAME) --yes || echo "Environment not found or already removed"
	@echo "✅ Environment removed successfully!"

## Full cleanup: remove environment, datasets, and generated files
reset: uninstall clean
	@echo ">>> Performing full project reset..."
	@rm -rf datasets/ mlruns/ mlflow.db ppera/rl_tmp/
	@echo "✅ Full reset complete! Run 'make quickstart' to set up again."
