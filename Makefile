#################################################################################
# GLOBALS                                                                       #
#################################################################################

CONDA_ENV_NAME = qpera-env
SRC_DIR = qpera
PYTHON_INTERPRETER = python
MLFLOW_HOST = 127.0.0.1
MLFLOW_PORT = 8080
DOCS_PORT ?= 8000

# Shell configuration for better error handling
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# Let the Makefile know that these are not actual files to be built
.PHONY: help install install-dev install-full setup requirements lint format clean \
		download-datasets kaggle-setup-help verify-datasets \
		run-all run-interactive run-mlflow stop-mlflow check-env \
		status uninstall reset quickstart docs

#################################################################################
# HELP                                                                          #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
lines = '\n'.join([line for line in sys.stdin])
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines)
print('QPERA - Quality of Personalization, Explainability and Robustness of Recommendation Algorithms')
print('=' * 80)
print('\nAvailable commands:\n')
for target, description in matches:
	print(f'{target:25} {description}')
print('\nFor more information, visit: https://github.com/PUT-RecSys-Research/qpera-thesis')
endef
export PRINT_HELP_PYSCRIPT

## Show this help message
help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

#################################################################################
# ENVIRONMENT SETUP                                                            #
#################################################################################

## Create conda environment and install core dependencies
install:
	@echo "=== Setting up QPERA Environment (Core) ==="
	@echo "Step 1/3: Creating conda environment '$(CONDA_ENV_NAME)'..."
	@if conda env list | grep -q "^$(CONDA_ENV_NAME) "; then \
		echo "Environment '$(CONDA_ENV_NAME)' already exists. Use 'make requirements' to update."; \
	else \
		conda env create -f environment.yml; \
	fi
	@echo ""
	@echo "Step 2/3: Installing PyTorch (CPU version)..."
	@conda run -n $(CONDA_ENV_NAME) pip install torch torchvision torchaudio \
		--index-url https://download.pytorch.org/whl/cpu
	@echo ""
	@echo "Step 3/3: Installing project package with core dependencies..."
	@conda run -n $(CONDA_ENV_NAME) pip install -e .
	@echo ""
	@echo "✅ Core environment setup complete!"
	@echo "💡 For development tools, run: make install-dev"
	@echo "💡 For all ML extras, run: make install-full"

## Install with development tools (recommended for contributors)
install-dev:
	@echo "=== Setting up PPERA Environment (Development) ==="
	@make install
	@echo ""
	@echo "Installing development tools..."
	@conda run -n $(CONDA_ENV_NAME) pip install -e ".[dev]"
	@echo ""
	@echo "✅ Development environment setup complete!"
	@echo "Available tools: pytest, ruff, black, jupyter, etc."
	@echo "Next steps:"
	@echo "  1. conda activate $(CONDA_ENV_NAME)"
	@echo "  2. make verify-datasets"
	@echo "  3. make run-all"

## Install with all optional dependencies (full ML stack)
install-full:
	@echo "=== Setting up PPERA Environment (Full ML Stack) ==="
	@make install
	@echo ""
	@echo "Installing all optional dependencies..."
	@conda run -n $(CONDA_ENV_NAME) pip install -e ".[all]"
	@echo ""
	@echo "✅ Full environment setup complete!"
	@echo "Includes: development tools + ML extras + all dependencies"

## Install project package in editable mode (run after activating environment)
setup:
	@echo ">>> Installing project package '$(SRC_DIR)' in editable mode..."
	@$(PYTHON_INTERPRETER) -m pip install -e .
	@echo "✅ Project package installed successfully!"

## Install with development extras (when environment is active)
setup-dev:
	@echo ">>> Installing project package with development tools..."
	@$(PYTHON_INTERPRETER) -m pip install -e ".[dev]"
	@echo "✅ Project package with dev tools installed successfully!"

## Update conda environment from environment.yml
requirements:
	@echo ">>> Updating conda environment '$(CONDA_ENV_NAME)'..."
	@conda env update --name $(CONDA_ENV_NAME) --file environment.yml --prune
	@echo ">>> Updating project dependencies..."
	@conda run -n $(CONDA_ENV_NAME) pip install -e . --upgrade
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

## Run linting with relaxed rules for ML code
lint:
	@echo ">>> Running code quality checks..."
	@if conda run -n $(CONDA_ENV_NAME) python -c "import ruff" 2>/dev/null; then \
		conda run -n $(CONDA_ENV_NAME) ruff check $(SRC_DIR) --fix-only --show-fixes; \
	else \
		echo "❌ Ruff not installed. Run 'make install-dev' to install development tools."; \
		exit 1; \
	fi
	@echo "✅ Linting complete!"

## Format code with conservative settings
format:
	@echo ">>> Formatting code..."
	@if conda run -n $(CONDA_ENV_NAME) python -c "import ruff" 2>/dev/null; then \
		conda run -n $(CONDA_ENV_NAME) ruff format $(SRC_DIR); \
		conda run -n $(CONDA_ENV_NAME) ruff check $(SRC_DIR) --fix-only --select=F,E9,W6; \
	else \
		echo "❌ Ruff not installed. Run 'make install-dev' to install development tools."; \
		exit 1; \
	fi
	@echo "✅ Code formatting complete!"

## Strict linting for CI/final review (optional)
lint-strict:
	@echo ">>> Running strict code quality checks..."
	@conda run -n $(CONDA_ENV_NAME) ruff check $(SRC_DIR)
	@echo "✅ Strict linting complete!"

## Clean up compiled Python files and caches
clean:
	@echo ">>> Cleaning up project files..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache build/ dist/ .coverage htmlcov/
	@echo "✅ Cleanup complete!"

#################################################################################
# DATASET MANAGEMENT                                                           #
#################################################################################

## Download all required datasets from Kaggle
download-datasets: check-env
	@echo ">>> Downloading datasets from Kaggle..."
	@if ! conda run -n $(CONDA_ENV_NAME) python -c "import kaggle" 2>/dev/null; then \
		echo "❌ Kaggle CLI not found. Installing..."; \
		conda run -n $(CONDA_ENV_NAME) pip install kaggle; \
	fi
	@conda run -n $(CONDA_ENV_NAME) $(PYTHON_INTERPRETER) $(SRC_DIR)/datasets_downloader.py
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

## Kaggle API automatic configuration
kaggle-autoconfig:
	@echo ">>> Kaggle API Autoconfig <<<"
	@echo ""
	@echo "1. Looking for kaggle.json in ~/Downloads..."
	@if [ -f ~/Downloads/kaggle.json ]; then \
		echo "2. Creating ~/.kaggle directory..."; \
		mkdir -p ~/.kaggle; \
		echo "3. Moving kaggle.json to ~/.kaggle/..."; \
		mv ~/Downloads/kaggle.json ~/.kaggle/; \
		echo "4. Setting permissions to 600..."; \
		chmod 600 ~/.kaggle/kaggle.json; \
		echo ""; \
		echo "✅ Kaggle API has been successfully configured!"; \
	else \
		echo "❌ kaggle.json not found in ~/Downloads. Please download it from Kaggle first."; \
	fi

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
	@if ! conda run -n $(CONDA_ENV_NAME) python -c "import mlflow" 2>/dev/null; then \
		echo "❌ MLflow not installed. Installing..."; \
		conda run -n $(CONDA_ENV_NAME) pip install mlflow; \
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
	@echo "=== Running Complete QPERA Pipeline ==="
	@echo ""
	@echo ">>> Starting MLflow server..."
	@if ! conda run -n $(CONDA_ENV_NAME) python -c "import mlflow" 2>/dev/null; then \
		echo "Installing MLflow..."; \
		conda run -n $(CONDA_ENV_NAME) pip install mlflow; \
	fi
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
	@echo "=== Running QPERA Pipeline (Interactive Mode) ==="
	@echo ""
	@echo ">>> Starting MLflow server..."
	@if ! conda run -n $(CONDA_ENV_NAME) python -c "import mlflow" 2>/dev/null; then \
		echo "Installing MLflow..."; \
		conda run -n $(CONDA_ENV_NAME) pip install mlflow; \
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
	@echo "=== QPERA Quick Start ==="
	@echo "This will set up everything and run the complete pipeline."
	@echo ""
	@make install-dev
	@echo ""
	@make download-datasets
	@echo ""
	@make run-all

#################################################################################
# DOCUMENTATION                                                                #
#################################################################################

## Install documentation dependencies
install-docs:
	@echo ">>> Installing documentation dependencies..."
	@conda run -n $(CONDA_ENV_NAME) pip install -e ".[docs]"
	@echo "✅ Documentation dependencies installed."

## Build and serve the documentation locally
docs: check-env
	@echo ">>> Building and serving documentation locally..."
	@conda run -n $(CONDA_ENV_NAME) mkdocs serve --dev-addr=127.0.0.1:$(DOCS_PORT) & \
	echo ">>> Waiting for MkDocs server to start..." && \
	timeout=30; \
	while [ $$timeout -gt 0 ] && ! curl -s --fail http://127.0.0.1:$(DOCS_PORT) >/dev/null 2>&1; do \
		sleep 1; \
		timeout=$$((timeout - 1)); \
	done; \
	if [ $$timeout -eq 0 ]; then \
		echo "❌ MkDocs server failed to start"; \
		exit 1; \
	fi; \
	DOCS_PID=$$(lsof -t -i:$(DOCS_PORT) | head -n 1); \
	echo "✅ MkDocs server started (PID: $$DOCS_PID)"; \
	echo "🌐 Access documentation at: http://127.0.0.1:$(DOCS_PORT)"; \
	echo ""; \
	echo "Press [Enter] to stop the server and exit..."; \
	read dummy < /dev/tty; \
	kill $$DOCS_PID && echo "✅ MkDocs server stopped" \

#################################################################################
# DEVELOPMENT HELPERS                                                          #
#################################################################################

## Show project status and diagnostics
status:
	@echo "=== QPERA Project Status ==="
	@echo ""
	@echo "Environment:"
	@if conda env list | grep -q "^$(CONDA_ENV_NAME) "; then \
		echo "  ✅ Conda environment: $(CONDA_ENV_NAME)"; \
		echo -n "  🔧 Dev tools: "; \
		if conda run -n $(CONDA_ENV_NAME) python -c "import ruff, pytest" 2>/dev/null; then \
			echo "✅ installed"; \
		else \
			echo "❌ missing (run 'make install-dev')"; \
		fi; \
		echo -n "  📊 MLflow: "; \
		if conda run -n $(CONDA_ENV_NAME) python -c "import mlflow" 2>/dev/null; then \
			echo "✅ installed"; \
		else \
			echo "⭕ not installed"; \
		fi; \
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
	@echo "Cleaning dataset contents (preserving folder structure)..."
	@for dataset in AmazonSales MovieLens PostRecommendations; do \
		if [ -d "datasets/$$dataset" ]; then \
			echo "  Cleaning datasets/$$dataset/..."; \
			find "datasets/$$dataset" -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true; \
		fi; \
	done
	@echo "Cleaning qpera directories..."
	@if [ -d "qpera/datasets" ]; then \
		echo "  Cleaning qpera/datasets/..."; \
		find qpera/datasets -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true; \
	fi
	@if [ -d "qpera/metrics" ]; then \
		echo "  Cleaning qpera/metrics/ (removing subfolders)..."; \
		find qpera/metrics -mindepth 1 -type d ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true; \
		find qpera/metrics -mindepth 1 -type f ! -name '.gitkeep' -delete 2>/dev/null || true; \
	fi
	@if [ -d "qpera/plots" ]; then \
		echo "  Cleaning qpera/plots/..."; \
		find qpera/plots -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true; \
	fi
	@echo "Removing ML artifacts and logs..."
	@rm -rf mlruns/ mlartifacts/ mlflow.db qpera/rl_tmp/ experiment_runner.log
	@echo "✅ Full reset complete! Run 'make quickstart' to set up again."
