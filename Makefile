#################################################################################
# GLOBALS                                                                       #
#################################################################################

CONDA_ENV_NAME = ppera-env
SRC_DIR = ppera
PYTHON_INTERPRETER = python
MLFLOW_HOST = 127.0.0.1
MLFLOW_PORT = 8080

# Let the Makefile know that these are not actual files to be built
.PHONY: help install setup requirements lint format clean run-all

#################################################################################
# SETUP COMMANDS                                                                #
#################################################################################

## Create the conda environment from the environment.yml file
# In your Makefile...

## Create the conda environment AND install special PyTorch dependencies
install:
	@echo "--- Step 1/2: Creating conda environment '$(CONDA_ENV_NAME)' from environment.yml..."
	@conda env create -f environment.yml --name $(CONDA_ENV_NAME)
	@echo ""
	@echo "--- Step 2/2: Installing PyTorch for CPU from special index..."
	@conda run -n $(CONDA_ENV_NAME) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	@echo ""
	@echo ">>> Environment created successfully."
	@echo ">>> Recommended next steps:"
	@echo "    conda activate $(CONDA_ENV_NAME)"
	@echo "    make setup"

## Install the project package in editable mode (CRUCIAL STEP)
setup:
	@echo ">>> Installing project package '$(SRC_DIR)' in editable mode..."
	# This command assumes you have activated the conda environment
	$(PYTHON_INTERPRETER) -m pip install -e .

## Update python dependencies from environment.yml for an existing environment
requirements:
	@echo ">>> Updating conda environment '$(CONDA_ENV_NAME)'..."
	conda env update --name $(CONDA_ENV_NAME) --file environment.yml --prune

#################################################################################
# DEVELOPMENT COMMANDS                                                          #
#################################################################################

## Lint and check formatting with ruff
lint:
	@echo ">>> Linting and checking format with ruff..."
	ruff check $(SRC_DIR)

## Format the code AND apply safe fixes with ruff
format:
	@echo ">>> Formatting code and applying safe fixes with ruff..."
	ruff format $(SRC_DIR)
	ruff check $(SRC_DIR) --fix

## Delete all compiled Python files and caches
clean:
	@echo ">>> Cleaning up project..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .ruff_cache

#################################################################################
# DATASET MANAGEMENT                                                            #
#################################################################################

## Download all datasets from Kaggle
download-datasets:
	@echo "Downloading datasets from Kaggle..."
	@conda run -n $(CONDA_ENV_NAME) --no-capture-output python ppera/datasets_downloader.py

## Show Kaggle authentication setup instructions
kaggle-setup-help:
	@conda run -n $(CONDA_ENV_NAME) --no-capture-output python ppera/datasets_downloader.py --setup-help

## Verify all datasets are downloaded correctly
verify-datasets:
	@echo "Verifying dataset downloads..."
	@failed=0; \
	for dataset in AmazonSales MovieLens PostRecommendations; do \
		echo -n "Checking datasets/$$dataset/... "; \
		if [ -d "datasets/$$dataset" ] && [ "$$(ls -A datasets/$$dataset 2>/dev/null | wc -l)" -gt 0 ]; then \
			echo "OK"; \
		else \
			echo "MISSING"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	if [ $$failed -eq 0 ]; then \
		echo "All datasets verified successfully!"; \
	else \
		echo "$$failed dataset(s) missing or empty. Run 'make download-datasets' to fix."; \
		exit 1; \
	fi

#################################################################################
# WORKFLOWS                                                                     #
#################################################################################

## Run the entire pipeline: start MLflow, wait, run main script, and cleanup
run-all:
	@echo "--- Running the entire pipeline..." ; \
	echo "Downloading datasets from Kaggle..." ; \
	conda run -n $(CONDA_ENV_NAME) --no-capture-output python ppera/datasets_downloader.py ; \
	echo "--- Starting MLflow server in the background..." ; \
	conda run -n $(CONDA_ENV_NAME) mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) & \
	echo "--- Waiting for MLflow server to be ready..." ; \
	while ! curl -s --fail http://$(MLFLOW_HOST):$(MLFLOW_PORT) > /dev/null; do \
		sleep 0.5; \
	done ; \
	MLFLOW_PID=$$(lsof -t -i:$(MLFLOW_PORT) | head -n 1) ; \
	trap 'echo "--- Shutting down MLflow server (PID: $$MLFLOW_PID)..."; kill $$MLFLOW_PID' EXIT ; \
	echo "--- MLflow server is up (PID: $$MLFLOW_PID). Cleanup trap is set." ; \
	echo "--- Running the main script..." ; \
	conda run -n $(CONDA_ENV_NAME) --no-capture-output $(PYTHON_INTERPRETER) -u -m $(SRC_DIR).main ; \
	echo "--- Main script finished."

## Run the entire pipeline and PAUSE before shutting down the MLflow server
run-interactive:
	@echo "--- Starting MLflow server in the background..." ; \
	conda run -n $(CONDA_ENV_NAME) mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) & \
	echo "--- Waiting for MLflow server to be ready..." ; \
	while ! curl -s --fail http://$(MLFLOW_HOST):$(MLFLOW_PORT) > /dev/null; do \
		sleep 0.5; \
	done ; \
	MLFLOW_PID=$$(lsof -t -i:$(MLFLOW_PORT) | head -n 1) ; \
	echo "--- MLflow server is up (PID: $$MLFLOW_PID). You can view it at http://$(MLFLOW_HOST):$(MLFLOW_PORT)" ; \
	echo "--- Running the main script..." ; \
	conda run -n $(CONDA_ENV_NAME) --no-capture-output $(PYTHON_INTERPRETER) -u -m $(SRC_DIR).main ; \
	echo "" ; \
	echo "--- Main script finished. The MLflow server is still running." ; \
	echo -n "--- Press [Enter] to shut down the MLflow server..." ; \
	read dummy < /dev/tty ; \
	kill $$MLFLOW_PID

# Run mlflow server in the background
run-mlflow:
	@echo "--- Starting MLflow server in the background..." ; \
	conda run -n $(CONDA_ENV_NAME) mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) & \
	echo "--- Waiting for MLflow server to be ready..." ; \
	while ! curl -s --fail http://$(MLFLOW_HOST):$(MLFLOW_PORT) > /dev/null; do \
		sleep 0.5; \
	done ; \
	MLFLOW_PID=$$(lsof -t -i:$(MLFLOW_PORT) | head -n 1) ; \
	echo "--- MLflow server is up (PID: $$MLFLOW_PID). You can view it at http://$(MLFLOW_HOST):$(MLFLOW_PORT)" ; \
	echo -n "--- Press [Enter] to shut down the MLflow server..." ; \
	read dummy < /dev/tty ; \
	kill $$MLFLOW_PID



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
