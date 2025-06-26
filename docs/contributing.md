# Development Workflow

This document provides a practical guide for the development workflow used in the QPERA project. Its purpose is to ensure consistency and reproducibility for the authors and any future researchers building on this work.

---

## 1. Development Environment Setup

### Prerequisites
- Python 3.9+
- Conda/Miniconda
- Git

### Installation Steps

1.  **Fork and Clone the Repository**
    ```bash
    # Fork the repository on GitHub, then clone your fork
    git clone https://github.com/YOUR-USERNAME/qpera-thesis.git
    cd qpera-thesis

    # Add the original repository as the "upstream" remote
    git remote add upstream https://github.com/PUT-RecSys-Research/qpera-thesis.git
    ```

2.  **Create and Activate the Environment**
    ```bash
    # Create the conda environment from the environment.yml file
    make install
    conda activate qpera-env
    ```

3.  **Install the Project Package**
    ```bash
    # Install the qpera package in editable mode for development
    make setup
    ```

---

## 2. Core Development Workflow

### Branching
All changes should be made in a feature branch to keep the `main` branch stable.
```bash
git checkout -b feature/your-descriptive-branch-name
```

### Code Quality
We use `ruff` for fast linting and formatting.
- **Check for issues**: `make lint`
- **Automatically format code**: `make format`

### Testing
Before committing, run a small-scale experiment to ensure your changes haven't broken the pipeline.
```bash
# Run a quick test on the MovieLens dataset
python -m qpera.main --algo CF --dataset movielens --rows 1000
```

---

## 3. Extending the Project

This section provides a high-level overview of how to add new components.

### Adding a New Algorithm
1.  Create a new file (e.g., `qpera/NEW_ALGORITHM.py`) with an experiment loop function that matches the signature of existing algorithms (like `cf_experiment_loop`).
2.  Implement the data loading, training, prediction, and evaluation logic.
3.  Register the new algorithm in the `EXPERIMENT_CONFIGS` list in `qpera/main.py`.

### Adding a New Metric
1.  Add the metric calculation function to `qpera/metrics.py`.
2.  Integrate the new metric into the evaluation section of the relevant algorithm loops (e.g., in `qpera/CF.py`).

### Adding a New Dataset
1.  Create a new loader class in `qpera/datasets_loader.py` that inherits from `BaseDatasetLoader`.
2.  Implement the `merge_datasets` method for your specific data source.
3.  Register the new loader in the `dataset_loaders` dictionary within the `loader` function.

---

## 4. Documentation

The documentation is built with MkDocs. To preview your changes locally:
```bash
# Serve the documentation at http://127.0.0.1:8000
mkdocs serve
```

---

## 5. Contact

For questions about the project, please contact the authors:
- [Julia Podsadna](https://github.com/GambiBambi) 
- [Bartosz Chwiłkowski](https://github.com/kooogi)
- **Supervisor:** [Prof. Mikołaj Morzy](https://github.com/megaduks)