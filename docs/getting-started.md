# Getting Started

This guide will help you set up and run your first experiment with the PPERA framework.

## Prerequisites

- **Python 3.8+** (3.9 recommended)
- **Conda** or **Miniconda**
- **Git**
- **Kaggle account** (for dataset downloads)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/personalization-privacy-and-explainability-of-recommendation-algorithms.git
cd personalization-privacy-and-explainability-of-recommendation-algorithms
```

### 2. Environment Setup

```bash
# Create and activate conda environment
make install
conda activate ppera-env

# Install dependencies and setup project
make setup
```

### 3. Verify Installation

```bash
# Test that everything is working
python -c "import ppera; print('✅ PPERA installed successfully')"
```

## First Run

### 1. Download Sample Dataset

Start with MovieLens (smallest dataset):

```bash
# Follow instructions in docs/datasets.md to download MovieLens
# Place files in data/raw/movielens/
```

### 2. Run Your First Experiment

```bash
# Run a quick collaborative filtering experiment
make run-cf-movielens

# Or run all algorithms on MovieLens (takes longer)
make run-all-movielens
```

### 3. View Results

```bash
# Start MLflow UI to see results
make run-mlflow
# Open http://localhost:5000 in your browser
```

## What's Next?

- **[Datasets Guide](datasets.md)** - Download and setup all datasets
- **[Experiments Guide](experiments.md)** - Run specific experiments and configure parameters
- **[Architecture Overview](architecture.md)** - Understand the codebase structure

## Troubleshooting

**Environment issues?** See [Troubleshooting Guide](troubleshooting.md)

**Questions?** Check our [Contributing Guidelines](contributing.md) for how to get help.