# Contributing to PPERA

Thank you for your interest in contributing to the **Personalization, Privacy, and Explainability of Recommendation Algorithms (PPERA)** framework! This guide will help you get started with development, testing, and submitting contributions.

## 🚀 Quick Start for Contributors

### Prerequisites

- **Python 3.9+** (3.9 recommended for compatibility)
- **Conda/Miniconda** for environment management
- **Git** for version control
- **CUDA-capable GPU** (optional, for RL training acceleration)

### Development Setup

1. **Fork and Clone**
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/personalization-privacy-and-explainability-of-recommendation-algorithms.git
cd personalization-privacy-and-explainability-of-recommendation-algorithms

# Add upstream remote
git remote add upstream https://github.com/Master-s-thesis-PPERA/personalization-privacy-and-explainability-of-recommendation-algorithms.git
```

2. **Environment Setup**
```bash
# Create and activate conda environment
make install
conda activate ppera-env

# Install package in development mode (CRUCIAL)
make setup

# Verify installation
python -c "import ppera; print('✅ PPERA installed successfully')"
```

3. **Download Test Datasets**
```bash
# Download MovieLens for testing (smallest dataset)
# Follow instructions in docs/datasets.md
# Place files in datasets/MovieLens/
```

## 🏗️ Development Workflow

### Branch Strategy

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code changes ...

# Commit with descriptive messages
git add .
git commit -m "feat: add new personalization metric"

# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

### Code Quality Standards

#### Linting and Formatting

```bash
# Check code quality
make lint

# Auto-format code
make format

# Both commands use ruff for fast linting and formatting
```

#### Code Style Guidelines

- **Follow PEP 8** for Python code style
- **Use descriptive variable names**: `eval_precision_at_k` not `p`
- **Add docstrings** for all functions and classes
- **Include type hints** where appropriate
- **Handle exceptions gracefully** with try/catch blocks

**Example of good code style:**
```python
def precision_at_k(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = "userID",
    col_item: str = "itemID",
    col_rating: str = "rating",
    col_prediction: str = "prediction",
    k: int = 10
) -> float:
    """
    Calculate precision@k for recommendation predictions.
    
    Args:
        rating_true: Ground truth ratings
        rating_pred: Model predictions
        k: Number of top recommendations to consider
        
    Returns:
        Precision@k score between 0 and 1
        
    Raises:
        ValueError: If k <= 0 or required columns missing
    """
    try:
        # Implementation with proper error handling
        pass
    except Exception as e:
        print(f"Error calculating precision@k: {e}")
        return None
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run specific algorithm on small dataset for testing
python -m ppera.main --algo CF --dataset movielens --rows 1000

# Test with privacy settings
python -m ppera.main --algo CBF --dataset movielens --rows 500 --privacy --fraction-to-hide 0.1

# Test MLflow integration
make run-mlflow
# In another terminal:
python -m ppera.main --algo CF --dataset movielens --rows 100
```

### Manual Testing Checklist

When contributing new features, verify:

- ✅ **Algorithm integration**: New algorithms work with main experiment loop
- ✅ **Error handling**: Graceful failure when metrics can't be calculated
- ✅ **MLflow logging**: All metrics and parameters logged correctly
- ✅ **Privacy/Personalization**: Data manipulation functions work as expected
- ✅ **Memory usage**: No memory leaks with large datasets
- ✅ **Cross-platform**: Works on Linux, macOS, Windows

### Dataset-Specific Testing

```bash
# Test all algorithms on each dataset
python -m ppera.main --algo CF --dataset movielens --rows 1000
python -m ppera.main --algo CBF --dataset amazonsales --rows 1000  
python -m ppera.main --algo RL --dataset postrecommendations --rows 1000
```

## 📊 Areas for Contribution

### 1. Algorithm Implementations

#### Adding New Recommendation Algorithms

**File structure for new algorithm:**
```python
# ppera/NEW_ALGORITHM.py
def new_algorithm_experiment_loop(
    TOP_K, dataset, want_col, num_rows, ratio, seed,
    personalization=False, fraction_to_change=0, change_rating=False,
    privacy=False, hide_type="values_in_column", 
    columns_to_hide=None, fraction_to_hide=0, records_to_hide=None
):
    """Follow the exact signature pattern of existing algorithms"""
    
    # 1. Load and preprocess data
    data = datasets_loader.loader(dataset, want_col, num_rows, seed)
    
    # 2. Apply privacy/personalization transforms
    if privacy:
        data = dm.hide_information_in_dataframe(...)
    if personalization:
        data = dm.change_items_in_dataframe(...)
    
    # 3. Train/test split
    train, test = python_stratified_split(data, ratio=ratio, seed=seed)
    
    # 4. Train your model
    model = YourModel()
    model.fit(train)
    
    # 5. Generate predictions
    predictions = model.predict(test)
    
    # 6. Calculate all standard metrics (with error handling)
    metrics = calculate_all_metrics(test, predictions)
    
    # 7. Log to MLflow
    log_mlflow.log_mlflow(dataset, predictions, metrics, ...)
```

**Register in main.py:**
```python
from . import NEW_ALGORITHM

EXPERIMENT_CONFIGS = [
    # ... existing configs ...
    {"algo": "NEW_ALG", "module": NEW_ALGORITHM, "func": "new_algorithm_experiment_loop", 
     "dataset": "movielens"},
]
```

#### Algorithm-Specific Requirements

**For Matrix Factorization algorithms:**
- Use sparse matrices for memory efficiency
- Implement both explicit and implicit feedback variants
- Handle cold-start users/items gracefully

**For Deep Learning algorithms:**
- Add GPU/CPU detection and switching
- Implement early stopping and checkpointing
- Use PyTorch for consistency with RL implementation

**For Knowledge Graph algorithms:**
- Extend entity/relation definitions in `rl_utils.py`
- Implement custom knowledge graph construction
- Add new path patterns for explainability

### 2. Evaluation Metrics

#### Adding New Metrics

**Custom metrics should follow this pattern:**
```python
# ppera/metrics.py
def new_metric_name(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
    **kwargs
) -> float:
    """
    Calculate new evaluation metric.
    
    Args:
        rating_true: Ground truth interactions
        rating_pred: Model predictions
        **kwargs: Additional parameters
        
    Returns:
        Metric value (higher/lower is better - document this!)
        
    Raises:
        ValueError: For invalid inputs
    """
    # Validate inputs
    check_column_dtypes(rating_true, rating_pred)
    
    # Calculate metric
    try:
        result = your_calculation_logic()
        return result
    except Exception as e:
        raise ValueError(f"Error calculating {new_metric_name.__name__}: {e}")
```

**Add to algorithm evaluation loops:**
```python
# In CF.py, CBF.py, RL.py
try:
    eval_new_metric = new_metric_name(test, top_k, ...)
except Exception as e:
    eval_new_metric = None
    print(f"Error calculating new metric: {e}")

metrics = {
    # ... existing metrics ...
    "new_metric": eval_new_metric,
}
```

#### Priority Metrics to Implement

1. **Fairness Metrics**
   - Demographic parity
   - Equalized odds
   - Individual fairness

2. **Novelty and Serendipity**
   - Item novelty based on popularity
   - Serendipity using content similarity
   - Discovery metrics

3. **Temporal Metrics**
   - Performance degradation over time
   - Concept drift detection
   - Real-time adaptation speed

### 3. Dataset Support

#### Adding New Datasets

**Extend datasets_loader.py:**
```python
class NewDatasetLoader(BaseDatasetLoader):
    def __init__(self, base_path):
        super().__init__(base_path)
        
    def merge_datasets(self) -> pd.DataFrame:
        """Implement dataset-specific loading logic"""
        # Load your CSV files
        interactions = pd.read_csv(self.base_path / "interactions.csv")
        metadata = pd.read_csv(self.base_path / "metadata.csv")
        
        # Standardize column names
        interactions = interactions.rename(columns={
            'user_id': 'userID',
            'item_id': 'itemID', 
            'score': 'rating'
        })
        
        # Merge and return
        return interactions.merge(metadata, on='itemID')

# Add to loader() function
dataset_loaders = {
    "movielens": MovieLensDataset,
    "amazonsales": AmazonSalesDataset,
    "postrecommendations": PostRecommendationsDataset,
    "newdataset": NewDatasetLoader,  # Add here
}
```

**Update RL utilities for knowledge graphs:**
```python
# ppera/rl_utils.py
DATASET_DIR = {
    # ... existing datasets ...
    "newdataset": "./datasets/NewDataset",
}

TMP_DIR = {
    # ... existing datasets ...
    "newdataset": "ppera/rl_tmp/NewDataset",
}
```

### 4. Privacy and Personalization

#### Privacy Attack Implementations

```python
# ppera/privacy_attacks.py (new file)
def membership_inference_attack(model, train_data, test_data):
    """Implement membership inference attack"""
    pass

def attribute_inference_attack(model, partial_data, target_attribute):
    """Implement attribute inference attack"""
    pass

def model_inversion_attack(model, target_user):
    """Implement model inversion attack"""
    pass
```

#### Differential Privacy Integration

```python
# ppera/differential_privacy.py (new file)
def add_laplace_noise(data, epsilon, sensitivity):
    """Add calibrated Laplace noise for differential privacy"""
    pass

def private_aggregation(data, epsilon, query_function):
    """Perform differentially private aggregation"""
    pass
```

### 5. Explainability Features

#### Explanation Generators

```python
# ppera/explainability.py (new file)
class CFExplainer:
    """Generate explanations for collaborative filtering"""
    
    def explain_recommendation(self, user_id, item_id, model, data):
        """Generate 'users like you also liked' explanations"""
        pass

class CBFExplainer:
    """Generate explanations for content-based filtering"""
    
    def explain_recommendation(self, user_id, item_id, model, data):
        """Generate 'because you liked similar items' explanations"""
        pass

class RLExplainer:
    """Generate explanations for reinforcement learning"""
    
    def explain_path(self, user_id, item_id, path, knowledge_graph):
        """Generate path-based explanations"""
        pass
```

## 🐛 Debugging and Troubleshooting

### Common Issues and Solutions

#### MLflow Connection Issues
```bash
# Check if MLflow server is running
curl -s --fail http://localhost:5000

# Start MLflow server
make run-mlflow

# Clear MLflow cache
rm -rf mlruns/
```

#### Memory Issues with Large Datasets
```python
# Use data chunking for large datasets
data = datasets_loader.loader(dataset, want_col, num_rows=10000, seed=seed)

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

#### CUDA/GPU Issues
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### Import Issues
```bash
# Reinstall in development mode
pip uninstall ppera
make setup

# Check package installation
pip show ppera
```

### Debugging Tools

#### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export PYTHONPATH="${PYTHONPATH}:."
```

#### Profile Performance
```python
import cProfile
import pstats

# Profile algorithm execution
cProfile.run('cf_experiment_loop(...)', 'profile_output.prof')
stats = pstats.Stats('profile_output.prof')
stats.sort_stats('cumtime').print_stats(10)
```

## 📝 Documentation Contributions

### Documentation Standards

- **Use clear, concise language**
- **Include code examples** for all functions
- **Add cross-references** between related components
- **Update docstrings** when modifying functions
- **Test documentation examples** to ensure they work

### Documentation Structure

```
docs/
├── index.md                    # Main landing page
├── getting-started.md          # Installation and first run
├── datasets.md                 # Dataset setup instructions
├── experiments.md              # Running experiments
├── architecture.md             # Code structure
├── api.md                      # API reference
├── contributing.md             # This file
├── troubleshooting.md          # Common issues
└── citation.md                 # How to cite
```

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve documentation locally
mkdocs serve

# Build static site
mkdocs build
```

## 🚦 Pull Request Process

### Before Submitting

1. **Test your changes** on at least one dataset
2. **Run linting**: `make lint`
3. **Format code**: `make format`
4. **Update documentation** if needed
5. **Add docstrings** for new functions
6. **Handle edge cases** and add error handling

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature (algorithm, metric, dataset)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tested on MovieLens dataset
- [ ] Tested with privacy settings
- [ ] Tested with personalization settings
- [ ] MLflow logging works correctly
- [ ] No memory leaks or performance regressions

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Backward compatibility maintained
```

### Review Process

1. **Automated checks** must pass (linting, basic tests)
2. **Manual review** by maintainers
3. **Testing** on different datasets and configurations
4. **Documentation review** for clarity and completeness
5. **Merge** after approval

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive in all interactions
- **Provide constructive feedback** in code reviews
- **Help newcomers** get started with the project
- **Follow academic integrity** standards
- **Give credit** where due in contributions

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Pull Requests**: Code contributions and reviews
- **Email**: Direct contact with maintainers for sensitive issues

### Attribution

All contributors will be acknowledged in:
- **README.md** contributor section
- **Academic publications** when appropriate
- **Release notes** for significant contributions

## 📚 Resources for Contributors

### Learning Resources

- **Recommendation Systems**: [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-4899-7637-6)
- **Privacy in ML**: [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- **Knowledge Graphs**: [Knowledge Graph Embedding Survey](https://arxiv.org/abs/2002.00819)
- **MLflow**: [Official MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Technical Documentation

- **Microsoft Recommenders**: [GitHub Repository](https://github.com/recommenders-team/recommenders)
- **PGPR**: [Original Paper](https://arxiv.org/abs/1906.05237)
- **Cornac**: [Documentation](https://cornac.readthedocs.io/)
- **PyTorch**: [Official Tutorials](https://pytorch.org/tutorials/)

### Development Tools

- **VS Code**: Recommended IDE with Python extension
- **Conda**: Environment management
- **Ruff**: Fast Python linter and formatter
- **Git**: Version control
- **MLflow**: Experiment tracking

## 🎯 Roadmap and Future Plans

### Short-term Goals (1-3 months)
- [ ] Add fairness metrics implementation
- [ ] Improve RL training stability and speed
- [ ] Add more privacy attack implementations
- [ ] Enhance documentation with more examples

### Medium-term Goals (3-6 months)
- [ ] Large Language Model integration for explanations
- [ ] Real-time recommendation capabilities
- [ ] Federated learning support
- [ ] Advanced visualization dashboard

### Long-term Goals (6+ months)
- [ ] Production deployment templates
- [ ] Integration with popular ML platforms
- [ ] Cross-domain recommendation evaluation
- [ ] Automated hyperparameter optimization

---

## 📞 Contact

For questions about contributing:

- **Julia Podsadna**: [GitHub](https://github.com/GambiBambi)
- **Bartosz Chwiłkowski**: [GitHub](https://github.com/kooogi)
- **Supervisor**: Prof. Mikołaj Morzy ([GitHub](https://github.com/megaduks))

**Institution**: Faculty of Computing and Telecommunications, Poznan University of Technology

---

Thank you for contributing to PPERA! Your contributions help advance research in privacy-aware, explainable recommendation systems. 🚀