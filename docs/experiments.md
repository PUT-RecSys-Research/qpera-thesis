# Running Experiments

This guide explains how to run and configure experiments with different recommendation algorithms in the PPERA framework.

## Quick Commands

```bash
# Run all algorithms on all datasets (full experiment suite)
python -m ppera.main

# Run specific algorithm combinations (from Makefile)
make run-cf-movielens      # Collaborative Filtering on MovieLens
make run-cbf-amazonsales   # Content-Based on Amazon
make run-rl-postrecommendations  # Reinforcement Learning on Posts

# Start MLflow tracking server
make run-mlflow

# Run with custom parameters (manual override)
python -m ppera.main  # Uses predefined experiment configurations
```

!!! note "Experiment Configuration"
    The PPERA framework runs **predefined experiment configurations** automatically. Individual CLI arguments are not currently supported - experiments are configured in [`ppera/main.py`](../ppera/main.py).

## Available Algorithms

### 1. Collaborative Filtering (CF)
- **Implementation**: Cornac BPR (Bayesian Personalized Ranking)
- **Hyperparameters**: 
  - Factors: `NUM_FACTORS = 200`
  - Epochs: `NUM_EPOCHS = 100`
  - Learning rate: `0.01`
  - Regularization: `lambda_reg = 0.001`
- **Best for**: Large user-item interaction datasets
- **Datasets**: All supported
- **Training time**: ~5-15 minutes

### 2. Content-Based Filtering (CBF)  
- **Implementation**: TF-IDF with BERT tokenization
- **Features**: Primarily uses `genres` column for similarity
- **Tokenization**: BERT-based text processing
- **Best for**: Rich item metadata, cold-start scenarios
- **Datasets**: All (requires `genres` column)
- **Training time**: ~10-30 minutes

### 3. Reinforcement Learning (RL)
- **Implementation**: Knowledge Graph + Policy Gradient (based on PGPR)
- **Components**:
  - **TransE embeddings**: Knowledge graph entity/relation embeddings
  - **Actor-Critic**: Policy network with `[512, 256]` hidden layers
  - **Beam search**: Multi-hop path generation with `topk=[25, 5, 1]`
- **Best for**: Sequential recommendations, explainable paths
- **Datasets**: All (requires preprocessing into knowledge graph)
- **Training time**: ~30-120 minutes (GPU recommended)

## Experiment Configurations

### Predefined Experiment Matrix

The framework automatically runs experiments across:

```python
# From ppera/main.py - EXPERIMENT_CONFIGS
EXPERIMENT_CONFIGS = [
    {"algo": "CBF", "dataset": "movielens", "rows": 14000},
    {"algo": "CF", "dataset": "movielens", "rows": 14000},
    {"algo": "RL", "dataset": "movielens", "rows": 14000},
    {"algo": "CBF", "dataset": "amazonsales"},
    {"algo": "CF", "dataset": "amazonsales"},
    {"algo": "RL", "dataset": "amazonsales"},
    {"algo": "CBF", "dataset": "postrecommendations", "rows": 14000},
    {"algo": "CF", "dataset": "postrecommendations", "rows": 14000},
    {"algo": "RL", "dataset": "postrecommendations", "rows": 14000},
]
```

### Configuration Categories

Each algorithm/dataset combination runs through **three configuration types**:

#### 1. Clear (Baseline)
```python
{
    "run_label": "Clear",
    "privacy": False,
    "personalization": False
}
```

#### 2. Privacy Experiments
```python
# Privacy fractions: [0.1, 0.25, 0.5, 0.8]
{
    "run_label": f"Privacy_{fraction:.2f}",
    "privacy": True,
    "hide_type": "values_in_column",
    "columns_to_hide": ["title", "genres"],  # Metadata hiding
    "fraction_to_hide": fraction,
    "personalization": False
}
```

#### 3. Personalization Experiments
```python
# Personalization fractions: [0.1, 0.25, 0.5, 0.8]
{
    "run_label": f"Personalization_{fraction:.2f}",
    "privacy": False,
    "personalization": True,
    "fraction_to_change": fraction,
    "change_rating": True  # Modify ratings to item averages
}
```

### Core Parameters

**Fixed across all experiments:**
```python
BASE_PARAMS = {
    "TOP_K": 10,
    "want_col": ["userID", "itemID", "rating", "timestamp", "title", "genres"],
    "ratio": 0.75,  # Train/test split
    "seed": 42
}
```

## Privacy & Personalization Details

### Privacy Simulation

**Data hiding implementation** ([`ppera/data_manipulation.py`](../ppera/data_manipulation.py)):

```python
def hide_information_in_dataframe(data, hide_type, columns_to_hide, 
                                 fraction_to_hide, records_to_hide, seed):
```

**Hide types:**
- `"values_in_column"`: Randomly hide percentage of values in specified columns
- Targets metadata columns: `["title", "genres"]` (not ratings)

**Example**: Privacy_0.25 hides 25% of movie titles and genres

### Personalization Simulation

**Preference modification** ([`ppera/data_manipulation.py`](../ppera/data_manipulation.py)):

```python
def change_items_in_dataframe(all, data, fraction_to_change, change_rating, seed):
```

**Process:**
1. **Item substitution**: Replace items based on global popularity distribution
2. **Rating adjustment**: Update ratings to item's average rating (if `change_rating=True`)
3. **Metadata sync**: Update title/genres for new items

**Example**: Personalization_0.10 changes 10% of user's items to more popular alternatives

## Evaluation Metrics

### Accuracy Metrics (Computed for all algorithms)

**From Microsoft Recommenders:**
```python
# External metrics
eval_mae = mae(test, predictions)
eval_rmse = rmse(test, predictions)  
eval_ndcg = ndcg_at_k(test, predictions, k=1)
```

**Custom PPERA metrics** ([`ppera/metrics.py`](../ppera/metrics.py)):
```python
eval_precision_at_k = precision_at_k(test, predictions, k=TOP_K)
eval_recall_at_k = recall_at_k(test, predictions, k=TOP_K)
eval_precision = precision_at_k(test, predictions, k=1)
eval_recall = recall_at_k(test, predictions, k=1)
eval_mrr = mrr(test, predictions)
```

### Coverage & Diversity Metrics

```python
eval_user_coverage = user_coverage(test, predictions)
eval_item_coverage = item_coverage(test, predictions)
eval_personalization = personalization_score(train, predictions)
eval_intra_list_similarity = intra_list_similarity_score(data, predictions, feature_cols=["genres"])
eval_intra_list_dissimilarity = intra_list_dissimilarity(data, predictions, feature_cols=["genres"])
```

### Error Handling

All metrics use defensive programming:
```python
try:
    eval_precision = precision_at_k(test, top_k, ...)
except Exception as e:
    eval_precision = None
    print(f"Error calculating precision: {e}")
```

## RL-Specific Pipeline

### Stage 1: Knowledge Graph Preprocessing

```python
# From ppera/RL.py
data_df, train_df, test_df = preprocess_rl(
    dataset=dataset, want_col=want_col, num_rows=num_rows, 
    ratio=ratio, seed=seed, privacy=privacy, personalization=personalization
)
```

**Entity extraction:**
- `USERID`, `ITEMID`, `TITLE`, `GENRES`, `RATING`
- Creates bidirectional mappings and relation tuples

### Stage 2: TransE Embedding Training

```python
train_transe_model_rl(dataset=dataset, seed=seed)
```

**Knowledge graph relations:**
```python
# From ppera/rl_utils.py
RELATIONS = {
    WATCHED: (user_idx, item_idx),
    BELONG_TO: (item_idx, genre_idx),
    DESCRIBED_AS: (title_idx, item_idx),
    RATED: (user_idx, title_idx),
    USER_RATED_WITH_VALUE: (user_idx, rating_idx),
    RATING_VALUE_FOR_ITEM: (rating_idx, item_idx)
}
```

### Stage 3: Policy Training

```python
train_agent_rl(dataset=dataset, seed=seed)
```

**Actor-Critic architecture:**
- State: User embedding + current node + history
- Action: Select relation/entity for next hop
- Training: Policy gradient with value function baseline

### Stage 4: Beam Search Inference

```python
test_agent_rl(dataset, TOP_K, ...)
```

**Path generation:**
- Multi-hop reasoning (up to 3 steps)
- Beam search with `topk=[25, 5, 1]`
- Path scoring: Policy probability × TransE compatibility

## Results Tracking

### MLflow Integration

**Automatic experiment tracking** ([`ppera/log_mlflow.py`](../ppera/log_mlflow.py)):

```bash
# Start MLflow server first
make run-mlflow
# Or manually: mlflow server --host 127.0.0.1 --port 5000
```

**Experiment organization:**
- **CF experiments**: `"MLflow Collaborative Filtering"`
- **CBF experiments**: `"MLflow Content Based Filtering"`
- **RL experiments**: `"MLflow Reinforcement Learning"`

**Logged artifacts:**
```python
# Parameters
{"dataset": dataset, "num_rows": num_rows, "seed": seed, 
 "privacy": privacy, "personalization": personalization}

# Metrics (all computed metrics with error handling)
{"precision": eval_precision, "recall": eval_recall, "ndcg_at_k": eval_ndcg, ...}

# Artifacts
- Dataset files (with row/seed suffixes)
- Model signatures (when supported)
- Prediction samples
```

### Console Output

**Real-time metrics display:**
```python
def format_metric(metric):
    return f"{metric:.4f}" if isinstance(metric, (float, int)) else "N/A"

print(
    "Precision:\t" + format_metric(eval_precision),
    "Recall@K:\t" + format_metric(eval_recall_at_k),
    "NDCG:\t" + format_metric(eval_ndcg),
    "User coverage:\t" + format_metric(eval_user_coverage),
    "Personalization:\t" + format_metric(eval_personalization),
    sep="\n"
)
```

### Experiment Progress Tracking

**Comprehensive logging** ([`ppera/main.py`](../ppera/main.py)):

```python
# Configuration-level tracking
logger.info(f"Starting Configuration Run: {run_label}")

# Individual experiment tracking  
logger.info(f"Starting Individual Experiment: Config {config_idx+1}/{total_configs} - Exp {exp_idx+1}/{total_exps}")

# Success/failure tracking
logger.info(f"Finished Individual Experiment successfully in {duration:.2f} seconds")
logger.error(f"Individual Experiment FAILED: {type(e).__name__}: {e}")
```

## Performance Optimization

### Dataset Size Management

**Row limiting** (sequential, not random):
```python
# From datasets_loader.py
data = loader("movielens", num_rows=14000, seed=42)  
# Creates cached file: merge_file_r14000_s42.csv
```

**Memory considerations:**
- **MovieLens**: Use `rows=14000` for faster testing
- **Amazon Sales**: Full dataset (large but manageable)
- **Post Recommendations**: Use `rows=14000` (generates ratings)

### RL Performance

**GPU acceleration:**
```python
# Automatic GPU detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Caching system:**
```python
# RL uses extensive caching in ppera/rl_tmp/
processed_dataset.pkl  # Entity mappings
kg.pkl                # Knowledge graph
transe_embed.pkl      # TransE embeddings
train_label.pkl       # Training labels
test_label.pkl        # Test labels
```

## Error Recovery & Robustness

### Graceful Degradation

**Metric calculation failures:**
```python
# Each metric wrapped in try/catch
# Experiment continues even if individual metrics fail
# Results show "N/A" for failed metrics
```

**Experiment-level failures:**
```python
# Individual experiments can fail without stopping the batch
# Detailed error logging with stack traces
# Summary shows success/failure counts
```

### Comprehensive Logging

**Multi-level tracking:**
```python
# Console: Real-time progress and results
# Logger: Detailed experiment tracking
# MLflow: Persistent metric storage
# Error logs: Full stack traces for debugging
```

## Troubleshooting

### Common Issues

#### 1. MLflow Connection
```bash
# Check if MLflow is running
curl -s --fail http://localhost:5000

# Start MLflow server
make run-mlflow

# Clear MLflow cache if corrupted
rm -rf mlruns/
```

#### 2. Memory Issues
```python
# Reduce dataset size
# Edit ppera/main.py EXPERIMENT_CONFIGS to add smaller "rows" values
{"algo": "CF", "dataset": "movielens", "rows": 5000}
```

#### 3. RL Training Issues
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Clear RL cache if corrupted
rm -rf ppera/rl_tmp/Movielens/

# Monitor memory usage during training
htop  # or top on some systems
```

#### 4. Dataset Loading Errors
```bash
# Verify datasets are downloaded
make check-datasets

# Check file structure
ls -la datasets/MovieLens/
ls -la datasets/AmazonSales/
ls -la datasets/PostRecommendations/
```

#### 5. Import Errors
```bash
# Reinstall in development mode
make setup

# Check package installation
python -c "import ppera; print('✅ PPERA imported successfully')"
```

### Performance Notes

**Algorithm comparison:**
- **CF**: Fastest baseline (5-15 minutes)
- **CBF**: Medium speed, good explanations (10-30 minutes)  
- **RL**: Slowest, best explainability (30-120 minutes)

**Dataset comparison:**
- **MovieLens**: Well-structured, fastest processing
- **Amazon Sales**: Largest, requires more memory
- **Post Recommendations**: Generates ratings, medium processing time

### Debugging Commands

```bash
# Test single dataset loading
python -c "from ppera.datasets_loader import loader; print(loader('movielens', num_rows=100).shape)"

# Test MLflow connection
python -c "import mlflow; print('MLflow version:', mlflow.__version__)"

# Check GPU/CUDA
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Monitor experiment progress
tail -f experiment_runner.log
```

### Expected Output Structure

**Successful experiment run produces:**
```
===== Config: Clear | Running Experiment: Algorithm = CF, Dataset = movielens =====
Took X.XX seconds for training.
Took X.XX seconds for prediction.

Precision:          0.XXXX
Recall@K:          0.XXXX
NDCG:              0.XXXX
User coverage:     0.XXXX
Personalization:   0.XXXX

--- Finished Individual Experiment in XX.XX seconds ---
```

**MLflow experiment shows:**
- Run name: `CF_movielens_Clear_rows14000_seed42`
- Parameters: All experimental settings
- Metrics: All computed evaluation metrics
- Artifacts: Dataset files, model info

---

*For detailed API documentation, see the [API Reference](api.md). For implementation details, see the [Architecture Guide](architecture.md).*