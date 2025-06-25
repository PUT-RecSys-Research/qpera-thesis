# Project Architecture

This document explains the codebase structure and key components of the PPERA framework.

## Project Structure

```
.
├── .gitignore          # Git ignore patterns for ML/data files
├── LICENSE             # MIT License file
├── Makefile            # Convenience commands for setup, testing, and execution
├── README.md           # You are here! Main project documentation
├── environment.yml     # Conda environment specification for reproducibility
├── mkdocs.yml          # Configuration for the MkDocs documentation site
├── pyproject.toml      # Python project configuration and dependencies (PEP 621)
│
├── datasets/           # ⚠️ Raw and processed datasets (ignored by Git)
│   ├── AmazonSales/
│   ├── MovieLens/
│   └── PostRecommendation/
│
├── docs/               # 📚 Project documentation source files (for MkDocs)
│   ├── api.md
│   ├── architecture.md
│   ├── citation.md
│   ├── contributing.md
│   ├── datasets.md
│   ├── experiments.md
│   └── results.md
│
├── ppera/              # 🐍 Main source code package
│   ├── __init__.py                     # Makes `ppera` a Python package
│   ├── main.py                         # Main CLI entry point and experiment orchestrator
│   │
│   ├── # --- Data Handling ---
│   ├── datasets_downloader.py          # Utilities for downloading datasets
│   ├── datasets_loader.py              # Unified data loading and preprocessing
│   ├── data_manipulation.py            # Data transformation, augmentation, and privacy
│   │
│   ├── # --- Core Algorithms ---
│   ├── CBF.py                          # Content-Based Filtering implementation
│   ├── CF.py                           # Collaborative Filtering implementation
│   ├── RL.py                           # Reinforcement Learning orchestrator
│   │
│   ├── # --- Reinforcement Learning Components ---
│   ├── rl_preprocess.py                # Data preprocessing for the RL environment
│   ├── rl_knowledge_graph.py           # Knowledge graph construction
│   ├── rl_kg_env.py                    # RL environment combining state, action, and KG
│   ├── rl_transe_model.py              # TransE model implementation for KG embeddings
│   ├── rl_train_transe_model.py        # Script to train the TransE model
│   ├── rl_train_agent.py               # RL agent training logic
│   ├── rl_test_agent.py                # RL agent testing and evaluation logic
│   ├── rl_prediction.py                # Generates recommendations using the trained RL agent
│   ├── rl_decoder.py                   # Decodes agent output into recommendations
│   ├── rl_utils.py                     # Utility functions for the RL components
│   │
│   ├── # --- Utilities & Tooling ---
│   ├── metrics.py                      # Evaluation metrics (e.g., NDCG, HR)
│   ├── log_mlflow.py                   # MLflow integration for experiment tracking
│   ├── frequency_based_rating_gen.py   # Synthetic rating generation
│   └── rating_timestamp_gen.py         # Utilities for generating synthetic timestamps
│
├── reports/            # 📊 Generated analysis, figures, and results
│   ├── clean_loop/                     # Baseline (unmodified) experiment results
│   ├── explainability/                 # Analysis of explanation methods
│   ├── personalization/                # Analysis of personalization quality
│   ├── privacy/                        # Analysis of privacy preservation
│   ├── conversion.ipynb              # Notebook for data format conversions
│   └── generate_figures.ipynb        # Notebook to generate final plots for reports
│
└── references/         # 📄 Academic papers, literature, and external resources
```

## Core Components

### 1. Main Entry Point ([`ppera/main.py`](../ppera/main.py))

The central orchestration script that manages experiment execution:

```python
# Example usage
python -m ppera.main --algo CF --dataset movielens --privacy
```

**Key responsibilities:**
- Parse command-line arguments and configuration
- Iterate through experiment configurations (Clear, Privacy, Personalization)
- Execute algorithm-specific experiment loops
- Handle errors and logging for batch experiments
- Track overall experiment progress and timing

**Experiment Configurations:**
```python
EXPERIMENT_CONFIGS = [
    {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "movielens"},
    {"algo": "CF", "module": CF, "func": "cf_experiment_loop", "dataset": "movielens"},
    {"algo": "RL", "module": RL, "func": "rl_experiment_loop", "dataset": "movielens", "rows": 14000},
    # ... more combinations for amazonsales and postrecommendations
]
```

### 2. Dataset Loading ([`ppera/datasets_loader.py`](../ppera/datasets_loader.py))

Implements dataset-specific loaders with inheritance pattern:

```python
class BaseDatasetLoader:
    def load_data(self, want_col, num_rows=None, seed=42) -> pd.DataFrame
    def merge_datasets(self) -> pd.DataFrame  # Abstract method

class MovieLensDataset(BaseDatasetLoader):
    # Handles rating.csv, movie.csv, tag.csv
    
class AmazonSalesDataset(BaseDatasetLoader):
    # Handles Amazon product interaction data
    
class PostRecommendationsDataset(BaseDatasetLoader):
    # Handles social media post interactions with frequency-based rating generation
```

**Key features:**
- **Automatic file validation** and download instructions
- **Column normalization** (userId → userID, movieId → itemID)
- **Genre preprocessing** (pipe-separated → space-separated)
- **Timestamp conversion** to Unix format
- **Duplicate removal** and data cleaning

### 3. Algorithm Implementations

#### Collaborative Filtering ([`ppera/CF.py`](../ppera/CF.py))

```python
def cf_experiment_loop(TOP_K, dataset, want_col, num_rows, ratio, seed, 
                      personalization=False, privacy=False, ...):
```

**Implementation details:**
- **Model**: Cornac BPR (Bayesian Personalized Ranking)
- **Hyperparameters**: `k=200` factors, `max_iter=100` epochs, `learning_rate=0.01`
- **Prediction**: Uses `predict_ranking()` with `remove_seen=True`
- **Data flow**: Load → Split → Privacy/Personalization transforms → Train → Predict → Evaluate

#### Content-Based Filtering ([`ppera/CBF.py`](../ppera/CBF.py))

```python
def cbf_experiment_loop(TOP_K, dataset, want_col, num_rows, ratio, seed,
                       personalization=False, privacy=False, ...):
```

**Implementation details:**
- **Model**: TfidfRecommender with BERT tokenization
- **Primary feature**: `genres` column for similarity matching
- **Preprocessing**: Item deduplication, text cleaning
- **Recommendation**: Cosine similarity-based ranking

#### Reinforcement Learning ([`ppera/RL.py`](../ppera/RL.py))

Orchestrates multi-stage RL pipeline:

```python
def rl_experiment_loop(TOP_K, dataset, want_col, num_rows, ratio, seed,
                      personalization=False, privacy=False, ...):
    preprocess_rl(...)      # Knowledge graph construction
    train_transe_model_rl(...) # TransE embedding training  
    train_agent_rl(...)    # Policy network training
    test_agent_rl(...)     # Evaluation and recommendation
```

### 4. Reinforcement Learning Pipeline

#### Knowledge Graph Preprocessing ([`ppera/rl_preprocess.py`](../ppera/rl_preprocess.py))

```python
def create_processed_dataset(df: pd.DataFrame) -> dict:
```

**Entity extraction:**
- `USERID`, `ITEMID`, `TITLE`, `GENRES`, `RATING`
- Creates bidirectional mappings: `original_id ↔ internal_index`

**Relation construction:**
- `WATCHED`: (user_idx, item_idx) from ratings
- `BELONG_TO`: (item_idx, genre_idx) from metadata  
- `DESCRIBED_AS`: (title_idx, item_idx) from movie titles
- `RATED`: (user_idx, title_idx) derived from watched + titles
- `RATING_VALUE_FOR_ITEM`: (rating_val_idx, item_idx)

**Distribution calculation:**
- Computes tail entity distributions for negative sampling
- Used during TransE training for corrupted triple generation

#### TransE Model ([`ppera/rl_transe_model.py`](../ppera/rl_transe_model.py))

```python
class KnowledgeEmbedding(nn.Module):
    def __init__(self, processed_dataset, idx_to_relation_name_map, args):
```

**Features:**
- **Dynamic layer creation**: Embedding layers for each entity type
- **Translation-based learning**: h + r ≈ t principle
- **Relation-specific distributions**: Different sampling strategies per relation
- **Loss function**: Margin-based ranking loss with L2 regularization

#### Policy Network ([`ppera/rl_train_agent.py`](../ppera/rl_train_agent.py))

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
```

**Architecture:**
- **Actor network**: Policy for selecting relations/entities
- **Critic network**: Value function estimation
- **State representation**: User embedding + current node + history
- **Action space**: Available relations from current node

#### Environment ([`ppera/rl_kg_env.py`](../ppera/rl_kg_env.py))

```python
class BatchKGEnvironment:
    def __init__(self, dataset, max_acts, max_path_len=3, state_history=1):
```

**State management:**
```python
class KGState:
    # History Length 0: [user_embed, node_embed]
    # History Length 1: [user_embed, node_embed, last_node_embed, last_relation_embed]  
    # History Length 2: [user_embed, node_embed, last_node_embed, last_relation_embed, 
    #                    older_node_embed, older_relation_embed]
```

#### Beam Search Inference ([`ppera/rl_test_agent.py`](../ppera/rl_test_agent.py))

```python
def batch_beam_search(env, model, uids, device, topk=[25, 5, 1]):
```

**Path generation:**
- **Multi-hop reasoning**: Up to 3-step paths from user to item
- **Beam search**: Maintains top-k paths at each step
- **Path validation**: Ensures valid relation transitions
- **Probability scoring**: Combines path probability with TransE scores

### 5. Evaluation Framework

#### Custom Metrics ([`ppera/metrics.py`](../ppera/metrics.py))

**Accuracy metrics:**
```python
def precision_at_k(rating_true, rating_pred, k=1) -> float
def recall_at_k(rating_true, rating_pred, k=1) -> float  
def mrr(rating_true, rating_pred, k=1) -> float
def f1(rating_true, rating_pred, k=1) -> float
```

**Coverage metrics:**
```python
def user_coverage(rating_true, rating_pred, threshold=10.0) -> float
def item_coverage(rating_true, rating_pred, threshold=10.0) -> float
```

**Diversity metrics:**
```python
def personalization(predicted: List[list]) -> float  # 1 - average_jaccard_similarity
def intra_list_similarity_score(item_features, rating_pred, feature_cols) -> float
def intra_list_dissimilarity(...) -> float  # 1 - intra_list_similarity
```

**Integration with external libraries:**
- Microsoft Recommenders: `mae()`, `rmse()`, `ndcg_at_k()`
- Recmetrics: `personalization()`, `intra_list_similarity()`

#### MLflow Integration ([`ppera/log_mlflow.py`](../ppera/log_mlflow.py))

```python
def log_mlflow(dataset, top_k, metrics, num_rows, seed, model, model_type, 
               params, data, train, privacy=None, personalization=None, ...):
```

**Experiment organization:**
- **CF Experiments**: `"MLflow Collaborative Filtering"`
- **CBF Experiments**: `"MLflow Content Based Filtering"`
- **RL Experiments**: `"MLflow Reinforcement Learning"`

**Logged artifacts:**
- Dataset files (with row/seed suffixes)
- Model signatures (when supported)
- Hyperparameters and experiment conditions
- All computed metrics
- Prediction visualization plots

### 6. Data Manipulation ([`ppera/data_manipulation.py`](../ppera/data_manipulation.py))

#### Privacy Simulation
```python
def hide_information_in_dataframe(data, hide_type, columns_to_hide, 
                                 fraction_to_hide, records_to_hide, seed):
```

**Privacy attack types:**
- `"values_in_column"`: Hide percentage of values in specified columns
- `"full_record"`: Remove entire user-item interaction records
- Targets: `["title", "genres"]` or `["rating"]` columns

#### Personalization Simulation
```python
def change_items_in_dataframe(all, data, fraction_to_change, change_rating, seed):
```

**Preference modifications:**
- **Item substitution**: Replace items based on global popularity distribution
- **Rating adjustment**: Update ratings to item's average rating
- **Metadata synchronization**: Update title/genres for new items

### 7. Knowledge Graph Constants ([`ppera/rl_utils.py`](../ppera/rl_utils.py))

**Entity definitions:**
```python
USERID = "user_id"
ITEMID = "item_id" 
TITLE = "title"
GENRES = "genres"
RATING = "rating"
```

**Relation mappings:**
```python
KG_RELATION = {
    USERID: {WATCHED: ITEMID, RATED: TITLE, USER_RATED_WITH_VALUE: RATING},
    ITEMID: {WATCHED: USERID, BELONG_TO: GENRES},
    TITLE: {RATED: USERID, DESCRIBED_AS: ITEMID},
    GENRES: {BELONG_TO: ITEMID},
    RATING: {USER_RATED_WITH_VALUE: USERID, RATING_VALUE_FOR_ITEM: ITEMID},
}
```

**Path patterns for recommendation:**
```python
PATH_PATTERN = {
    1: ((None, USERID), (RATED, TITLE), (DESCRIBED_AS, ITEMID)),           # User→Title→Item
    2: ((None, USERID), (USER_RATED_WITH_VALUE, RATING), (RATING_VALUE_FOR_ITEM, ITEMID)), # User→Rating→Item
    # ... up to 15 different path types
}
```

## Data Flow Architecture

### 1. Experiment Execution Flow

```
CLI Arguments → Parameter Configuration → Algorithm Selection → 
Dataset Loading → Privacy/Personalization Transforms → 
Algorithm Training → Prediction → Evaluation → MLflow Logging
```

### 2. RL Knowledge Graph Pipeline

```
Raw DataFrame → Entity Extraction → Relation Mining → 
TransE Embedding Training → Policy Network Training → 
Beam Search Inference → Path-based Recommendations
```

### 3. Evaluation Pipeline

```
Predictions + Test Data → Metric Calculation → 
Error Handling (try/catch) → Result Formatting → 
MLflow Logging + Console Output
```

## Key Design Patterns

### 1. Modular Algorithm Interface

Each algorithm implements a consistent experiment loop signature:

```python
def algorithm_experiment_loop(TOP_K, dataset, want_col, num_rows, ratio, seed,
                             personalization=False, fraction_to_change=0, change_rating=False,
                             privacy=False, hide_type="values_in_column", 
                             columns_to_hide=None, fraction_to_hide=0, records_to_hide=None):
```

### 2. Comprehensive Error Handling

All metric calculations use defensive programming:

```python
try:
    eval_precision = precision_at_k(test, top_k, ...)
except Exception as e:
    eval_precision = None
    print(f"Error calculating precision: {e}")
```

### 3. Configuration-Driven Experiments

Experiments use parameter dictionaries for flexibility:

```python
all_param_configurations = [
    {"run_label": "Clear", "privacy": False, "personalization": False},
    {"run_label": "Privacy_0.10", "privacy": True, "fraction_to_hide": 0.1},
    {"run_label": "Personalization_0.25", "personalization": True, "fraction_to_change": 0.25},
]
```

### 4. Caching and Persistence

RL components use extensive caching:

```python
# Cache processed datasets, knowledge graphs, embeddings, and labels
processed_dataset_file = TMP_DIR[dataset] + "/processed_dataset.pkl"
kg_file = TMP_DIR[dataset] + "/kg.pkl"
embed_file = TMP_DIR[dataset] + "/transe_embed.pkl"
```

## Performance Optimizations

### Memory Management
- **Sparse relations**: Sets for unique relation tuples
- **Chunked processing**: Batch operations for large datasets
- **Lazy loading**: Load cached files when available

### Computational Efficiency  
- **GPU acceleration**: CUDA support for RL training
- **Vectorized operations**: NumPy/PyTorch for matrix computations
- **Early stopping**: Configurable training epochs

### Error Recovery
- **Graceful degradation**: Continue experiments if individual metrics fail
- **Detailed logging**: Track failures with stack traces
- **Fallback mechanisms**: Random split if stratified split fails
