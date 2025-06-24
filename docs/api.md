# API Reference

This document provides detailed API documentation for the PPERA framework components.

## Core Modules

### Main Entry Point

#### [`ppera.main`](../ppera/main.py)

The central orchestration module for running experiments.

**Key Functions:**

```python
def main()
```
Main CLI entry point that parses arguments and runs experiments.

**Experiment Configurations:**
- **Base Parameters**: `TOP_K=10`, `want_col`, `ratio=0.75`, `seed=42`
- **Privacy Configurations**: Fractions `[0.1, 0.25, 0.5, 0.8]` with `hide_type="values_in_column"`
- **Personalization Configurations**: Fractions `[0.1, 0.25, 0.5]` with optional rating changes

**Default Experiment Matrix:**
```python
EXPERIMENT_CONFIGS = [
    {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "movielens"},
    {"algo": "CF", "module": CF, "func": "cf_experiment_loop", "dataset": "movielens"},
    {"algo": "RL", "module": RL, "func": "rl_experiment_loop", "dataset": "movielens", "rows": 14000},
    # ... and more combinations
]
```

---

## Algorithm Implementations

### Collaborative Filtering

#### [`ppera.CF`](../ppera/CF.py)

**Main Function:**
```python
def cf_experiment_loop(
    TOP_K: int,
    dataset: str,
    want_col: list,
    num_rows: int = None,
    ratio: float = 0.75,
    seed: int = 42,
    personalization: bool = False,
    fraction_to_change: float = 0,
    change_rating: bool = False,
    privacy: bool = False,
    hide_type: str = "values_in_column",
    columns_to_hide: list = None,
    fraction_to_hide: float = 0,
    records_to_hide: list = None,
) -> None
```

**Core Implementation:**
- **Model**: Cornac BPR (Bayesian Personalized Ranking)
- **Hyperparameters**: `k=100` factors, `max_iter=100` epochs, `learning_rate=0.01`
- **Train/Test Split**: Uses `train_test_split()` with specified ratio

**Metrics Computed:**
- `precision`, `precision_at_k`, `recall`, `recall_at_k`
- `mae`, `rmse`, `mrr`, `ndcg_at_k`
- `user_coverage`, `item_coverage`
- `personalization`, `intra_list_similarity`, `intra_list_dissimilarity`

### Content-Based Filtering

#### [`ppera.CBF`](../ppera/CBF.py)

**Main Function:**
```python
def cbf_experiment_loop(
    TOP_K: int,
    dataset: str,
    want_col: list,
    num_rows: int = None,
    ratio: float = 0.75,
    seed: int = 42,
    personalization: bool = False,
    fraction_to_change: float = 0,
    change_rating: bool = False,
    privacy: bool = False,
    hide_type: str = "values_in_column",
    columns_to_hide: list = None,
    fraction_to_hide: float = 0,
    records_to_hide: list = None,
) -> None
```

**Core Implementation:**
- **Model**: TfidfRecommender with BERT tokenization
- **Features**: Primarily uses `genres` column for similarity
- **Recommendation**: Top-K based on TF-IDF cosine similarity

**Data Processing:**
```python
# TF-IDF vectorization
recommender = TfidfRecommender(id_col="itemID", tokenization_method="bert")
```

### Reinforcement Learning

#### [`ppera.RL`](../ppera/rl_test_agent.py)

**Core Classes:**

##### `ActorCritic` Model
```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, gamma: float = 0.99, hidden_sizes: list = [512, 256])
    
    def forward(self, inputs: torch.Tensor) -> tuple
    def select_action(self, batch_state: torch.Tensor, batch_act_mask: torch.Tensor, device: torch.device) -> tuple
    def update(self, optimizer: torch.optim.Optimizer, device: torch.device, ent_weight: float) -> None
```

**Key Functions:**

```python
def batch_beam_search(env, model, uids, device, topk=[25, 5, 1]) -> tuple
```
Performs beam search for path recommendation with configurable top-k at each hop.

```python
def predict_paths(policy_file: str, path_file: str, args) -> None
```
Loads trained policy and generates recommendation paths.

```python
def run_evaluation(path_file: str, train_labels, test_labels, TOP_K: int, data, train, test, args) -> None
```
Evaluates RL recommendations using path-based predictions.

**Knowledge Graph Components:**
- **Entities**: `USERID`, `ITEMID`, `TITLE`, `GENRES`, `RATING`
- **Relations**: `WATCHED`, `RATED`, `DESCRIBED_AS`, `BELONG_TO`, `RATING_VALUE_FOR_ITEM`
- **Embedding**: TransE model for knowledge graph embeddings

---

## Data Processing

### Dataset Loading

#### [`ppera.datasets_loader`](../ppera/datasets_loader.py)

**Core Function:**
```python
def loader(dataset: str, want_col: list, num_rows: int = None, seed: int = 42) -> pd.DataFrame
```

**Supported Datasets:**
- **MovieLens**: `"movielens"` - Movie ratings with genres
- **Amazon Sales**: `"amazonsales"` - Product interactions
- **Post Recommendations**: `"postrecommendations"` - Social media posts

### Knowledge Graph Preprocessing

#### [`ppera.rl_preprocess`](../ppera/rl_preprocess.py)

**Key Function:**
```python
def create_processed_dataset(df: pd.DataFrame) -> dict
```

**Output Structure:**
```python
{
    "entity_maps": {
        "user_id": {"map": dict, "vocab_size": int},
        "item_id": {"map": dict, "vocab_size": int},
        # ... other entities
    },
    "relations": {
        "watched": [(user_idx, item_idx), ...],
        "belongs_to": [(item_idx, genre_idx), ...],
        # ... other relations
    }
}
```

### Knowledge Graph Embeddings

#### [`ppera.rl_transe_model`](../ppera/rl_transe_model.py)

```python
class KnowledgeEmbedding(nn.Module):
    def __init__(self, processed_dataset: dict, idx_to_relation_name_map: dict, args)
```

**Features:**
- **TransE Model**: Knowledge graph embedding using translation-based approach
- **Dynamic Layer Creation**: Automatically creates embedding layers for each entity type
- **Relation-Specific Distributions**: Handles different relation types with appropriate distributions

---

## Evaluation Metrics

### Core Metrics

#### [`ppera.metrics`](../ppera/metrics.py)

**Accuracy Metrics:**
```python
def precision_at_k(rating_true, rating_pred, k: int = 1) -> float
def recall_at_k(rating_true, rating_pred, k: int = 1) -> float
def f1(rating_true, rating_pred, k: int = 1) -> float
def mrr(rating_true, rating_pred, k: int = 1) -> float
```

**Coverage Metrics:**
```python
def user_coverage(rating_true, rating_pred, threshold: float = 10.0) -> float
def item_coverage(rating_true, rating_pred, threshold: float = 10.0) -> float
```

**Diversity Metrics:**
```python
def personalization_score(rating_true, rating_pred) -> float
def intra_list_similarity_score(item_features, rating_pred, feature_cols: list = None) -> float
def intra_list_dissimilarity(item_features, rating_pred, feature_cols: list = None) -> float
```

**Advanced Personalization:**
```python
def personalization(predicted: List[list]) -> float
```
Computes 1 - average_jaccard_similarity across all user pairs' recommendation lists.

### External Metrics

**From Microsoft Recommenders:**
- `mae()` - Mean Absolute Error
- `rmse()` - Root Mean Square Error  
- `ndcg_at_k()` - Normalized Discounted Cumulative Gain

---

## MLflow Integration

### Experiment Tracking

#### [`ppera.log_mlflow`](../ppera/log_mlflow.py)

**Main Function:**
```python
def log_mlflow(
    dataset: str,
    top_k: pd.DataFrame,
    metrics: dict,
    num_rows: int,
    seed: int,
    model: object,
    model_type: str,
    params: dict,
    data: pd.DataFrame,
    train: pd.DataFrame,
    tf: object = None,
    vectors_tokenized: object = None,
    privacy: bool = None,
    fraction_to_hide: float = None,
    personalization: bool = None,
    fraction_to_change: float = None,
) -> None
```

**Experiment Organization:**
- **CF Experiments**: `"MLflow Collaborative Filtering"`
- **CBF Experiments**: `"MLflow Content Based Filtering"`  
- **RL Experiments**: `"MLflow Reinforcement Learning"`

**Logged Artifacts:**
- Dataset CSV files
- Model signatures (when possible)
- Input examples
- Hyperparameters
- All computed metrics

---

## Utility Functions

### Data Manipulation

#### [`ppera.data_modifier`](../ppera/data_modifier.py)

**Privacy Functions:**
```python
def hide_data_in_dataframe(data, hide_type, columns_to_hide, fraction_to_hide, records_to_hide, seed)
```

**Personalization Functions:**
```python
def change_items_in_dataframe(all, data, fraction_to_change, change_rating, seed)
```

### RL-Specific Utilities

#### [`ppera.rl_utils`](../ppera/rl_utils.py)

**Constants:**
```python
# Dataset paths
DATASET_DIR = {
    "movielens": "./datasets/movielens",
    "amazonsales": "./datasets/amazonsales", 
    "postrecommendations": "./datasets/PostRecommendations",
}

# Temporary directories for RL models
TMP_DIR = {
    "movielens": "ppera/rl_tmp/Movielens",
    "amazonsales": "ppera/rl_tmp/AmazonSales",
    "postrecommendations": "ppera/rl_tmp/PostRecommendations",
}
```

**File I/O Functions:**
```python
def save_embed(dataset: str, embed: dict) -> None
def load_embed(dataset: str) -> dict
def save_kg(dataset: str, kg: dict) -> None  
def load_kg(dataset: str) -> dict
def save_labels(dataset: str, labels: dict, mode: str = "train") -> None
def load_labels(dataset: str, mode: str = "train") -> dict
```

**Knowledge Graph Relations:**
```python
KG_RELATION = {
    USERID: {WATCHED: ITEMID, RATED: TITLE},
    ITEMID: {WATCHED: USERID, DESCRIBED_AS: TITLE, BELONG_TO: GENRES, RATING_VALUE_FOR_ITEM: RATING},
    TITLE: {DESCRIBED_AS: ITEMID, RATED: USERID},
    GENRES: {BELONG_TO: ITEMID},
    RATING: {RATING_VALUE_FOR_ITEM: ITEMID},
}
```

---

## Environment & State Management

### RL Environment

#### [`ppera.rl_kg_env`](../ppera/rl_kg_env.py)

```python
class KGState(object):
    def __init__(self, embed_size: int, history_len: int = 1)
    
    def __call__(self, user_embed, node_embed, last_node_embed, last_relation_embed, 
                 older_node_embed, older_relation_embed) -> np.ndarray
```

**State Representations:**
- **History Length 0**: `[user_embed, node_embed]`
- **History Length 1**: `[user_embed, node_embed, last_node_embed, last_relation_embed]`
- **History Length 2**: `[user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed]`

```python
class BatchKGEnvironment:
    def __init__(self, dataset: str, max_acts: int, max_path_len: int = 3, state_history: int = 1)
```

---

## Error Handling & Robustness

### Metric Calculation Safety

All algorithm implementations include comprehensive error handling:

```python
def format_metric(metric):
    return f"{metric:.4f}" if isinstance(metric, (float, int)) else "N/A"

# Example usage in CF.py
try:
    eval_precision = precision_at_k(test, top_k, col_user="userID", col_item="itemID", 
                                  col_rating="rating", col_prediction="prediction", k=1)
except Exception as e:
    eval_precision = None
    print(f"Error calculating precision: {e}")
```

### Path Validation (RL)

```python
# In batch_beam_search
if current_node_type not in KG_RELATION or relation not in KG_RELATION[current_node_type]:
    print(f"Warning: Invalid relation '{relation}' for node type '{current_node_type}'. Skipping path extension.")
    continue
```

---

## Configuration & Hyperparameters

### Default Values

**CF (Collaborative Filtering):**
- `NUM_FACTORS = 100`
- `NUM_EPOCHS = 100`  
- `learning_rate = 0.01`
- `lambda_reg = 0.001`

**CBF (Content-Based):**
- `tokenization_method = "bert"`
- Primary feature: `genres`

**RL (Reinforcement Learning):**
- `gamma = 0.99`
- `hidden_sizes = [512, 256]`
- `max_path_len = 3`
- `state_history = 1`
- `topk = [25, 5, 1]` for beam search

### Column Mappings

```python
header = {
    "col_user": "userID",
    "col_item": "itemID", 
    "col_rating": "rating",
    "col_timestamp": "timestamp",
    "col_title": "title",
    "col_genres": "genres",
    "col_year": "year", 
    "col_prediction": "prediction",
}
```

---

## Usage Examples

### Running Single Algorithm

```python
from ppera.CF import cf_experiment_loop

cf_experiment_loop(
    TOP_K=10,
    dataset="movielens",
    want_col=["userID", "itemID", "rating", "timestamp", "title", "genres"],
    num_rows=1000,
    ratio=0.75,
    seed=42,
    privacy=True,
    fraction_to_hide=0.3,
    columns_to_hide=["rating"]
)
```

### Custom Metric Calculation

```python
from ppera.metrics import precision_at_k, personalization_score

# Calculate precision@10
precision = precision_at_k(test_data, predictions, k=10)

# Calculate personalization score
pers_score = personalization_score(test_data, predictions)
```

### Knowledge Graph Operations

```python
from ppera.rl_utils import load_embed, save_embed
from ppera.rl_preprocess import create_processed_dataset

# Load pre-trained embeddings
embeddings = load_embed("movielens")

# Process dataset for knowledge graph
processed_data = create_processed_dataset(df)
```

This API documentation covers all the major components and functions in your PPERA framework, providing developers with the information needed to understand, extend, and use the system effectively.