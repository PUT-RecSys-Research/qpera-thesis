# API Reference

This document provides detailed API documentation for the QPERA project components, verified against the source code.

---

## 1. Main Entry Point

### [`qpera.main`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/main.py)

This is the central orchestration module for running all experiments. It does not use command-line arguments — instead, the `ExperimentRunner` class manages all experiment definitions and configurations internally.

**Key Class: `ExperimentRunner`**

```python
class ExperimentRunner:
    """Manages and executes recommendation system experiments with various configurations."""

    def _define_experiments(self) -> List[Dict[str, Any]]:
        """Define all algorithm-dataset combinations (9 total: 3 algos × 3 datasets)."""

    def _build_configurations(self) -> List[Dict[str, Any]]:
        """Build all scenario configurations: 1 Clean + 4 Privacy + 4 Personalization."""

    def run_all_experiments(self) -> None:
        """Execute all experiments across all configurations."""
```

**Experiment Definitions:**
```python
# 9 algorithm-dataset combinations defined in _define_experiments()
[
    {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "movielens", "rows": 14000},
    {"algo": "CF",  "module": CF,  "func": "cf_experiment_loop",  "dataset": "movielens", "rows": 14000},
    {"algo": "RL",  "module": RL,  "func": "rl_experiment_loop",  "dataset": "movielens", "rows": 14000},
    {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "amazonsales"},
    {"algo": "CF",  "module": CF,  "func": "cf_experiment_loop",  "dataset": "amazonsales"},
    {"algo": "RL",  "module": RL,  "func": "rl_experiment_loop",  "dataset": "amazonsales"},
    {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "postrecommendations", "rows": 14000},
    {"algo": "CF",  "module": CF,  "func": "cf_experiment_loop",  "dataset": "postrecommendations", "rows": 14000},
    {"algo": "RL",  "module": RL,  "func": "rl_experiment_loop",  "dataset": "postrecommendations", "rows": 14000},
]
```

**Base Parameters:**
```python
# Common parameters from _get_base_params()
{"TOP_K": 10, "want_col": ["userID", "itemID", "rating", "timestamp", "title", "genres"], "ratio": 0.75, "seed": 42}
```

---

## 2. Algorithm Implementations

This section details the core recommendation algorithm experiment loops. All three share a consistent function signature for interchangeable use by the `ExperimentRunner`.

### Collaborative Filtering

#### [`qpera.CF`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/CF.py)

Implements the collaborative filtering experiment loop using Cornac BPR (Bayesian Personalized Ranking).

**Main Function:**
```python
def cf_experiment_loop(
    TOP_K: int,
    dataset: str,
    want_col: list,
    ratio: float,
    seed: int,
    num_rows: int = None,
    privacy: bool = False,
    hide_type: str = None,
    columns_to_hide: list = None,
    fraction_to_hide: float = None,
    personalization: bool = False,
    fraction_to_change: float = None,
    change_rating: bool = False,
) -> None
```
- **Core Model**: `cornac.models.BPR` (Bayesian Personalized Ranking).
- **Key Characteristic**: Operates exclusively on the user-item interaction matrix — blind to item content metadata.

### Content-Based Filtering

#### [`qpera.CBF`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/CBF.py)

Implements the content-based filtering experiment loop using TF-IDF vectorization.

**Main Function:**
```python
def cbf_experiment_loop(
    TOP_K: int,
    dataset: str,
    want_col: list,
    ratio: float,
    seed: int,
    num_rows: int = None,
    privacy: bool = False,
    hide_type: str = None,
    columns_to_hide: list = None,
    fraction_to_hide: float = None,
    personalization: bool = False,
    fraction_to_change: float = None,
    change_rating: bool = False,
) -> None
```
- **Core Model**: A TF-IDF-based recommender that merges all descriptive data (title, genres) into a single text field for vectorization and cosine similarity.

### Reinforcement Learning

#### [`qpera.RL`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/RL.py)

This module contains the main experiment loop for the PGPR-based Reinforcement Learning approach.

**Main Function:**
```python
def rl_experiment_loop(
    TOP_K: int,
    dataset: str,
    want_col: list,
    ratio: float,
    seed: int,
    num_rows: int = None,
    privacy: bool = False,
    hide_type: str = None,
    columns_to_hide: list = None,
    fraction_to_hide: float = None,
    personalization: bool = False,
    fraction_to_change: float = None,
    change_rating: bool = False,
) -> None
```
- **Orchestration**: Coordinates the multi-stage RL pipeline: preprocessing → KG creation → TransE training → agent training → beam search inference → evaluation.

---

## 3. Data Processing

### Dataset Loading

#### [`qpera.datasets_loader`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/datasets_loader.py)

Handles loading and basic preprocessing of datasets with caching support.

**Core Function:**
```python
def loader(
    dataset_name: str = "movielens",
    want_col: Optional[List[str]] = None,
    num_rows: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame
```
- **Supported Datasets**: `"movielens"`, `"amazonsales"`, `"postrecommendations"`.
- **Caching**: Processed data is cached as `merge_file.csv`; subset-specific caches include row count and seed in the filename.

**Dataset Classes:**

- `BaseDatasetLoader(ABC)` — Abstract base class with `load_dataset()` and `merge_datasets()`.
- `MovieLensDataset` — Loads and processes MovieLens 20M data.
- `AmazonSalesDataset` — Loads Amazon Sales data with VADER-generated ratings.
- `PostRecommendationsDataset` — Loads post data with frequency-based generated ratings.

### Data Manipulation

#### [`qpera.data_manipulation`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/data_manipulation.py)

Contains functions to apply privacy and personalization transformations to the data.

**Privacy Function:**
```python
def hide_information_in_dataframe(
    data: pd.DataFrame,
    hide_type: str = "columns",
    columns_to_hide: Union[str, List[str]] = None,
    fraction_to_hide: float = 0.0,
    records_to_hide: List[int] = None,
    seed: int = 42,
) -> pd.DataFrame
```
Supports four hiding modes:

- `"columns"` — Remove entire columns.
- `"records_random"` — Remove a random fraction of rows.
- `"records_selective"` — Remove specific rows by index.
- `"values_in_column"` — Replace random values within columns with NaN.

**Personalization Function:**
```python
def change_items_in_dataframe(
    all: pd.DataFrame,
    data: pd.DataFrame,
    fraction_to_change: float = 0.0,
    change_rating: bool = False,
    seed: int = 42,
) -> pd.DataFrame
```
Replaces a fraction of each user's items with new items drawn from a global item frequency distribution. Updates metadata (title, genres) and optionally replaces ratings with the new item's average rating.

### Data Generation

#### [`qpera.rating_timestamp_gen`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rating_timestamp_gen.py)
Generates timestamps (randomly within 2018–2022) and ratings via VADER sentiment analysis for the AmazonSales dataset.

#### [`qpera.frequency_based_rating_gen`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/frequency_based_rating_gen.py)
Generates synthetic ratings for PostRecommendations based on user interaction frequency with post categories.

#### [`qpera.datasets_downloader`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/datasets_downloader.py)
`KaggleDatasetDownloader` class for automated dataset downloads from Kaggle.

---

## 4. Evaluation & Tracking

### Metrics

#### [`qpera.metrics`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/metrics.py)

A collection of custom and third-party evaluation metrics.

**Accuracy Metrics:**

- `precision_at_k(rating_true, rating_pred, ..., k=10) -> float` — Binary precision: 1 if any relevant item in top-k, 0 otherwise, averaged across users.
- `recall_at_k(rating_true, rating_pred, ..., k=10) -> float` — Average proportion of relevant items in top-k per user.
- `f1_score(rating_true, rating_pred, ..., k=1) -> float` — Harmonic mean of precision and recall at k.
- `accuracy(rating_true, rating_pred, ...) -> float` — Compares rounded predictions with actual ratings.

**Ranking Metrics:**

- `mrr(rating_true, rating_pred, ..., k=1) -> float` — Mean Reciprocal Rank.

**Error Metrics:**

- `mae`, `rmse` — From Microsoft Recommenders library.

**Coverage Metrics:**

- `user_coverage(rating_true, rating_pred, ..., threshold) -> float` — Percentage of users with at least one meaningful recommendation (within threshold).
- `item_coverage(rating_true, rating_pred, ..., threshold) -> float` — Percentage of items with at least one meaningful recommendation.

**Diversity Metrics:**

- `intra_list_similarity(predicted, feature_df) -> float` — Average cosine similarity within recommendation lists (lower = more diverse).
- `intra_list_dissimilarity(item_features, rating_pred, ...) -> float` — 1 − intra-list similarity (higher = more diverse).
- `intra_list_similarity_score(item_features, rating_pred, ...) -> float` — DataFrame-based version with automatic feature vectorization.
- `personalization(predicted) -> float` — Inter-diversity: 1 − average cosine similarity between users' recommendation lists.
- `personalization_score(rating_true, rating_pred, ...) -> float` — DataFrame interface to the `personalization` function.

### MLflow Integration

#### [`qpera.log_mlflow`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/log_mlflow.py)

Handles all logging of experiments to the MLflow Tracking server.

**Main Function:**
```python
def log_mlflow(
    dataset: str,
    top_k: pd.DataFrame,
    metrics: Dict[str, Any],
    num_rows: Optional[int],
    seed: int,
    model: Any,
    model_type: str,
    params: Dict[str, Any],
    data: pd.DataFrame,
    train: pd.DataFrame,
    tf: Any = None,
    vectors_tokenized: Any = None,
    privacy: Optional[bool] = None,
    fraction_to_hide: Optional[float] = None,
    personalization: Optional[bool] = None,
    fraction_to_change: Optional[float] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> None
```
- **Logged Data**: Parameters, all computed metrics, model files, and a representative stratified sample of the data.

---

## 5. Reinforcement Learning Components

This section details the modules specific to the RL-based recommendation approach.

### Knowledge Graph Utilities

#### [`qpera.rl_utils`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_utils.py)

Provides constants and helper functions for the RL pipeline.

**Key Constants:**

- `DATASET_DIR`, `TMP_DIR`: Define file paths for datasets and temporary RL artifacts.
- `KG_RELATION`: Defines the structure and valid connections between entities in the knowledge graph.
- `PATH_PATTERN`: Defines valid multi-hop paths for generating recommendations and explanations.

**Key Functions:**

- `save_embed`, `load_embed`: Save/load trained embeddings.
- `save_kg`, `load_kg`: Save/load the constructed knowledge graph.
- `save_labels`, `load_labels`: Save/load user-item interaction labels.

### Knowledge Graph

#### [`qpera.rl_knowledge_graph`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_knowledge_graph.py)

`KnowledgeGraph` class for building and managing the knowledge graph structure.

### RL Environment

#### [`qpera.rl_kg_env`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_kg_env.py)

Defines the reinforcement learning environment built on the knowledge graph.

- **`KGState`**: Constructs the state representation for the agent, combining user embeddings with path history.
- **`BatchKGEnvironment`**: Manages state transitions and rewards for a batch of users interacting with the knowledge graph.

### TransE Model

#### [`qpera.rl_transe_model`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_transe_model.py)

`KnowledgeEmbedding` — An implementation of the TransE algorithm for learning entity and relation embeddings.

#### [`qpera.rl_train_transe_model`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_train_transe_model.py)

Training loop for the TransE knowledge graph embedding model.

### RL Agent

#### [`qpera.rl_train_agent`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_train_agent.py)

Contains the `ActorCritic` model (PPO agent) and the training loop for learning a navigation policy over the knowledge graph.

#### [`qpera.rl_test_agent`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_test_agent.py)

- **`batch_beam_search(...)`**: Performs beam search to generate recommendation paths.
- **`run_evaluation(...)`**: Evaluates generated paths against test data.

### RL Inference

#### [`qpera.rl_decoder`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_decoder.py)

`RLRecommenderDecoder` — Decodes RL agent paths into recommendation lists.

#### [`qpera.rl_prediction`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_prediction.py)

Prediction and scoring utilities for the RL pipeline.

#### [`qpera.rl_preprocess`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_preprocess.py)

Preprocessing module that extracts entities (users, items, genres) and relations from raw data to build the knowledge graph structure.
