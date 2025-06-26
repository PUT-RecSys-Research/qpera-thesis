# API Reference

This document provides detailed API documentation for the QPERA project components. It is generated based on the source code to ensure accuracy.

---

## 1. Main Entry Point

### [`qpera.main`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/main.py)

This is the central orchestration module for running all experiments. It parses command-line arguments to select and execute predefined experiment configurations.

**Key Function:**
```python
def main()
```
The main CLI entry point that parses arguments (`--algo`, `--dataset`, `--privacy`, etc.) to select and run experiment configurations from the `EXPERIMENT_CONFIGS` list.

**Experiment Configuration:**
The core of this module is the `EXPERIMENT_CONFIGS` list, which defines the matrix of experiments to be run. Each entry is a dictionary specifying the algorithm, module, function, and dataset.

```python
# Located in qpera/main.py
EXPERIMENT_CONFIGS = [
    {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "movielens"},
    {"algo": "CF", "module": CF, "func": "cf_experiment_loop", "dataset": "movielens"},
    {"algo": "RL", "module": RL, "func": "rl_experiment_loop", "dataset": "movielens", "rows": 14000},
    # ... and more combinations for amazonsales and postrecommendations
]
```

---

## 2. Algorithm Implementations

This section details the core recommendation algorithm experiment loops.

### Collaborative Filtering

#### [`qpera.CF`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/CF.py)

Implements the collaborative filtering experiment loop using the Cornac library.

**Main Function:**
```python
def cf_experiment_loop(
    TOP_K: int,
    dataset: str,
    want_col: list,
    # ... and other parameters for privacy/personalization
) -> None
```
- **Core Model**: `cornac.models.BPR` (Bayesian Personalized Ranking).
- **Hyperparameters**: `k=100` (factors), `max_iter=100`, `learning_rate=0.01`.
- **Metrics Computed**: Includes precision, recall, F1, MRR, MAE, RMSE, NDCG, coverage, and personalization scores.

### Content-Based Filtering

#### [`qpera.CBF`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/CBF.py)

Implements the content-based filtering experiment loop.

**Main Function:**
```python
def cbf_experiment_loop(
    TOP_K: int,
    dataset: str,
    want_col: list,
    # ... (same parameters as cf_experiment_loop)
) -> None
```
- **Core Model**: A custom `TfidfRecommender` that uses TF-IDF vectorization on item features.
- **Features**: Primarily uses the `genres` column for similarity calculation.

### Reinforcement Learning

The RL implementation is distributed across several modules, orchestrated by `RL.py`.

#### [`qpera.RL`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/RL.py)

This module contains the main experiment loop for the Reinforcement Learning approach.

**Main Function:**
```python
def rl_experiment_loop(
    TOP_K: int,
    dataset: str,
    want_col: list,
    # ... (same parameters as cf_experiment_loop)
) -> None
```
- **Orchestration**: This function coordinates the entire RL pipeline: data preprocessing, knowledge graph creation, TransE model training, agent training, and evaluation.

---

## 3. Data Processing

### Dataset Loading

#### [`qpera.datasets_loader`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/datasets_loader.py)

Handles loading and basic preprocessing of datasets.

**Core Function:**
```python
def loader(dataset: str, want_col: list, num_rows: int = None, seed: int = 42) -> pd.DataFrame
```
- **Supported Datasets**: `"movielens"`, `"amazonsales"`, `"postrecommendations"`.
- **Functionality**: Loads data from CSV, samples if `num_rows` is specified, and selects columns based on `want_col`.

### Data Manipulation

#### [`qpera.data_manipulation`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/data_manipulation.py)

Contains functions to apply privacy and personalization transformations to the data.

**Core Functions:**
```python
def hide_data_in_dataframe(data, hide_type, columns_to_hide, fraction_to_hide, records_to_hide, seed)
```
Simulates privacy scenarios by hiding or altering data.

```python
def change_items_in_dataframe(all, data, fraction_to_change, change_rating, seed)
```
Simulates personalization scenarios by modifying user interaction data.

---

## 4. Evaluation & Tracking

### Metrics

#### [`qpera.metrics`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/metrics.py)

A collection of custom and third-party evaluation metrics.

**Accuracy & Ranking Metrics:**
- `precision_at_k`, `recall_at_k`, `f1`, `mrr`, `ndcg_at_k`

**Coverage & Diversity Metrics:**
- `user_coverage`, `item_coverage`
- `personalization` (based on Jaccard similarity)
- `intra_list_similarity`

**Error Metrics:**
- `mae`, `rmse` (from Microsoft Recommenders)

### MLflow Integration

#### [`qpera.log_mlflow`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/log_mlflow.py)

Handles all logging of experiments to the MLflow Tracking server.

**Main Function:**
```python
def log_mlflow(
    dataset: str,
    top_k: pd.DataFrame,
    metrics: dict,
    # ... many other parameters for logging context
) -> None
```
- **Experiment Naming**: Organizes runs into experiments like `"MLflow Collaborative Filtering"`.
- **Logged Artifacts**: Logs parameters, all computed metrics, model files, and dataset samples to ensure full reproducibility.
- **Representative Sampling**: Uses a `_create_representative_sample` function to log a stratified sample of the data for inspection.

---

## 5. Reinforcement Learning Components

This section details the modules specific to the RL-based recommendation approach.

### Knowledge Graph Utilities

#### [`qpera.rl_utils`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_utils.py)

Provides constants and helper functions for the RL pipeline.

**Key Constants:**
- `DATASET_DIR`, `TMP_DIR`: Define file paths for datasets and temporary RL artifacts.
- `KG_RELATION`: A dictionary defining the structure and valid connections between entities in the knowledge graph.
- `PATH_PATTERN`: Defines valid multi-hop paths for generating recommendations and explanations.
- Entity and relation name constants (`USERID`, `WATCHED`, etc.).

**Key Functions:**
- `save_embed`, `load_embed`: Save/load trained embeddings.
- `save_kg`, `load_kg`: Save/load the constructed knowledge graph.
- `save_labels`, `load_labels`: Save/load user-item interaction labels for training/testing.
- `cleanup_dataset_files`: Removes temporary files generated during the RL pipeline.

### RL Environment

#### [`qpera.rl_kg_env`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_kg_env.py)

Defines the reinforcement learning environment built on the knowledge graph.

**Core Classes:**
- **`KGState`**: A class to construct the state representation for the agent, combining user embeddings with path history.
- **`BatchKGEnvironment`**: Manages the agent's interaction with the knowledge graph, including state transitions and rewards, for a batch of users.

### RL Agent & Evaluation

#### [`qpera.rl_test_agent`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_test_agent.py)

Contains the core PPO agent and evaluation logic.

**Core Classes & Functions:**
- **`ActorCritic(nn.Module)`**: The policy and value network for the PPO agent.
- **`batch_beam_search(...)`**: Performs beam search to generate recommendation paths in the knowledge graph.
- **`run_evaluation(...)`**: Evaluates the generated paths against test data and computes metrics.

<!-- 
  TODO: This section can be expanded by the authors with details about other RL modules, such as:
  - `qpera.rl_train_agent`: The main training loop for the PPO agent.
  - `qpera.rl_train_knowledge_graph`: The training loop for the TransE knowledge graph embeddings.
-->