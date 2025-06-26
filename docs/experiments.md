# Running Experiments

This guide explains how to run and configure experiments with different recommendation algorithms in the QPERA project.

## 1. Quick Commands

### Full Experiment Suite
To run all predefined experiments across all datasets and scenarios:
```bash
make run-all
```

### Viewing Results
All results are tracked with MLflow. Start the UI to view them:
```bash
make run-mlflow
```
Then, navigate to `http://127.0.0.1:8080` in your browser.

---

## 2. Experiment Configuration

This project uses a **configuration-driven** approach. Instead of passing many command-line arguments, experiments are defined in a list within [`qpera/main.py`](../qpera/main.py).

### Predefined Experiment Matrix
The `EXPERIMENT_CONFIGS` list defines the core combinations of algorithms and datasets to be tested:
```python
# Located in qpera/main.py
EXPERIMENT_CONFIGS = [
    {"algo": "CBF", "dataset": "movielens", "rows": 14000},
    {"algo": "CF", "dataset": "movielens", "rows": 14000},
    {"algo": "RL", "dataset": "movielens", "rows": 14000},
    # ... and more combinations for amazonsales and postrecommendations
]
```

### Experiment Scenarios
For each entry in the matrix, the main script runs three types of scenarios:

1.  **Clear (Baseline)**: The algorithm runs on the original, unmodified data.
2.  **Privacy**: Metadata is progressively hidden to test algorithm robustness. This is done by hiding a fraction (`0.1`, `0.25`, `0.5`, `0.8`) of values in the `title` and `genres` columns.
3.  **Personalization**: User preferences are shifted by replacing a fraction of their interactions with globally popular items to test how personalization is affected.

---

## 3. Available Algorithms

### Collaborative Filtering (CF)
- **Implementation**: Cornac BPR (Bayesian Personalized Ranking).
- **Key Hyperparameters**: `k=100` (factors), `max_iter=100`, `learning_rate=0.01`.
- **Best for**: Datasets with strong user-item interaction signals.

### Content-Based Filtering (CBF)
- **Implementation**: A custom `TfidfRecommender` using item metadata.
- **Features**: Primarily uses the `genres` column for TF-IDF vectorization and cosine similarity.
- **Best for**: Cold-start scenarios and datasets with rich item descriptions.

### Reinforcement Learning (RL)
- **Implementation**: A PGPR-based agent that navigates a knowledge graph.
- **Key Components**:
    - **Knowledge Graph**: Built from users, items, and their attributes.
    - **TransE Embeddings**: Learns representations for entities and relations.
    - **Actor-Critic Agent**: Learns a policy to find recommendation paths.
- **Best for**: Generating explainable, sequential recommendations.
- **Note**: Requires a multi-stage pipeline and is computationally intensive (GPU recommended).

---

## 4. Evaluation Metrics

The project computes a comprehensive set of metrics, all logged to MLflow.

- **Accuracy & Ranking**: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr`, `mae`, `rmse`.
- **Coverage & Diversity**: `user_coverage`, `item_coverage`, `personalization`, `intra_list_similarity`.
- **Robustness**: All metrics are calculated in `try...except` blocks to ensure that a failure in one metric does not stop the entire experiment.

---

## 5. Reinforcement Learning Pipeline

The RL algorithm follows a distinct, multi-stage pipeline orchestrated by [`qpera/RL.py`](../qpera/RL.py).

1.  **Preprocessing**: Extracts entities and relations from the raw data to build a knowledge graph structure.
2.  **TransE Training**: Trains a TransE model on the graph to learn embeddings for all entities and relations.
3.  **Policy Training**: Trains the Actor-Critic agent using PPO to learn how to navigate the graph.
4.  **Inference**: Uses a beam search algorithm to generate the top-k recommendation paths for each user in the test set.

<!--
  TODO: Authors can add a note here explaining the design choice for a multi-stage RL pipeline versus an end-to-end model, e.g., for modularity and debuggability.
-->

---

## 6. Caching & Performance

To speed up development and repeated runs, the project uses two caching systems:

- **Dataset Cache**: Processed and merged datasets are saved in `qpera/datasets/<DatasetName>/`. Subsequent loads read from this cache.
- **RL Artifact Cache**: The RL pipeline saves its intermediate artifacts (knowledge graphs, embeddings, labels) to `qpera/rl_tmp/<DatasetName>/`.

---

## 7. Troubleshooting

### MLflow Connection Issues
- **Problem**: MLflow UI is not accessible or experiments are not logging.
- **Solution**:
    ```bash
    # Ensure the server is running
    make run-mlflow

    # If it's corrupted, force-clear the local tracking data
    rm -rf mlruns/ mlflow.db
    ```

### RL Pipeline Fails
- **Problem**: The RL experiment fails, often during training or graph creation.
- **Solution**: The RL cache can sometimes become corrupted. Clear it for the specific dataset and retry.
    ```bash
    # Example for MovieLens
    rm -rf qpera/rl_tmp/MovieLens/
    make run-rl-movielens
    ```

### Memory Errors
- **Problem**: An experiment fails with an `OutOfMemoryError`.
- **Solution**: Reduce the dataset size for testing by editing the `rows` parameter in the `EXPERIMENT_CONFIGS` list in `qpera/main.py`.
    ```python
    # Example of reducing Amazon Sales for a quick test
    {"algo": "CF", "dataset": "amazonsales", "rows": 5000},
    ```