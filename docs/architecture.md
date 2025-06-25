# Project Architecture

This document explains the codebase structure, core components, and design patterns of the **QPERA** project.

## 1. Project Structure

The repository is organized using a structure inspired by the Cookiecutter Data Science template to ensure reproducibility and maintainability.

```
.
├── Makefile            # Convenience commands for setup, testing, and execution
├── README.md           # Main project documentation for GitHub
├── environment.yml     # Conda environment specification
├── mkdocs.yml          # Configuration for the documentation site
├── pyproject.toml      # Python project configuration (PEP 621)
│
├── datasets/           # 🗄️ Raw datasets downloaded from sources (e.g., Kaggle)
│   ├── MovieLens/
│   ├── AmazonSales/
│   └── PostRecommendations/
│
├── docs/               # 📚 Project documentation source files
│   ├── api.md
│   ├── architecture.md
│   └── ...
│
├── qpera/              # 🐍 Main source code package
│   ├── __init__.py
│   ├── main.py         # Main CLI entry point and experiment orchestrator
│   │
│   ├── datasets/       # 💾 Processed & cached datasets (e.g., merge_file.csv)
│   │
│   ├── rl_tmp/         # ⚙️ Cached artifacts for the RL pipeline (e.g., kg.pkl)
│   │
│   ├── # --- Data Handling ---
│   ├── datasets_loader.py
│   ├── data_manipulation.py
│   │
│   ├── # --- Core Algorithms ---
│   ├── CBF.py
│   ├── CF.py
│   ├── RL.py           # Reinforcement Learning orchestrator
│   │
│   ├── # --- Reinforcement Learning Components ---
│   ├── rl_preprocess.py
│   ├── rl_knowledge_graph.py
│   ├── rl_kg_env.py
│   ├── rl_transe_model.py
│   ├── rl_train_agent.py
│   ├── rl_test_agent.py
│   └── rl_utils.py
│   │
│   ├── # --- Utilities & Tooling ---
│   ├── metrics.py
│   └── log_mlflow.py
│
├── references/         # 📄 Research papers, articles, and reference materials
│
└── reports/            # 📊 Generated analysis, figures, and results
    └── plots/          # 📈 Visualizations (e.g., precision/recall plots)
```

---

## 2. Core Components

This section details the key modules within the `qpera/` source directory.

### Main Entry Point ([`qpera/main.py`](../qpera/main.py))
The central orchestration script that manages experiment execution based on command-line arguments. It iterates through predefined experiment configurations to run tests for different algorithms, datasets, and scenarios (clear, privacy, personalization).

### Dataset Loading ([`qpera/datasets_loader.py`](../qpera/datasets_loader.py))
Implements a unified, class-based system for loading and preprocessing datasets.
- **`BaseDatasetLoader`**: An abstract base class defining the loading interface.
- **Concrete Loaders**: `MovieLensDataset`, `AmazonSalesDataset`, and `PostRecommendationsDataset` handle the specifics of each data source, including column normalization, data cleaning, and merging.

### Algorithm Implementations
Each algorithm is encapsulated in its own module with a consistent experiment loop function.

- **Collaborative Filtering ([`qpera/CF.py`](../qpera/CF.py))**: Implements a BPR (Bayesian Personalized Ranking) model using the Cornac library.
- **Content-Based Filtering ([`qpera/CBF.py`](../qpera/CBF.py))**: Implements a `TfidfRecommender` using item features (e.g., genres) and cosine similarity.
- **Reinforcement Learning ([`qpera/RL.py`](../qpera/RL.py))**: Orchestrates the complex, multi-stage RL pipeline, including knowledge graph creation, model training, and evaluation.

### Reinforcement Learning Pipeline
The RL approach is broken down into several specialized modules.

- **Preprocessing ([`qpera/rl_preprocess.py`](../qpera/rl_preprocess.py))**: Extracts entities (users, items, genres) and relations from the raw data to build a knowledge graph.
- **KG Environment ([`qpera/rl_kg_env.py`](../qpera/rl_kg_env.py))**: Defines the `BatchKGEnvironment` where the agent interacts with the knowledge graph, managing state transitions and rewards.
- **TransE Model ([`qpera/rl_transe_model.py`](../qpera/rl_transe_model.py))**: An implementation of the TransE algorithm to learn low-dimensional embeddings for entities and relations in the knowledge graph.
- **Agent ([`qpera/rl_train_agent.py`](../qpera/rl_train_agent.py))**: Contains the `ActorCritic` model (PPO agent) that learns a policy for navigating the knowledge graph to find recommendations.
- **Inference ([`qpera/rl_test_agent.py`](../qpera/rl_test_agent.py))**: Uses a `batch_beam_search` function to generate recommendation paths from the trained agent and knowledge graph.

### Data Manipulation ([`qpera/data_manipulation.py`](../qpera/data_manipulation.py))
Provides functions to simulate different scenarios for robustness testing.
- **`hide_information_in_dataframe`**: Simulates privacy attacks by removing or obscuring values in specified columns or entire records.
- **`change_items_in_dataframe`**: Simulates personalization shifts by substituting items in a user's history based on global popularity.

### Evaluation Framework
- **Metrics ([`qpera/metrics.py`](../qpera/metrics.py))**: A comprehensive collection of metrics, including accuracy (`precision_at_k`, `ndcg_at_k`), coverage (`user_coverage`), and diversity (`personalization`, `intra_list_similarity`).
- **MLflow Logging ([`qpera/log_mlflow.py`](../qpera/log_mlflow.py))**: A centralized function to log all experiment parameters, metrics, and artifacts to MLflow, ensuring reproducibility.

---

## 3. Data Flow Architecture

The project follows distinct data flows for standard experiments and the RL pipeline.

**1. General Experiment Flow**
```
CLI Arguments → Main Orchestrator → Dataset Loading → Data Manipulation (Privacy/Personalization) → Algorithm Training & Prediction → Evaluation → MLflow Logging
```

**2. RL Pipeline Flow**
```
Raw DataFrame → KG Preprocessing → TransE Embedding Training → Agent Policy Training → Beam Search Inference → Path-based Recommendations → Evaluation
```
<!--
  TODO: Authors could add a note here explaining the rationale behind choosing a multi-stage pipeline for RL instead of an end-to-end model.
-->

---

## 4. Key Design Patterns

The project employs several key design patterns to promote modularity and robustness.

- **Modular Algorithm Interface**: All main algorithm loops (`cf_experiment_loop`, etc.) share a consistent function signature, allowing the main orchestrator to call them interchangeably.
- **Configuration-Driven Experiments**: Experiments are defined in a central list of dictionaries (`EXPERIMENT_CONFIGS`), making it easy to add or modify test runs without changing the core logic.
- **Defensive Programming**: Metric calculations are wrapped in `try...except` blocks to prevent a single failure from halting an entire experiment batch.
- **Caching and Persistence**: The RL pipeline extensively caches intermediate artifacts (processed data, knowledge graphs, trained embeddings) to speed up subsequent runs and debugging.

<!--
  TODO: This section could be expanded with details on other patterns, such as the use of the Factory or Strategy pattern if applicable in the dataset loaders or algorithm selection.
-->
