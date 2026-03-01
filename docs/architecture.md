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
│   ├── main.py         # ExperimentRunner class and experiment orchestrator
│   │
│   ├── datasets/       # 💾 Processed & cached datasets (e.g., merge_file.csv)
│   │
│   ├── rl_tmp/         # ⚙️ Cached artifacts for the RL pipeline (e.g., kg.pkl)
│   │
│   ├── # --- Data Handling ---
│   ├── datasets_loader.py          # Unified dataset loading with caching
│   ├── datasets_downloader.py      # KaggleDatasetDownloader for auto-download
│   ├── data_manipulation.py        # Privacy/personalization data transformations
│   ├── frequency_based_rating_gen.py  # Synthetic rating generation for PostRecommendations
│   ├── rating_timestamp_gen.py     # Timestamp and rating generation for AmazonSales
│   │
│   ├── # --- Core Algorithms ---
│   ├── CBF.py          # Content-Based Filtering experiment loop
│   ├── CF.py           # Collaborative Filtering (BPR) experiment loop
│   ├── RL.py           # Reinforcement Learning orchestrator
│   │
│   ├── # --- Reinforcement Learning Components ---
│   ├── rl_preprocess.py            # KG entity/relation extraction
│   ├── rl_knowledge_graph.py       # KnowledgeGraph class
│   ├── rl_kg_env.py                # KGState and BatchKGEnvironment
│   ├── rl_transe_model.py          # KnowledgeEmbedding (TransE) model
│   ├── rl_train_transe_model.py    # TransE/KGE training loop
│   ├── rl_train_agent.py           # ActorCritic agent and PPO training
│   ├── rl_test_agent.py            # Beam search inference and evaluation
│   ├── rl_decoder.py               # RLRecommenderDecoder
│   ├── rl_prediction.py            # Prediction and scoring
│   └── rl_utils.py                 # I/O, TF-IDF, logging, seed utilities
│   │
│   ├── # --- Evaluation & Tracking ---
│   ├── metrics.py      # All evaluation metrics
│   └── log_mlflow.py   # MLflow experiment logging
│
├── references/         # 📄 Research papers, articles, and reference materials
│
└── reports/            # 📊 Generated analysis, figures, and results
    └── plots/          # 📈 Visualizations (e.g., precision/recall plots)
```

---

## 2. Core Components

This section details the key modules within the `qpera/` source directory.

### Main Entry Point ([`qpera/main.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/main.py))
The central orchestration module built around the `ExperimentRunner` class. It defines all algorithm–dataset combinations in `_define_experiments()` and generates scenario configurations (Clean, Privacy, Personalization) in `_build_configurations()`. The script runs all experiments without requiring command-line arguments.

### Dataset Loading ([`qpera/datasets_loader.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/datasets_loader.py))
Implements a unified, class-based system for loading and preprocessing datasets.
- **`BaseDatasetLoader`**: An abstract base class defining the loading interface.
- **Concrete Loaders**: `MovieLensDataset`, `AmazonSalesDataset`, and `PostRecommendationsDataset` handle the specifics of each data source, including column normalization, data cleaning, and merging.

### Algorithm Implementations
Each algorithm is encapsulated in its own module with a consistent experiment loop function.

- **Collaborative Filtering ([`qpera/CF.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/CF.py))**: Implements a BPR (Bayesian Personalized Ranking) model using the Cornac library.
- **Content-Based Filtering ([`qpera/CBF.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/CBF.py))**: Implements a `TfidfRecommender` using item features (e.g., genres) and cosine similarity.
- **Reinforcement Learning ([`qpera/RL.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/RL.py))**: Orchestrates the complex, multi-stage RL pipeline, including knowledge graph creation, model training, and evaluation.

### Reinforcement Learning Pipeline
The RL approach is broken down into several specialized modules.

- **Preprocessing ([`qpera/rl_preprocess.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_preprocess.py))**: Extracts entities (users, items, genres) and relations from the raw data to build a knowledge graph.
- **KG Environment ([`qpera/rl_kg_env.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_kg_env.py))**: Defines the `BatchKGEnvironment` where the agent interacts with the knowledge graph, managing state transitions and rewards.
- **TransE Model ([`qpera/rl_transe_model.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_transe_model.py))**: An implementation of the TransE algorithm to learn low-dimensional embeddings for entities and relations in the knowledge graph.
- **Agent ([`qpera/rl_train_agent.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_train_agent.py))**: Contains the `ActorCritic` model (PPO agent) that learns a policy for navigating the knowledge graph to find recommendations.
- **Inference ([`qpera/rl_test_agent.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rl_test_agent.py))**: Uses a `batch_beam_search` function to generate recommendation paths from the trained agent and knowledge graph.

### Data Manipulation ([`qpera/data_manipulation.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/data_manipulation.py))
Provides functions to simulate different scenarios for robustness and personalization testing.
- **`hide_information_in_dataframe`**: Simulates privacy-preserving scenarios with four modes: hiding entire columns, removing random records, removing specific records, or replacing values within columns with NaN.
- **`change_items_in_dataframe`**: Simulates personalization shifts by replacing a fraction of each user's items with others drawn from a global item frequency distribution, updating metadata and ratings accordingly.

### Data Generation
- **[`qpera/rating_timestamp_gen.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/rating_timestamp_gen.py)**: Generates timestamps (randomly within 2018–2022) and ratings (via VADER sentiment analysis) for the AmazonSales dataset.
- **[`qpera/frequency_based_rating_gen.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/frequency_based_rating_gen.py)**: Generates synthetic ratings for the PostRecommendations dataset based on user interaction frequency with categories.

### Evaluation Framework
- **Metrics ([`qpera/metrics.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/metrics.py))**: A comprehensive collection of metrics including:
    - Accuracy: `precision_at_k`, `recall_at_k`, `f1_score`, `accuracy`
    - Error: `mae`, `rmse` (from Microsoft Recommenders)
    - Ranking: `mrr`
    - Coverage: `user_coverage`, `item_coverage`
    - Diversity: `intra_list_dissimilarity`, `intra_list_similarity_score`, `personalization_score`
- **MLflow Logging ([`qpera/log_mlflow.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/log_mlflow.py))**: A centralized function to log all experiment parameters, metrics, and artifacts (including representative data samples) to MLflow, ensuring reproducibility.

---

## 3. Data Flow Architecture

The project follows distinct data flows for standard experiments and the RL pipeline.

**1. General Experiment Flow**
```
ExperimentRunner Configuration → Main Orchestrator → Dataset Loading → Data Manipulation (Privacy/Personalization) → Algorithm Training & Prediction → Evaluation → MLflow Logging
```

**2. RL Pipeline Flow**
```
Raw DataFrame → KG Preprocessing → TransE Embedding Training → Agent Policy Training → Beam Search Inference → Path-based Recommendations → Evaluation
```
---

## 4. Key Design Patterns

The project employs several key design patterns to promote modularity and robustness.

- **Modular Algorithm Interface**: All main algorithm loops (`cf_experiment_loop`, etc.) share a consistent function signature, allowing the main orchestrator to call them interchangeably.
- **Configuration-Driven Experiments**: Experiments are defined within the `ExperimentRunner` class — algorithm–dataset combinations in `_define_experiments()` and scenario parameters in `_build_configurations()` — making it easy to add or modify test runs without changing the core logic.
- **Defensive Programming**: Metric calculations are wrapped in `try...except` blocks to prevent a single failure from halting an entire experiment batch.
- **Caching and Persistence**: The RL pipeline extensively caches intermediate artifacts (processed data, knowledge graphs, trained embeddings) to speed up subsequent runs and debugging.

