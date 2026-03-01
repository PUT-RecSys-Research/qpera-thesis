# Datasets Guide

This document provides a comprehensive guide to the datasets used in the QPERA project, including setup, processing details, and instructions for adding new data sources.

## Overview

This project uses three main datasets to evaluate recommendation algorithms across domains representing varying levels of **decision-making consequences**.

| Dataset | Domain | Decision Cost | Records | Ratings | Kaggle Source |
|---------|--------|:------------:|--------:|:-------:|---------------|
| [Amazon Sales](#amazon-sales) | E-commerce | **High** | 1,351 | Generated (VADER) | `karkavelrajaj/amazon-sales-dataset` |
| [MovieLens 20M](#movielens) | Movies | **Moderate** | 20,000,263 | Explicit | `grouplens/movielens-20m-dataset` |
| [Post Recs](#post-recommendations) | Social Media | **Low** | 70,616 | Generated (frequency) | `vatsalparsaniya/post-pecommendation` |

!!! info "Decision-Making Context"
    The datasets were selected to represent a spectrum of user decision-making commitment: **AmazonSales** involves financial commitment (high-cost), **MovieLens** involves time investment (moderate-cost), and **PostRecommendations** involves minimal effort (low-cost). All experiments limit data to a maximum of **14,000 rows** per run.

---

## 1. Automatic Download & Setup

### Prerequisites
- **Kaggle Account**: You need a Kaggle account to download the datasets.
- **Kaggle API Token**: Download your `kaggle.json` file from your Kaggle account page.

### Setup Steps

1.  **Install the Kaggle API client**:
    ```bash
    pip install kaggle
    ```

2.  **Configure Kaggle Credentials**:
    Place your `kaggle.json` file in the `~/.kaggle/` directory. For automated setup (moves `kaggle.json` from `~/Downloads` to `~/.kaggle/`):
    ```bash
    make kaggle-autoconfig
    ```
    Or for manual setup instructions:
    ```bash
    make kaggle-setup-help
    ```

3.  **Download All Datasets**:
    Use the `Makefile` command to download all required datasets automatically.
    ```bash
    make download-datasets
    ```

4.  **Verify Downloads**:
    Check that all datasets were downloaded correctly.
    ```bash
    make verify-datasets
    ```

---

## 2. Dataset Details & Processing

The project uses a unified loading system ([`qpera/datasets_loader.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/datasets_loader.py)) that standardizes column names and applies specific preprocessing for each dataset.

### MovieLens
- **Source**: `grouplens/movielens-20m-dataset`
- **Raw Files**: `rating.csv`, `movie.csv`, `tag.csv`
- **Records**: 20,000,263
- **Key Processing Steps**:
    - Columns are mapped to standard names (e.g., `movieId` -> `itemID`).
    - Genres are converted from `Action|Adventure` to `Action Adventure`.
    - Timestamps are converted to a standard Unix format.
    - Duplicate user-item interactions are removed.
    - The dataset length is controlled by specifying the number of initial rows (up to 14,000) in the experiment configuration.

### Amazon Sales
- **Source**: `karkavelrajaj/amazon-sales-dataset`
- **Raw Files**: `amazon.csv`
- **Records**: 1,351
- **Key Processing Steps**:
    - Columns are mapped (e.g., `product_id` -> `itemID`).
    - `category` and `about_product` are combined to create a `genres` field.
    - **Ratings are generated** from sentiment analysis of reviews using the VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon.
    - **Timestamps are generated** randomly within the range from 2018 to 2022.
    - Unnecessary columns (e.g., `discounted_price`, `img_link`) are dropped.

### Post Recommendations
- **Source**: `vatsalparsaniya/post-pecommendation`
- **Raw Files**: `user_data.csv`, `view_data.csv`, `post_data.csv`
- **Records**: 70,616
- **Key Processing Steps**:
    - **Rating Generation**: This dataset lacks explicit ratings. They are generated using a **frequency-based algorithm** that uses categories to assess user interest.
    - Columns are mapped (e.g., `post_id` -> `itemID`, `category` -> `genres`).
    - User, post, and view data are merged into a single interaction table.

---

## 3. Data Loading & Caching

The `loader` function in [`qpera/datasets_loader.py`](https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/qpera/datasets_loader.py) provides a single, consistent interface for accessing all datasets.

### Caching Mechanism
To speed up repeated experiments, the loader uses a caching system:
- On the first load, raw files are processed and saved as a single `merge_file.csv` in `qpera/datasets/<DatasetName>/`.
- Subsequent loads read directly from this cached file.
- If you specify `num_rows`, a separate cached file is created (e.g., `merge_file_r14000_s42.csv`), allowing you to work with smaller subsets without reprocessing.

### Usage Example
```python
from qpera.datasets_loader import loader

# Load the full, cached MovieLens dataset
data = loader("movielens")

# Load a 14,000-row subset for faster RL experiments
data_subset = loader("movielens", num_rows=14000, seed=42)
```

---

## 4. Reinforcement Learning Data Pipeline

The Reinforcement Learning (RL) algorithm uses a separate, more complex data pipeline.
- **Input**: The same processed data from the `loader`.
- **Process**: It builds a knowledge graph by extracting entities (users, items, genres) and relations (watched, belongs_to).
- **Output & Cache**: The processed graph, embeddings, and labels are cached as `.pkl` files in the `qpera/rl_tmp/<DatasetName>/` directory. This cache is separate from the main dataset cache.

---

## 5. Adding a New Dataset

To integrate a new dataset into the project, follow these steps:

1.  **Create a Loader Class**: In `qpera/datasets_loader.py`, create a new class that inherits from `BaseDatasetLoader`. Implement the `_check_local_files_exist` and `merge_datasets` methods to handle your specific files and processing logic.
2.  **Register the Loader**: Add your new class to the `dataset_loaders` dictionary inside the `loader` function.
3.  **Add Downloader Support (Optional)**: In `qpera/datasets_downloader.py`, add your dataset's information to the `DATASET_CONFIG` dictionary to enable automatic downloads with `make download-datasets`.
4.  **Add RL Support (Optional)**: If the dataset should be used with the RL algorithm, update the path dictionaries (`DATASET_DIR`, `TMP_DIR`, `LABELS`) in `qpera/rl_utils.py`.

---

## 6. Troubleshooting

- **`FileNotFoundError`**: Ensure you have run `make download-datasets` to download all raw data.
- **Kaggle API `401 Unauthorized`**: Verify your `~/.kaggle/kaggle.json` file is correctly placed and has the right permissions (`chmod 600`).
- **RL Pipeline Errors**: If you encounter issues with the RL pipeline, try clearing its specific cache by deleting the `qpera/rl_tmp/<DatasetName>` directory and re-running the experiment.