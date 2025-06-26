# Datasets Guide

This document provides a comprehensive guide to the datasets used in the QPERA project, including setup, processing details, and instructions for adding new data sources.

## Overview

This project uses three main datasets to evaluate recommendation algorithms across different domains.

| Dataset | Domain | Raw Size | Users | Items | Ratings | Kaggle Source |
|---------|--------|----------|-------|-------|---------|---------------|
| [MovieLens](#movielens) | Movies | ~20M | ~138K | ~27K | ~20M | `grouplens/movielens-20m-dataset` |
| [Amazon Sales](#amazon-sales) | E-commerce | ~1.4M | ~1M | ~200K | Generated | `karkavelrajaj/amazon-sales-dataset` |
| [Post Recs](#post-recommendations) | Social Media | ~150K | ~10K | ~50K | Generated | `vatsalparsaniya/post-pecommendation` |

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

2.  **Auto Configure Kaggle Credentials**:
    Places your `kaggle.json` file in the `~/.kaggle/` directory.
    ```bash
    make kaggle-autoconfig
    ```

3.  **Download All Datasets**:
    Use the `Makefile` command to download and extract all required datasets automatically.
    ```bash
    make check-datasets
    ```
    This command checks for existing files and only downloads what is missing.

---

## 2. Dataset Details & Processing

The project uses a unified loading system ([`qpera/datasets_loader.py`](../qpera/datasets_loader.py)) that standardizes column names and applies specific preprocessing for each dataset.

### MovieLens
- **Source**: `grouplens/movielens-20m-dataset`
- **Raw Files**: `rating.csv`, `movie.csv`, `tag.csv`
- **Key Processing Steps**:
    - Columns are mapped to standard names (e.g., `movieId` -> `itemID`).
    - Genres are converted from `Action|Adventure` to `Action Adventure`.
    - Timestamps are converted to a standard Unix format.
    - Duplicate user-item interactions are removed.

### Amazon Sales
- **Source**: `karkavelrajaj/amazon-sales-dataset`
- **Raw Files**: `amazon.csv`
- **Key Processing Steps**:
    - Columns are mapped (e.g., `product_id` -> `itemID`).
    - `category` and `about_product` are combined to create a `genres` field.
    - Missing timestamps are generated based on user interaction order.
    - Unnecessary columns (e.g., `discounted_price`, `img_link`) are dropped.

### Post Recommendations
- **Source**: `vatsalparsaniya/post-pecommendation`
- **Raw Files**: `user_data.csv`, `view_data.csv`, `post_data.csv`
- **Key Processing Steps**:
    - **Rating Generation**: This dataset lacks explicit ratings. They are generated based on user interaction frequency with different post categories.
    - Columns are mapped (e.g., `post_id` -> `itemID`, `category` -> `genres`).
    - User, post, and view data are merged into a single interaction table.

---

## 3. Data Loading & Caching

The `loader` function in [`qpera/datasets_loader.py`](../qpera/datasets_loader.py) provides a single, consistent interface for accessing all datasets.

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

<!--
  TODO: Authors can add a note here explaining why the RL pipeline requires a separate caching and processing mechanism, e.g., due to the need for graph structures and embeddings not used by other models.
-->

---

## 5. Adding a New Dataset

To integrate a new dataset into the project, follow these steps:

1.  **Create a Loader Class**: In `qpera/datasets_loader.py`, create a new class that inherits from `BaseDatasetLoader`. Implement the `_check_local_files_exist` and `merge_datasets` methods to handle your specific files and processing logic.
2.  **Register the Loader**: Add your new class to the `dataset_loaders` dictionary inside the `loader` function.
3.  **Add Downloader Support (Optional)**: In `qpera/datasets_downloader.py`, add your dataset's information to the `DATASET_CONFIG` dictionary to enable automatic downloads with `make check-datasets`.
4.  **Add RL Support (Optional)**: If the dataset should be used with the RL algorithm, update the path dictionaries (`DATASET_DIR`, `TMP_DIR`, `LABELS`) in `qpera/rl_utils.py`.

---

## 6. Troubleshooting

- **`FileNotFoundError`**: Ensure you have run `make check-datasets` to download all raw data.
- **Kaggle API `401 Unauthorized`**: Verify your `~/.kaggle/kaggle.json` file is correctly placed and has the right permissions (`chmod 600`).
- **RL Pipeline Errors**: If you encounter issues with the RL pipeline, try clearing its specific cache by deleting the `qpera/rl_tmp/<DatasetName>` directory and re-running the experiment.