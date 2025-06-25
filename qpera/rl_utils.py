from __future__ import absolute_import, division, print_function

import logging
import logging.handlers
import os
import pickle
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfTransformer

# Dataset names
MOVIELENS = "movielens"
AMAZONSALES = "amazonsales"
POSTRECOMMENDATIONS = "postrecommendations"

# Dataset directories
DATASET_DIR = {
    MOVIELENS: "./datasets/MovieLens",
    AMAZONSALES: "./datasets/AmazonSales",
    POSTRECOMMENDATIONS: "./datasets/PostRecommendations",
}

# Model result directories
TMP_DIR = {
    MOVIELENS: "qpera/rl_tmp/MovieLens",
    AMAZONSALES: "qpera/rl_tmp/AmazonSales",
    POSTRECOMMENDATIONS: "qpera/rl_tmp/PostRecommendations",
}

# Label files
LABELS = {
    MOVIELENS: (TMP_DIR[MOVIELENS] + "/train_label.pkl", TMP_DIR[MOVIELENS] + "/test_label.pkl"),
    AMAZONSALES: (
        TMP_DIR[AMAZONSALES] + "/train_label.pkl",
        TMP_DIR[AMAZONSALES] + "/test_label.pkl",
    ),
    POSTRECOMMENDATIONS: (
        TMP_DIR[POSTRECOMMENDATIONS] + "/train_label.pkl",
        TMP_DIR[POSTRECOMMENDATIONS] + "/test_label.pkl",
    ),
}

# Entity types
USERID = "user_id"
ITEMID = "item_id"
TITLE = "title"
GENRES = "genres"
RATING = "rating"
PREDICTION = "prediction"

# Relation types
WATCHED = "watched"
RATED = "rated"
DESCRIBED_AS = "described_as"
BELONG_TO = "belongs_to"
USER_RATED_WITH_VALUE = "user_rated_with_value"
RATING_VALUE_FOR_ITEM = "rating_value_for_item"
SELF_LOOP = "self_loop"  # only for kg env

KG_RELATION_TYPES_ORDERED = [WATCHED, RATED, DESCRIBED_AS, BELONG_TO]

# Knowledge graph relation mappings
KG_RELATION = {
    USERID: {
        WATCHED: ITEMID,
        RATED: TITLE,
        USER_RATED_WITH_VALUE: RATING,
    },
    TITLE: {
        RATED: USERID,
        DESCRIBED_AS: ITEMID,
    },
    ITEMID: {
        WATCHED: USERID,
        BELONG_TO: GENRES,
    },
    GENRES: {
        BELONG_TO: ITEMID,
    },
    RATING: {
        USER_RATED_WITH_VALUE: USERID,
        RATING_VALUE_FOR_ITEM: ITEMID,
    },
}

# Path patterns for recommendation reasoning
PATH_PATTERN = {
    # length = 3
    1: ((None, USERID), (RATED, TITLE), (DESCRIBED_AS, ITEMID)),
    2: ((None, USERID), (USER_RATED_WITH_VALUE, RATING), (RATING_VALUE_FOR_ITEM, ITEMID)),
    # length = 4
    11: ((None, USERID), (WATCHED, ITEMID), (WATCHED, USERID), (WATCHED, ITEMID)),
    12: ((None, USERID), (WATCHED, ITEMID), (DESCRIBED_AS, TITLE), (DESCRIBED_AS, ITEMID)),
    13: ((None, USERID), (WATCHED, ITEMID), (BELONG_TO, GENRES), (BELONG_TO, ITEMID)),
    14: ((None, USERID), (RATED, TITLE), (RATED, USERID), (WATCHED, ITEMID)),
    15: (
        (None, USERID),
        (WATCHED, ITEMID),
        (RATING_VALUE_FOR_ITEM, RATING),
        (RATING_VALUE_FOR_ITEM, ITEMID),
    ),
}


def get_entities() -> List[str]:
    """
    Get list of all entity types in the knowledge graph.

    Returns:
        List of entity type names
    """
    return list(KG_RELATION.keys())


def get_relations(entity_head: str) -> List[str]:
    """
    Get list of all relation types for a given head entity.

    Args:
        entity_head: Head entity type

    Returns:
        List of relation types from the head entity

    Raises:
        KeyError: If entity_head is not in KG_RELATION
    """
    if entity_head not in KG_RELATION:
        raise KeyError(f"Entity '{entity_head}' not found in KG_RELATION")
    return list(KG_RELATION[entity_head].keys())


def get_entity_tail(entity_head: str, relation: str) -> str:
    """
    Get tail entity type for a given head entity and relation.

    Args:
        entity_head: Head entity type
        relation: Relation type

    Returns:
        Tail entity type

    Raises:
        KeyError: If entity_head or relation not found
    """
    if entity_head not in KG_RELATION:
        raise KeyError(f"Entity '{entity_head}' not found in KG_RELATION")
    if relation not in KG_RELATION[entity_head]:
        raise KeyError(f"Relation '{relation}' not found for entity '{entity_head}'")
    return KG_RELATION[entity_head][relation]


def compute_tfidf_fast(vocab: List[Any], docs: List[List[int]]) -> sp.csr_matrix:
    """
    Compute TFIDF scores for all vocabulary terms across documents.

    Args:
        vocab: Vocabulary list
        docs: List of documents, where each document is a list of term indices

    Returns:
        Sparse matrix of TFIDF scores with shape [num_docs, num_vocab]
    """
    if not docs:
        return sp.csr_matrix((0, len(vocab)))

    # Compute term frequency in each document
    data, indices, indptr = [], [], [0]

    for doc in docs:
        if not doc:  # Handle empty documents
            indptr.append(len(indices))
            continue

        term_count = {}
        for term_idx in doc:
            if 0 <= term_idx < len(vocab):  # Validate term index
                term_count[term_idx] = term_count.get(term_idx, 0) + 1

        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))

    # Create term frequency matrix
    tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # Compute normalized TFIDF
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def get_logger(logname: str) -> logging.Logger:
    """
    Create and configure a logger with both console and file output.

    Args:
        logname: Path to log file

    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(logname)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter("[%(levelname)s]  %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.handlers.RotatingFileHandler(logname, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set deterministic algorithms for reproducibility
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # For older PyTorch versions
        torch.set_deterministic(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_dataset(dataset: str, dataset_obj: Any) -> None:
    """
    Save dataset object to pickle file.

    Args:
        dataset: Dataset name
        dataset_obj: Dataset object to save

    Raises:
        KeyError: If dataset name is not recognized
        IOError: If file cannot be written
    """
    if dataset not in TMP_DIR:
        raise KeyError(f"Unknown dataset: {dataset}")

    dataset_file = os.path.join(TMP_DIR[dataset], "dataset.pkl")
    _ensure_directory_exists(os.path.dirname(dataset_file))

    try:
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset_obj, f)
    except IOError as e:
        raise IOError(f"Failed to save dataset to {dataset_file}: {e}")


def load_dataset(dataset: str) -> Any:
    """
    Load dataset object from pickle file.

    Args:
        dataset: Dataset name

    Returns:
        Loaded dataset object

    Raises:
        KeyError: If dataset name is not recognized
        FileNotFoundError: If dataset file doesn't exist
    """
    if dataset not in TMP_DIR:
        raise KeyError(f"Unknown dataset: {dataset}")

    dataset_file = os.path.join(TMP_DIR[dataset], "dataset.pkl")

    try:
        with open(dataset_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")


def save_labels(dataset: str, labels: Dict[int, List[Tuple]], mode: str = "train") -> None:
    """
    Save user labels to pickle file.

    Args:
        dataset: Dataset name
        labels: Dictionary mapping user IDs to item-rating tuples
        mode: Either "train" or "test"

    Raises:
        ValueError: If mode is not "train" or "test"
        KeyError: If dataset name is not recognized
    """
    if mode not in ["train", "test"]:
        raise ValueError("mode should be one of {'train', 'test'}")

    if dataset not in LABELS:
        raise KeyError(f"Unknown dataset: {dataset}")

    label_file = LABELS[dataset][0] if mode == "train" else LABELS[dataset][1]
    _ensure_directory_exists(os.path.dirname(label_file))

    try:
        with open(label_file, "wb") as f:
            pickle.dump(labels, f)
    except IOError as e:
        raise IOError(f"Failed to save labels to {label_file}: {e}")


def load_labels(dataset: str, mode: str = "train") -> Dict[int, List[Tuple]]:
    """
    Load user labels from pickle file.

    Args:
        dataset: Dataset name
        mode: Either "train" or "test"

    Returns:
        Dictionary mapping user IDs to item-rating tuples

    Raises:
        ValueError: If mode is not "train" or "test"
        KeyError: If dataset name is not recognized
        FileNotFoundError: If label file doesn't exist
    """
    if mode not in ["train", "test"]:
        raise ValueError("mode should be one of {'train', 'test'}")

    if dataset not in LABELS:
        raise KeyError(f"Unknown dataset: {dataset}")

    label_file = LABELS[dataset][0] if mode == "train" else LABELS[dataset][1]

    try:
        with open(label_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Label file not found: {label_file}")


def save_embed(dataset: str, embed: Dict[str, Any]) -> None:
    """
    Save embeddings to pickle file.

    Args:
        dataset: Dataset name
        embed: Dictionary containing embeddings

    Raises:
        KeyError: If dataset name is not recognized
    """
    if dataset not in TMP_DIR:
        raise KeyError(f"Unknown dataset: {dataset}")

    embed_file = os.path.join(TMP_DIR[dataset], "transe_embed.pkl")
    _ensure_directory_exists(os.path.dirname(embed_file))

    try:
        with open(embed_file, "wb") as f:
            pickle.dump(embed, f)
    except IOError as e:
        raise IOError(f"Failed to save embeddings to {embed_file}: {e}")


def load_embed(dataset: str) -> Dict[str, Any]:
    """
    Load embeddings from pickle file.

    Args:
        dataset: Dataset name

    Returns:
        Dictionary containing embeddings

    Raises:
        KeyError: If dataset name is not recognized
        FileNotFoundError: If embedding file doesn't exist
    """
    if dataset not in TMP_DIR:
        raise KeyError(f"Unknown dataset: {dataset}")

    embed_file = os.path.join(TMP_DIR[dataset], "transe_embed.pkl")
    print(f"Loading embeddings from: {embed_file}")

    try:
        with open(embed_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Embedding file not found: {embed_file}")


def save_kg(dataset: str, kg: Any) -> None:
    """
    Save knowledge graph to pickle file.

    Args:
        dataset: Dataset name
        kg: Knowledge graph object

    Raises:
        KeyError: If dataset name is not recognized
    """
    if dataset not in TMP_DIR:
        raise KeyError(f"Unknown dataset: {dataset}")

    kg_file = os.path.join(TMP_DIR[dataset], "kg.pkl")
    _ensure_directory_exists(os.path.dirname(kg_file))

    try:
        with open(kg_file, "wb") as f:
            pickle.dump(kg, f)
    except IOError as e:
        raise IOError(f"Failed to save knowledge graph to {kg_file}: {e}")


def load_kg(dataset: str) -> Any:
    """
    Load knowledge graph from pickle file.

    Args:
        dataset: Dataset name

    Returns:
        Knowledge graph object

    Raises:
        KeyError: If dataset name is not recognized
        FileNotFoundError: If KG file doesn't exist
    """
    if dataset not in TMP_DIR:
        raise KeyError(f"Unknown dataset: {dataset}")

    kg_file = os.path.join(TMP_DIR[dataset], "kg.pkl")

    try:
        with open(kg_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Knowledge graph file not found: {kg_file}")


def _ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Directory path to check/create
    """
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create directory {directory}: {e}")


def validate_dataset_name(dataset: str) -> bool:
    """
    Validate that a dataset name is recognized.

    Args:
        dataset: Dataset name to validate

    Returns:
        True if dataset name is valid, False otherwise
    """
    return dataset in TMP_DIR


def get_supported_datasets() -> List[str]:
    """
    Get list of all supported dataset names.

    Returns:
        List of supported dataset names
    """
    return list(TMP_DIR.keys())


def get_file_paths(dataset: str) -> Dict[str, str]:
    """
    Get all file paths for a given dataset.

    Args:
        dataset: Dataset name

    Returns:
        Dictionary mapping file types to their paths

    Raises:
        KeyError: If dataset name is not recognized
    """
    if dataset not in TMP_DIR:
        raise KeyError(f"Unknown dataset: {dataset}")

    base_dir = TMP_DIR[dataset]
    train_label, test_label = LABELS[dataset]

    return {
        "base_dir": base_dir,
        "dataset_file": os.path.join(base_dir, "dataset.pkl"),
        "processed_dataset_file": os.path.join(base_dir, "processed_dataset.pkl"),
        "kg_file": os.path.join(base_dir, "kg.pkl"),
        "embed_file": os.path.join(base_dir, "transe_embed.pkl"),
        "train_labels": train_label,
        "test_labels": test_label,
    }


def cleanup_dataset_files(dataset: str, confirm: bool = False) -> None:
    """
    Clean up all generated files for a dataset.

    Args:
        dataset: Dataset name
        confirm: If True, actually delete files; if False, just list them

    Raises:
        KeyError: If dataset name is not recognized
    """
    if dataset not in TMP_DIR:
        raise KeyError(f"Unknown dataset: {dataset}")

    file_paths = get_file_paths(dataset)
    files_to_remove = [
        file_paths["dataset_file"],
        file_paths["processed_dataset_file"],
        file_paths["kg_file"],
        file_paths["embed_file"],
        file_paths["train_labels"],
        file_paths["test_labels"],
    ]

    existing_files = [f for f in files_to_remove if os.path.exists(f)]

    if not existing_files:
        print(f"No files found to clean up for dataset: {dataset}")
        return

    if not confirm:
        print(f"Files that would be removed for dataset '{dataset}':")
        for file_path in existing_files:
            print(f"  {file_path}")
        print("Use confirm=True to actually delete these files.")
        return

    for file_path in existing_files:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except OSError as e:
            print(f"Failed to remove {file_path}: {e}")
