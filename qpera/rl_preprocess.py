from __future__ import absolute_import, division, print_function

import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from recommenders.datasets.python_splitters import python_stratified_split
from sklearn.model_selection import train_test_split

from . import data_manipulation as dm
from .datasets_loader import loader as load_dataframe
from .rl_knowledge_graph import KnowledgeGraph
from .rl_utils import (
    BELONG_TO,
    DESCRIBED_AS,
    GENRES,
    ITEMID,
    KG_RELATION,
    LABELS,
    RATED,
    RATING,
    RATING_VALUE_FOR_ITEM,
    TITLE,
    TMP_DIR,
    USER_RATED_WITH_VALUE,
    USERID,
    WATCHED,
    get_entity_tail,
    save_kg,
    save_labels,
)


def create_processed_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process DataFrame to create entity vocabularies/mappings and structured relation data.

    Args:
        df: Input DataFrame with user-item interaction data

    Returns:
        Dictionary containing entity mappings, relations, and distributions
    """
    print("Processing DataFrame to create dataset object...")
    processed_data = {"entity_maps": {}, "relations": {}}

    # Create entity mappings
    processed_data["entity_maps"] = _create_entity_mappings(df)

    # Extract relations
    processed_data["relations"] = _extract_relations(df, processed_data["entity_maps"])

    # Calculate distributions for negative sampling
    processed_data["distributions"] = _calculate_distributions(processed_data)

    print("Finished processing DataFrame.")
    return processed_data


def _create_entity_mappings(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Create entity vocabularies and mappings from DataFrame columns."""
    print("  Creating entity mappings...")

    entity_columns = {
        USERID: "userID",
        ITEMID: "itemID",
        TITLE: "title",
        RATING: "rating",
        GENRES: "genres",
    }

    entity_maps = {}

    for entity_name, col_name in entity_columns.items():
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found for entity '{entity_name}'. Skipping.")
            continue

        entity_maps[entity_name] = _create_single_entity_mapping(df, entity_name, col_name)

    return entity_maps


def _create_single_entity_mapping(df: pd.DataFrame, entity_name: str, col_name: str) -> Dict[str, Any]:
    """Create mapping for a single entity type."""
    valid_series = df[col_name].dropna()

    if entity_name == GENRES:
        # Handle genres as space-separated values
        all_values = set(g for genres_list in valid_series.astype(str) for g in genres_list.split() if g)
    else:
        all_values = set(valid_series.unique())

    vocab = sorted(list(all_values))
    vocab_size = len(vocab)
    original_to_idx = {val: i for i, val in enumerate(vocab)}
    idx_to_original = {i: val for i, val in enumerate(vocab)}

    print(f"    Entity '{entity_name}': {vocab_size} unique values.")

    return {
        "vocab": vocab,
        "map": original_to_idx,
        "inv_map": idx_to_original,
        "vocab_size": vocab_size,
    }


def _extract_relations(df: pd.DataFrame, entity_maps: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
    """Extract all relations from the DataFrame."""
    print("  Extracting relations...")

    relations = {}

    # Extract each type of relation
    relations[WATCHED] = _extract_watched_relations(df, entity_maps)
    relations[BELONG_TO] = _extract_belong_to_relations(df, entity_maps)
    relations[DESCRIBED_AS] = _extract_described_as_relations(df, entity_maps)
    relations[RATED] = _extract_rated_relations(df, entity_maps, relations[WATCHED])
    relations[USER_RATED_WITH_VALUE] = _extract_user_rated_value_relations(df, entity_maps)
    relations[RATING_VALUE_FOR_ITEM] = _extract_rating_item_relations(df, entity_maps)

    return relations


def _get_entity_index(entity_type: str, value: Any, entity_maps: Dict[str, Dict[str, Any]]) -> Optional[int]:
    """Helper function to get entity index from value."""
    if entity_type in entity_maps and pd.notna(value) and value in entity_maps[entity_type]["map"]:
        return entity_maps[entity_type]["map"][value]
    return None


def _extract_watched_relations(df: pd.DataFrame, entity_maps: Dict[str, Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Extract WATCHED relations (USERID -> ITEMID)."""
    if USERID not in entity_maps or ITEMID not in entity_maps:
        return []

    watched_relations = set()
    for _, row in df.iterrows():
        uid_idx = _get_entity_index(USERID, row["userID"], entity_maps)
        iid_idx = _get_entity_index(ITEMID, row["itemID"], entity_maps)
        if uid_idx is not None and iid_idx is not None:
            watched_relations.add((uid_idx, iid_idx))

    result = list(watched_relations)
    print(f"    Relation '{WATCHED}': {len(result)} unique interactions.")
    return result


def _extract_belong_to_relations(df: pd.DataFrame, entity_maps: Dict[str, Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Extract BELONG_TO relations (ITEMID -> GENRES)."""
    if ITEMID not in entity_maps or GENRES not in entity_maps:
        return []

    belong_to_relations = set()
    item_genre_df = df[["itemID", "genres"]].drop_duplicates(subset=["itemID"]).dropna()

    for _, row in item_genre_df.iterrows():
        item_idx = _get_entity_index(ITEMID, row["itemID"], entity_maps)
        if item_idx is not None:
            genres_str = str(row["genres"])
            for genre in genres_str.split():
                if genre:
                    genre_idx = _get_entity_index(GENRES, genre, entity_maps)
                    if genre_idx is not None:
                        belong_to_relations.add((item_idx, genre_idx))

    result = list(belong_to_relations)
    print(f"    Relation '{BELONG_TO}': {len(result)} unique item-genre links.")
    return result


def _extract_described_as_relations(df: pd.DataFrame, entity_maps: Dict[str, Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Extract DESCRIBED_AS relations (TITLE -> ITEMID)."""
    if TITLE not in entity_maps or ITEMID not in entity_maps:
        return []

    described_as_relations = set()
    item_title_df = df[["itemID", "title"]].drop_duplicates(subset=["itemID"]).dropna()

    for _, row in item_title_df.iterrows():
        title_idx = _get_entity_index(TITLE, row["title"], entity_maps)
        item_idx = _get_entity_index(ITEMID, row["itemID"], entity_maps)
        if title_idx is not None and item_idx is not None:
            described_as_relations.add((title_idx, item_idx))

    result = list(described_as_relations)
    print(f"    Relation '{DESCRIBED_AS}': {len(result)} unique title-item links.")
    return result


def _extract_rated_relations(df: pd.DataFrame, entity_maps: Dict[str, Dict[str, Any]], watched_relations: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Extract RATED relations (USERID -> TITLE) derived from watched relations."""
    if USERID not in entity_maps or TITLE not in entity_maps or ITEMID not in entity_maps:
        return []

    # Build item to title mapping
    item_to_title = {}
    item_title_df = df[["itemID", "title"]].drop_duplicates(subset=["itemID"]).dropna()

    for _, row in item_title_df.iterrows():
        item_idx = _get_entity_index(ITEMID, row["itemID"], entity_maps)
        title_idx = _get_entity_index(TITLE, row["title"], entity_maps)
        if item_idx is not None and title_idx is not None:
            item_to_title[item_idx] = title_idx

    # Create rated relations from watched relations
    rated_relations = set()
    for user_idx, item_idx in watched_relations:
        if item_idx in item_to_title:
            title_idx = item_to_title[item_idx]
            rated_relations.add((user_idx, title_idx))

    result = list(rated_relations)
    print(f"    Relation '{RATED}': {len(result)} unique user-title links (derived).")
    return result


def _extract_user_rated_value_relations(df: pd.DataFrame, entity_maps: Dict[str, Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Extract USER_RATED_WITH_VALUE relations (USERID -> RATING)."""
    if USERID not in entity_maps or RATING not in entity_maps or "userID" not in df.columns or "rating" not in df.columns:
        return []

    user_rated_value_relations = set()
    for _, row in df.iterrows():
        uid_idx = _get_entity_index(USERID, row["userID"], entity_maps)
        rating_val_idx = _get_entity_index(RATING, row["rating"], entity_maps)
        if uid_idx is not None and rating_val_idx is not None:
            user_rated_value_relations.add((uid_idx, rating_val_idx))

    result = list(user_rated_value_relations)
    print(f"    Relation '{USER_RATED_WITH_VALUE}': {len(result)} unique user-rating_value links.")
    return result


def _extract_rating_item_relations(df: pd.DataFrame, entity_maps: Dict[str, Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Extract RATING_VALUE_FOR_ITEM relations (RATING -> ITEMID)."""
    if RATING not in entity_maps or ITEMID not in entity_maps or "rating" not in df.columns or "itemID" not in df.columns:
        return []

    rating_item_relations = set()
    for _, row in df.iterrows():
        rating_val_idx = _get_entity_index(RATING, row["rating"], entity_maps)
        iid_idx = _get_entity_index(ITEMID, row["itemID"], entity_maps)
        if rating_val_idx is not None and iid_idx is not None:
            rating_item_relations.add((rating_val_idx, iid_idx))

    result = list(rating_item_relations)
    print(f"    Relation '{RATING_VALUE_FOR_ITEM}': {len(result)} unique rating_value-item links.")
    return result


def _calculate_distributions(processed_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Calculate tail entity distributions for negative sampling."""
    print("  Calculating tail entity distributions for negative sampling...")

    distributions = {}
    relations = processed_data.get("relations", {})
    entity_maps = processed_data.get("entity_maps", {})

    for relation_name, relation_data in relations.items():
        print(f"    Calculating distribution for: {relation_name}")

        try:
            distribution = _calculate_single_distribution(relation_name, relation_data, entity_maps)
            if distribution is not None:
                distributions[relation_name] = distribution
                print(f"      Stored distribution for '{relation_name}' with {len(distribution)} elements.")

        except Exception as e:
            print(f"    Unexpected error calculating distribution for '{relation_name}': {e}")

    return distributions


def _calculate_single_distribution(relation_name: str, relation_data: List[Tuple[int, int]], entity_maps: Dict[str, Dict[str, Any]]) -> Optional[np.ndarray]:
    """Calculate distribution for a single relation."""
    # Find head entity type
    head_entity_type = None
    for h_type, rels in KG_RELATION.items():
        if relation_name in rels:
            head_entity_type = h_type.lower()
            break

    if head_entity_type is None:
        print(f"      Warning: Could not find head entity for relation '{relation_name}'. Skipping.")
        return None

    # Get tail entity information
    try:
        tail_entity_type = get_entity_tail(head_entity_type, relation_name)
        tail_map_data = entity_maps.get(tail_entity_type, {})
        tail_vocab_size = tail_map_data.get("vocab_size", 0)
    except KeyError as e:
        print(f"    Error processing distribution for '{relation_name}': Missing key {e}")
        return None

    if tail_vocab_size == 0:
        print(f"      Warning: Tail vocab size is 0 for relation '{relation_name}'. Skipping.")
        return None

    # Count tail entity occurrences
    tail_counts = np.zeros(tail_vocab_size, dtype=np.float64)
    num_valid_triples = 0

    for _, tail_idx in relation_data:
        if 0 <= tail_idx < tail_vocab_size:
            tail_counts[tail_idx] += 1
            num_valid_triples += 1

    # Handle edge cases
    if num_valid_triples == 0:
        print(f"      Warning: No valid triples found for relation '{relation_name}'. Setting uniform distribution.")
        return np.ones(tail_vocab_size, dtype=np.float64) / tail_vocab_size

    # Apply power transformation and normalize
    print(f"      Counted {int(np.sum(tail_counts))} occurrences across {num_valid_triples} valid triples.")
    distrib = np.power(tail_counts, 0.75)

    sum_distrib = np.sum(distrib)
    if sum_distrib > 0:
        return distrib / sum_distrib
    else:
        print(f"      Warning: Sum of distribution is zero for '{relation_name}'. Using uniform.")
        return np.ones(tail_vocab_size, dtype=np.float64) / tail_vocab_size


def generate_labels_from_df(dataset_name: str, df: pd.DataFrame, user_map: Dict[Any, int], item_map: Dict[Any, int], mode: str) -> None:
    """
    Generate item interaction labels for users from a DataFrame.

    Args:
        dataset_name: Name of the dataset
        df: DataFrame containing user-item interactions
        user_map: Mapping from original user IDs to indices
        item_map: Mapping from original item IDs to indices
        mode: Either "train" or "test"
    """
    print(f"Generating labels for mode: {mode}")

    user_item_ratings = defaultdict(list)
    processed_interactions = 0
    skipped_interactions = 0

    # Process each interaction
    for _, row in df.iterrows():
        uid_orig = row["userID"]
        iid_orig = row["itemID"]
        rating_val = row["rating"]

        if _is_valid_interaction(uid_orig, iid_orig, rating_val, user_map, item_map):
            user_idx = user_map[uid_orig]
            item_idx = item_map[iid_orig]
            user_item_ratings[user_idx].append((item_idx, float(rating_val)))
            processed_interactions += 1
        else:
            skipped_interactions += 1

    # Remove duplicates and sort
    final_user_items = _deduplicate_user_items(user_item_ratings)

    print(f"  Generated labels for {len(final_user_items)} users.")
    print(f"  Processed {processed_interactions} interactions, skipped {skipped_interactions}.")

    save_labels(dataset_name, final_user_items, mode=mode)


def _is_valid_interaction(uid_orig: Any, iid_orig: Any, rating_val: Any, user_map: Dict[Any, int], item_map: Dict[Any, int]) -> bool:
    """Check if an interaction is valid for label generation."""
    return pd.notna(uid_orig) and pd.notna(iid_orig) and pd.notna(rating_val) and uid_orig in user_map and iid_orig in item_map


def _deduplicate_user_items(user_item_ratings: Dict[int, List[Tuple[int, float]]]) -> Dict[int, List[Tuple[int, float]]]:
    """Remove duplicate items for each user and sort by item index."""
    final_user_items = {}

    for uid, item_rating_list in user_item_ratings.items():
        seen_items = set()
        dedup_list = []

        for item_idx, rating_val in item_rating_list:
            if item_idx not in seen_items:
                dedup_list.append((item_idx, rating_val))
                seen_items.add(item_idx)

        # Sort by item_idx for consistency
        final_user_items[uid] = sorted(dedup_list, key=lambda x: x[0])

    return final_user_items


def preprocess_rl(
    dataset: str,
    want_col: List[str],
    num_rows: Optional[int],
    ratio: float,
    seed: int,
    personalization: bool = False,
    fraction_to_change: float = 0,
    change_rating: bool = False,
    privacy: bool = False,
    hide_type: str = "values_in_column",
    columns_to_hide: Optional[List[str]] = None,
    fraction_to_hide: float = 0,
    records_to_hide: Optional[List[int]] = None,
    force_reprocess: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing function for RL-based recommendation system.

    Args:
        dataset: Dataset name
        want_col: Required columns for the dataset
        num_rows: Number of rows to use (None for all)
        ratio: Train/test split ratio
        seed: Random seed for reproducibility
        personalization: Whether to apply personalization modifications
        fraction_to_change: Fraction of data to modify for personalization
        change_rating: Whether to change rating values
        privacy: Whether to apply privacy modifications
        hide_type: Type of privacy hiding
        columns_to_hide: Columns to hide for privacy
        fraction_to_hide: Fraction of data to hide
        records_to_hide: Specific records to hide
        force_reprocess: Whether to force reprocessing of cached data

    Returns:
        Tuple of (full_data, train_data, test_data) DataFrames
    """
    # Load raw data
    print(f"Loading raw data for dataset: {dataset}")
    try:
        data_df = _load_raw_data(dataset, want_col, num_rows, seed)
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        raise

    # Split data
    print("Splitting data into train/test sets...")
    train_df, test_df = _split_data(data_df, ratio, seed)

    # Apply modifications
    if privacy:
        data_df = _apply_privacy_modifications(data_df, hide_type, columns_to_hide, fraction_to_hide, records_to_hide, seed)

    if personalization:
        train_df = _apply_personalization_modifications(data_df, train_df, fraction_to_change, change_rating, seed)

    # Process and cache dataset
    processed_dataset = _get_or_create_processed_dataset(dataset, data_df, force_reprocess)

    # Create and cache knowledge graph
    _get_or_create_knowledge_graph(dataset, processed_dataset, force_reprocess)

    # Generate labels
    _generate_or_load_labels(dataset, train_df, test_df, processed_dataset, force_reprocess)

    print("Preprocessing finished.")
    return data_df, train_df, test_df


def _load_raw_data(dataset: str, want_col: List[str], num_rows: Optional[int], seed: int) -> pd.DataFrame:
    """Load raw data using custom loader."""
    try:
        data_df = load_dataframe(
            dataset_name=dataset.lower(),
            want_col=want_col,
            num_rows=num_rows,
            seed=seed,
        )
        print(f"Loaded DataFrame with shape: {data_df.shape}")
        print(data_df.head())
        return data_df

    except KeyError as e:
        print(f"Missing required column(s) - {e}")
        print("Please ensure the merge_file contains:", want_col)
        raise
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        raise


def _split_data(data_df: pd.DataFrame, ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    try:
        train_df, test_df = python_stratified_split(data_df, ratio=ratio, col_user="userID", col_item="itemID", seed=seed)
        print("Stratified split successful.")
    except Exception as e:
        print(f"Stratified split failed ({e}), using simple random split.")
        train_df, test_df = train_test_split(data_df, test_size=1.0 - ratio, random_state=seed)

    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    return train_df, test_df


def _apply_privacy_modifications(
    data_df: pd.DataFrame, hide_type: str, columns_to_hide: Optional[List[str]], fraction_to_hide: float, records_to_hide: Optional[List[int]], seed: int
) -> pd.DataFrame:
    """Apply privacy modifications to the data."""
    return dm.hide_information_in_dataframe(
        data=data_df,
        hide_type=hide_type,
        columns_to_hide=columns_to_hide,
        fraction_to_hide=fraction_to_hide,
        records_to_hide=records_to_hide,
        seed=seed,
    )


def _apply_personalization_modifications(
    data_df: pd.DataFrame, train_df: pd.DataFrame, fraction_to_change: float, change_rating: bool, seed: int
) -> pd.DataFrame:
    """Apply personalization modifications to the training data."""
    return dm.change_items_in_dataframe(
        all=data_df,
        data=data_df,
        fraction_to_change=fraction_to_change,
        change_rating=change_rating,
        seed=seed,
    )


def _get_or_create_processed_dataset(dataset: str, data_df: pd.DataFrame, force_reprocess: bool) -> Dict[str, Any]:
    """Get cached processed dataset or create a new one."""
    processed_dataset_file = f"{TMP_DIR[dataset]}/processed_dataset.pkl"

    # Try to load cached version
    if not force_reprocess and os.path.exists(processed_dataset_file):
        processed_dataset = _load_cached_processed_dataset(processed_dataset_file)
        if processed_dataset is not None:
            return processed_dataset

    # Create new processed dataset
    print("Creating processed dataset object from DataFrame...")
    _ensure_directory_exists(TMP_DIR[dataset])

    processed_dataset = create_processed_dataset(data_df)

    # Save to cache
    _save_processed_dataset(processed_dataset, processed_dataset_file)

    return processed_dataset


def _load_cached_processed_dataset(processed_dataset_file: str) -> Optional[Dict[str, Any]]:
    """Load cached processed dataset if valid."""
    print(f"Loading cached processed dataset from {processed_dataset_file}...")

    try:
        with open(processed_dataset_file, "rb") as f:
            processed_dataset = pickle.load(f)

        if _is_valid_processed_dataset(processed_dataset):
            return processed_dataset
        else:
            print("Cached processed dataset seems invalid or incomplete. Reprocessing...")
            return None

    except Exception as e:
        print(f"Failed to load cached processed dataset ({e}). Reprocessing...")
        return None


def _is_valid_processed_dataset(processed_dataset: Any) -> bool:
    """Validate processed dataset structure."""
    return isinstance(processed_dataset, dict) and "entity_maps" in processed_dataset and "relations" in processed_dataset


def _save_processed_dataset(processed_dataset: Dict[str, Any], file_path: str) -> None:
    """Save processed dataset to file."""
    print(f"Saving processed dataset object to {file_path}...")
    try:
        with open(file_path, "wb") as f:
            pickle.dump(processed_dataset, f)
    except Exception as e:
        print(f"Error saving processed dataset to {file_path}: {e}")


def _get_or_create_knowledge_graph(dataset: str, processed_dataset: Dict[str, Any], force_reprocess: bool) -> None:
    """Get cached knowledge graph or create a new one."""
    kg_file = f"{TMP_DIR[dataset]}/kg.pkl"

    if not force_reprocess and os.path.exists(kg_file):
        print(f"Loading cached knowledge graph from {kg_file}...")
        return

    print("Creating knowledge graph from processed dataset...")
    try:
        kg = KnowledgeGraph(processed_dataset)
        print("Computing node degrees...")
        kg.compute_degrees()
        print(f"Saving knowledge graph to {kg_file}...")
        save_kg(dataset, kg)
    except Exception as e:
        print(f"Error during Knowledge Graph creation/saving: {e}")


def _generate_or_load_labels(dataset: str, train_df: pd.DataFrame, test_df: pd.DataFrame, processed_dataset: Dict[str, Any], force_reprocess: bool) -> None:
    """Generate labels or load cached ones."""
    train_label_file, test_label_file = LABELS[dataset]

    if not force_reprocess and os.path.exists(train_label_file) and os.path.exists(test_label_file):
        print("Train/test labels already exist. Skipping generation.")
        return

    print("Generating train/test labels from split DataFrames.")

    # Validate entity maps
    entity_maps = processed_dataset.get("entity_maps", {})
    if USERID not in entity_maps or ITEMID not in entity_maps:
        print("Error: User or Item map not found in processed_dataset. Cannot generate labels.")
        return

    user_map = entity_maps[USERID]["map"]
    item_map = entity_maps[ITEMID]["map"]

    try:
        generate_labels_from_df(dataset, train_df, user_map, item_map, "train")
        generate_labels_from_df(dataset, test_df, user_map, item_map, "test")
    except Exception as e:
        print(f"Error generating labels: {e}")


def _ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            raise
