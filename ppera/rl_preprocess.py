from __future__ import absolute_import, division, print_function

import os
import pickle
from collections import defaultdict

from . import data_manipulation as dm
import numpy as np
import pandas as pd
from .datasets_loader import loader as load_dataframe
from recommenders.datasets.python_splitters import python_stratified_split
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
from sklearn.model_selection import train_test_split


def create_processed_dataset(df: pd.DataFrame) -> dict:
    """
    Processes the DataFrame to create entity vocabularies/mappings
    and structured relation data. Returns a standard dictionary.
    """
    print("Processing DataFrame to create dataset object...")
    processed_data = {"entity_maps": {}, "relations": {}}

    entity_columns = {
        USERID: "userID",
        ITEMID: "itemID",
        TITLE: "title",
        RATING: "rating",
        GENRES: "genres",
    }

    print("  Creating entity mappings...")
    for entity_name, col_name in entity_columns.items():
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found for entity '{entity_name}'. Skipping.")
            continue

        valid_series = df[col_name].dropna()

        if entity_name == GENRES:
            all_values = set(g for genres_list in valid_series.astype(str) for g in genres_list.split() if g)
        else:
            all_values = set(valid_series.unique())

        vocab = sorted(list(all_values))
        vocab_size = len(vocab)
        original_to_idx = {val: i for i, val in enumerate(vocab)}
        idx_to_original = {i: val for i, val in enumerate(vocab)}

        processed_data["entity_maps"][entity_name] = {
            "vocab": vocab,
            "map": original_to_idx,
            "inv_map": idx_to_original,
            "vocab_size": vocab_size,
        }
        print(f"    Entity '{entity_name}': {vocab_size} unique values.")

    print("  Extracting relations...")

    def get_idx(entity_type, value, maps):
        if entity_type in maps and pd.notna(value) and value in maps[entity_type]["map"]:
            return maps[entity_type]["map"][value]
        return None

    # 1. WATCHED (USERID -> ITEMID)
    if USERID in processed_data["entity_maps"] and ITEMID in processed_data["entity_maps"]:
        watched_relations = set()
        for _, row in df.iterrows():
            uid_idx = get_idx(USERID, row[entity_columns[USERID]], processed_data["entity_maps"])
            iid_idx = get_idx(ITEMID, row[entity_columns[ITEMID]], processed_data["entity_maps"])
            if uid_idx is not None and iid_idx is not None:
                watched_relations.add((uid_idx, iid_idx))
        processed_data["relations"][WATCHED] = list(watched_relations)
        print(f"    Relation '{WATCHED}': {len(processed_data['relations'][WATCHED])} unique interactions.")

    # 2. BELONG_TO (ITEMID -> GENRES)
    if ITEMID in processed_data["entity_maps"] and GENRES in processed_data["entity_maps"]:
        belong_to_relations = set()
        item_genre_df = df[[entity_columns[ITEMID], entity_columns[GENRES]]].drop_duplicates(subset=[entity_columns[ITEMID]]).dropna()
        for _, row in item_genre_df.iterrows():
            item_idx = get_idx(ITEMID, row[entity_columns[ITEMID]], processed_data["entity_maps"])
            if item_idx is not None:
                genres_str = str(row[entity_columns[GENRES]])
                for genre in genres_str.split():
                    if genre:
                        genre_idx = get_idx(GENRES, genre, processed_data["entity_maps"])
                        if genre_idx is not None:
                            belong_to_relations.add((item_idx, genre_idx))
        processed_data["relations"][BELONG_TO] = list(belong_to_relations)
        print(f"    Relation '{BELONG_TO}': {len(processed_data['relations'][BELONG_TO])} unique item-genre links.")

    # 3. DESCRIBED_AS (TITLE -> ITEMID) - Assuming one primary title per item
    if TITLE in processed_data["entity_maps"] and ITEMID in processed_data["entity_maps"]:
        described_as_relations = set()
        item_title_df = df[[entity_columns[ITEMID], entity_columns[TITLE]]].drop_duplicates(subset=[entity_columns[ITEMID]]).dropna()
        for _, row in item_title_df.iterrows():
            title_idx = get_idx(TITLE, row[entity_columns[TITLE]], processed_data["entity_maps"])
            item_idx = get_idx(ITEMID, row[entity_columns[ITEMID]], processed_data["entity_maps"])
            if title_idx is not None and item_idx is not None:
                described_as_relations.add((title_idx, item_idx))
        processed_data["relations"][DESCRIBED_AS] = list(described_as_relations)
        print(f"    Relation '{DESCRIBED_AS}': {len(processed_data['relations'][DESCRIBED_AS])} unique title-item links.")

    # 4. RATED (USERID -> TITLE) - Simplified
    if USERID in processed_data["entity_maps"] and TITLE in processed_data["entity_maps"] and ITEMID in processed_data["entity_maps"]:
        rated_relations = set()
        item_to_title = {}
        item_title_df = df[[entity_columns[ITEMID], entity_columns[TITLE]]].drop_duplicates(subset=[entity_columns[ITEMID]]).dropna()
        for _, row in item_title_df.iterrows():
            item_idx = get_idx(ITEMID, row[entity_columns[ITEMID]], processed_data["entity_maps"])
            title_idx = get_idx(TITLE, row[entity_columns[TITLE]], processed_data["entity_maps"])
            if item_idx is not None and title_idx is not None:
                item_to_title[item_idx] = title_idx

        watched_rel_data = processed_data.get("relations", {}).get(WATCHED, [])
        for user_idx, item_idx in watched_rel_data:
            if item_idx in item_to_title:
                title_idx = item_to_title[item_idx]
                rated_relations.add((user_idx, title_idx))

        processed_data["relations"][RATED] = list(rated_relations)
        print(f"    Relation '{RATED}': {len(processed_data['relations'][RATED])} unique user-title links (derived).")
    # 5. USER_RATED_WITH_VALUE (USERID -> RATING entity)
    if (
        USERID in processed_data["entity_maps"]
        and RATING in processed_data["entity_maps"]
        and entity_columns[USERID] in df.columns
        and entity_columns[RATING] in df.columns
    ):
        user_rated_value_relations = set()
        for _, row in df.iterrows():
            uid_idx = get_idx(USERID, row[entity_columns[USERID]], processed_data["entity_maps"])
            # The 'rating' column from df contains the actual rating value (e.g., 4.5)
            # This value needs to be mapped to an index in the RATING entity's vocab
            rating_val_idx = get_idx(RATING, row[entity_columns[RATING]], processed_data["entity_maps"])
            if uid_idx is not None and rating_val_idx is not None:
                user_rated_value_relations.add((uid_idx, rating_val_idx))
        processed_data["relations"][USER_RATED_WITH_VALUE] = list(user_rated_value_relations)
        print(f"    Relation '{USER_RATED_WITH_VALUE}': {len(processed_data['relations'][USER_RATED_WITH_VALUE])} unique user-rating_value links.")

    # 6. RATING_VALUE_FOR_ITEM (RATING entity -> ITEMID)
    if (
        RATING in processed_data["entity_maps"]
        and ITEMID in processed_data["entity_maps"]
        and entity_columns[RATING] in df.columns
        and entity_columns[ITEMID] in df.columns
    ):
        rating_item_relations = set()
        for _, row in df.iterrows():
            rating_val_idx = get_idx(RATING, row[entity_columns[RATING]], processed_data["entity_maps"])
            iid_idx = get_idx(ITEMID, row[entity_columns[ITEMID]], processed_data["entity_maps"])
            if rating_val_idx is not None and iid_idx is not None:
                rating_item_relations.add((rating_val_idx, iid_idx))
        processed_data["relations"][RATING_VALUE_FOR_ITEM] = list(rating_item_relations)
        print(f"    Relation '{RATING_VALUE_FOR_ITEM}': {len(processed_data['relations'][RATING_VALUE_FOR_ITEM])} unique rating_value-item links.")

    print("  Calculating tail entity distributions for negative sampling...")
    processed_data["distributions"] = {}

    for relation_name, relation_data in processed_data.get("relations", {}).items():
        print(f"    Calculating distribution for: {relation_name}")
        try:
            head_entity_type = None
            for h_type, rels in KG_RELATION.items():
                if relation_name in rels:
                    head_entity_type = h_type.lower()
                    break
            if head_entity_type is None:
                print(f"      Warning: Could not find head entity for relation '{relation_name}'. Skipping distribution calculation.")
                continue

            tail_entity_type = get_entity_tail(head_entity_type, relation_name)
            tail_map_data = processed_data.get("entity_maps", {}).get(tail_entity_type, {})
            tail_vocab_size = tail_map_data.get("vocab_size", 0)

            if tail_vocab_size == 0:
                print(f"      Warning: Tail vocab size is 0 for relation '{relation_name}'. Skipping distribution calculation.")
                continue

            tail_counts = np.zeros(tail_vocab_size, dtype=np.float64)
            num_valid_triples = 0
            for _, tail_idx in relation_data:
                if 0 <= tail_idx < tail_vocab_size:
                    tail_counts[tail_idx] += 1
                    num_valid_triples += 1

            if num_valid_triples == 0:
                print(f"      Warning: No valid triples found for relation '{relation_name}'. Setting uniform distribution.")
                distrib = np.ones(tail_vocab_size, dtype=np.float64)
            else:
                print(f"      Counted {int(np.sum(tail_counts))} occurrences across {num_valid_triples} valid triples.")
                distrib = np.power(tail_counts, 0.75)

            sum_distrib = np.sum(distrib)
            if sum_distrib > 0:
                normalized_distrib = distrib / sum_distrib
            else:
                print(f"      Warning: Sum of distribution is zero for '{relation_name}'. Using uniform.")
                normalized_distrib = np.ones(tail_vocab_size, dtype=np.float64) / tail_vocab_size
            processed_data["distributions"][relation_name] = normalized_distrib
            print(f"      Stored distribution for '{relation_name}' with {len(normalized_distrib)} elements.")

        except KeyError as e:
            print(f"    Error processing distribution for '{relation_name}': Missing key {e}")
        except Exception as e:
            print(f"    Unexpected error calculating distribution for '{relation_name}': {e}")

    print("Finished processing DataFrame.")
    return processed_data


def generate_labels_from_df(dataset_name: str, df: pd.DataFrame, user_map: dict, item_map: dict, mode: str):
    """Generates item interaction labels for users from a DataFrame."""
    print(f"Generating labels for mode: {mode}")
    user_col = "userID"
    item_col = "itemID"
    rating_col = "rating"
    user_items = defaultdict(list)
    user_item_ratings = defaultdict(list)
    processed_interactions = 0
    skipped_interactions = 0

    for _, row in df.iterrows():
        uid_orig = row[user_col]
        iid_orig = row[item_col]
        rating_val = row[rating_col]

        if pd.notna(uid_orig) and pd.notna(iid_orig) and pd.notna(rating_val) and uid_orig in user_map and iid_orig in item_map:
            user_idx = user_map[uid_orig]
            item_idx = item_map[iid_orig]
            user_items[user_idx].append(item_idx)
            user_item_ratings[user_idx].append((item_idx, float(rating_val)))
            processed_interactions += 1
        else:
            skipped_interactions += 1

    final_user_items = {uid: sorted(list(set(items))) for uid, items in user_items.items()}

    final_user_items = {}
    for uid, item_rating_list in user_item_ratings.items():
        seen_items = set()
        dedup_list = []
        for item_idx, rating_val in item_rating_list:
            if item_idx not in seen_items:
                dedup_list.append((item_idx, rating_val))
                seen_items.add(item_idx)
        # Sort by item_idx for consistency (optional)
        final_user_items[uid] = sorted(dedup_list, key=lambda x: x[0])

    print(f"  Generated labels for {len(final_user_items)} users.")
    print(f"  Processed {processed_interactions} interactions, skipped {skipped_interactions}.")
    save_labels(dataset_name, final_user_items, mode=mode)


def preprocess_rl(
    dataset,
    want_col,
    num_rows,
    ratio,
    seed,
    personalization=False,
    fraction_to_change=0,
    change_rating=False,
    privacy=False,
    hide_type="values_in_column",
    columns_to_hide=None,
    fraction_to_hide=0,
    records_to_hide=None,
    force_reprocess=True,
):
    # --- 1. Load Raw Data using Custom Loader ---
    print(f"Loading raw data for dataset: {dataset}")
    try:
        data_df = load_dataframe(
            dataset_name=dataset.lower(),  # loader uses lowercase names
            want_col=want_col,
            num_rows=num_rows,
            seed=seed,
        )
    except KeyError as e:
        print(f"\nError loading DataFrame: Missing required column(s) - {e}")
        print("Please ensure the merge_file generated by your custom loader contains:", want_col)
        return  # Exit if data loading fails fundamentally
    except Exception as e:
        print(f"\nError loading DataFrame: {e}")
        return

    print(f"Loaded DataFrame with shape: {data_df.shape}")
    print(data_df.head())

    # --- 2. Train/Test Split ---
    print("Splitting data into train/test sets...")
    try:
        train_df, test_df = python_stratified_split(data_df, ratio=ratio, col_user="userID", col_item="itemID", seed=seed)
        print("Stratified split successful.")

    except Exception as e:
        print(f"Stratified split failed ({e}), using simple random split.")
        train_df, test_df = train_test_split(
            data_df,
            test_size=1.0 - ratio,
            random_state=seed,
        )
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

    if privacy:
        data_df = dm.hide_information_in_dataframe(
            data=data_df,
            hide_type=hide_type,
            columns_to_hide=columns_to_hide,
            fraction_to_hide=fraction_to_hide,
            records_to_hide=records_to_hide,
            seed=seed,
        )

    if personalization:
        train_df = dm.change_items_in_dataframe(
            all=data_df,
            data=data_df,
            fraction_to_change=fraction_to_change,
            change_rating=change_rating,
            seed=seed,
        )

    # --- 3. Create and Cache Processed Dataset Object ---
    processed_dataset_file = TMP_DIR[dataset] + "/processed_dataset.pkl"
    processed_dataset = None  # Initialize
    if not force_reprocess and os.path.exists(processed_dataset_file):
        print(f"Loading cached processed dataset from {processed_dataset_file}...")
        try:
            with open(processed_dataset_file, "rb") as f:
                processed_dataset = pickle.load(f)
            if not isinstance(processed_dataset, dict) or "entity_maps" not in processed_dataset or "relations" not in processed_dataset:
                print("Cached processed dataset seems invalid or incomplete. Reprocessing...")
                processed_dataset = None
                force_reprocess = True
        except Exception as e:
            print(f"Failed to load cached processed dataset ({e}). Reprocessing...")
            processed_dataset = None
            force_reprocess = True

    if processed_dataset is None:
        print("Creating processed dataset object from DataFrame...")
        if not os.path.isdir(TMP_DIR[dataset]):
            try:
                os.makedirs(TMP_DIR[dataset])
            except OSError as e:
                print(f"Error creating directory {TMP_DIR[dataset]}: {e}")
                return

        processed_dataset = create_processed_dataset(data_df)
        # processed_dataset = create_processed_dataset(train_df)
        print(f"Saving processed dataset object to {processed_dataset_file}...")
        try:
            with open(processed_dataset_file, "wb") as f:
                pickle.dump(processed_dataset, f)
        except Exception as e:
            print(f"Error saving processed dataset to {processed_dataset_file}: {e}")

    # --- 4. Generate and Cache Knowledge Graph ---
    kg_file = TMP_DIR[dataset] + "/kg.pkl"
    if not force_reprocess and os.path.exists(kg_file):
        print(f"Loading cached knowledge graph from {kg_file}...")
    if force_reprocess or not os.path.exists(kg_file):
        print("Creating knowledge graph from processed dataset...")
        if processed_dataset is None:
            print("Error: Cannot create Knowledge Graph because processed_dataset is None.")
            return
        try:
            kg = KnowledgeGraph(processed_dataset)
            print("Computing node degrees...")
            kg.compute_degrees()
            print(f"Saving knowledge graph to {kg_file}...")
            save_kg(dataset, kg)
        except Exception as e:
            print(f"Error during Knowledge Graph creation/saving: {e}")

    # --- 5. Generate Train/Test Labels ---
    train_label_file, test_label_file = LABELS[dataset]
    if force_reprocess or not os.path.exists(train_label_file) or not os.path.exists(test_label_file):
        print("Generating train/test labels from split DataFrames.")
        if processed_dataset and USERID in processed_dataset.get("entity_maps", {}) and ITEMID in processed_dataset.get("entity_maps", {}):
            user_map = processed_dataset["entity_maps"][USERID]["map"]
            item_map = processed_dataset["entity_maps"][ITEMID]["map"]
            try:
                generate_labels_from_df(dataset, train_df, user_map, item_map, "train")
                generate_labels_from_df(dataset, test_df, user_map, item_map, "test")
            except Exception as e:
                print(f"Error generating labels: {e}")
        else:
            print("Error: User or Item map not found in processed_dataset. Cannot generate labels.")

    else:
        print("Train/test labels already exist. Skipping generation.")

    print("Preprocessing finished.")

    return data_df, train_df, test_df
