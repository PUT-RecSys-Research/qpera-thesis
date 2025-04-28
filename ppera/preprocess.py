from __future__ import absolute_import, division, print_function

# from recommenders.datasets.python_splitters import python_stratified_split

import os
import pickle
import argparse
import pandas as pd
from collections import defaultdict
# Removed EasyDict import - we will use standard dicts
# from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from recommenders.datasets.python_splitters import python_stratified_split

# Import your custom loader and the modified utils
from datasets_loader import loader as load_dataframe # Renamed to avoid conflict
from utils import *
# We need to modify KnowledgeGraph class as well
# Let's assume the KnowledgeGraph class is modified as discussed below
# and potentially moved to a different file or adapted here.

# --- KNOWLEDGE GRAPH CLASS (Adapted for Standard Dicts) ---

class KnowledgeGraph:
    """
    Adapted KnowledgeGraph class to build from standard dictionaries.
    """
    def __init__(self, processed_dataset):
        self.G = dict()
        # Ensure processed_dataset is a dict before accessing keys
        if not isinstance(processed_dataset, dict):
             raise TypeError("processed_dataset must be a dictionary.")
        self._load_entities(processed_dataset)
        self._load_relations(processed_dataset)
        self._clean()
        # self.compute_degrees() # Can be called separately after saving/loading

    def _load_entities(self, dataset):
        print('Load entities...')
        # Access using dictionary keys
        self.entity_maps = dataset.get('entity_maps', {})
        num_nodes = 0
        for entity in get_entities(): # Uses new entities from utils.py
            self.G[entity] = {}
            # Get vocab_size safely using .get()
            entity_map_data = self.entity_maps.get(entity, {})
            vocab_size = entity_map_data.get('vocab_size', 0)

            if vocab_size == 0:
                 print(f"  Warning: Vocab size is 0 for entity '{entity}'. Skipping graph initialization for it.")
                 continue

            # Initialize graph structure for integer indices
            for idx in range(vocab_size):
                self.G[entity][idx] = {r: [] for r in get_relations(entity)}
            num_nodes += vocab_size
            print(f'  Loaded {entity} with {vocab_size} nodes.')
        print(f'Total {num_nodes} nodes.')

    def _load_relations(self, dataset):
        print('Load relations...')
        # Access processed relations stored during dataset creation
        relations_dict = dataset.get('relations', {})
        for relation_name, relation_data in relations_dict.items():
            print(f'  Processing relation: {relation_name} ({len(relation_data)} edges)')
            num_edges = 0
            # relation_data is expected to be a list of (head_idx, tail_idx) tuples
            try:
                head_entity_type = self._get_head_entity_type(relation_name)
                tail_entity_type = get_entity_tail(head_entity_type, relation_name)
            except ValueError as e:
                print(f"  Warning: Skipping relation '{relation_name}'. Error finding head/tail types: {e}")
                continue
            except KeyError as e:
                 print(f"  Warning: Skipping relation '{relation_name}'. Entity type missing in KG_RELATION: {e}")
                 continue


            head_map_data = self.entity_maps.get(head_entity_type, {})
            tail_map_data = self.entity_maps.get(tail_entity_type, {})
            head_vocab_size = head_map_data.get('vocab_size', 0)
            tail_vocab_size = tail_map_data.get('vocab_size', 0)

            if head_vocab_size == 0 or tail_vocab_size == 0:
                 print(f"  Warning: Skipping relation '{relation_name}' due to zero vocab size for head or tail.")
                 continue

            for head_idx, tail_idx in relation_data:
                 # Ensure indices are valid integers and within bounds
                if isinstance(head_idx, int) and isinstance(tail_idx, int) and \
                   0 <= head_idx < head_vocab_size and 0 <= tail_idx < tail_vocab_size:
                    self._add_edge(head_entity_type, head_idx, relation_name, tail_entity_type, tail_idx)
                    num_edges += 2 # Counting both directions
                else:
                     print(f"  Warning: Invalid indices found for relation {relation_name}: head={head_idx} (type={type(head_idx)}), tail={tail_idx} (type={type(tail_idx)}). Expected ints within bounds (Head Size: {head_vocab_size}, Tail Size: {tail_vocab_size})")

            print(f'    Added {num_edges} edges for {relation_name}.')

    def _get_head_entity_type(self, relation_name):
        # Infer head entity type based on KG_RELATION schema
        for head_type, relations in KG_RELATION.items():
            if relation_name in relations:
                return head_type
        raise ValueError(f"Could not determine head entity type for relation: {relation_name}")


    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
         # Check if entity types exist and indices are valid before adding edge
        if etype1 in self.G and etype2 in self.G and \
           eid1 in self.G[etype1] and eid2 in self.G[etype2]:
            # Check if relation is valid for the head entity type
            if relation in self.G[etype1][eid1]:
                self.G[etype1][eid1][relation].append(eid2)
            # Check if relation is valid for the tail entity type (for inverse)
            # Note: KG_RELATION defines direction, inverse relation might not be the same name
            # Assuming symmetric relation addition for now, as per original code structure
            if relation in self.G[etype2][eid2]:
                self.G[etype2][eid2][relation].append(eid1)
        # else: # Optional logging for debugging
        #    print(f"Debug: Edge add skipped. Type1={etype1}, ID1={eid1}, Type2={etype2}, ID2={eid2}. Check if keys/indices exist in self.G.")


    def _clean(self):
        print('Cleaning graph: Removing duplicate edges...')
        for etype in self.G:
            for eid in self.G.get(etype, {}): # Use .get for safety
                for r in self.G[etype][eid]:
                    # Use list(set(...)) for fast deduplication and convert back to list
                    unique_neighbors = list(set(self.G[etype][eid][r]))
                    # Sort for consistency (optional, but good practice)
                    self.G[etype][eid][r] = sorted(unique_neighbors)

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G.get(etype, {}): # Use .get for safety
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count
        print('Node degrees computed.')

    # Optional: Add back getter methods if needed by other parts (like kg_env)
    def get(self, eh_type, eh_id=None, relation=None):
        # Basic getter, add error handling as needed
        data = self.G
        if eh_type is not None:
            data = data.get(eh_type, {})
        if eh_id is not None:
            # Ensure eh_id is treated as an integer index if it's numeric-like
            try:
                eh_id_int = int(eh_id)
                data = data.get(eh_id_int, {})
            except (ValueError, TypeError):
                 data = {} # Or handle non-integer IDs differently if needed
        if relation is not None:
            data = data.get(relation, [])
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

# --- END OF Adapted KnowledgeGraph Class ---


# Return type is now dict
def create_processed_dataset(df: pd.DataFrame) -> dict:
    """
    Processes the DataFrame to create entity vocabularies/mappings
    and structured relation data. Returns a standard dictionary.
    """
    print("Processing DataFrame to create dataset object...")
    # Use standard Python dictionaries
    processed_data = {'entity_maps': {}, 'relations': {}}

    # Define columns corresponding to entities
    entity_columns = {
        USERID: 'userID',
        ITEMID: 'itemID',
        TITLE: 'title',
        GENRES: 'genres',
    }

    # --- Create Entity Mappings ---
    print("  Creating entity mappings...")
    for entity_name, col_name in entity_columns.items():
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found for entity '{entity_name}'. Skipping.")
            continue

        # Handle potential NaN/missing values before processing
        valid_series = df[col_name].dropna()

        if entity_name == GENRES:
            # Split genres string into individual genres
            all_values = set(g for genres_list in valid_series.astype(str) for g in genres_list.split() if g) # Ensure non-empty strings
        else:
            # Get unique values for other entities
            all_values = set(valid_series.unique())

        vocab = sorted(list(all_values))
        vocab_size = len(vocab)
        # Create mappings
        original_to_idx = {val: i for i, val in enumerate(vocab)}
        idx_to_original = {i: val for i, val in enumerate(vocab)}

        # Store the inner data as a standard dictionary
        processed_data['entity_maps'][entity_name] = {
            'vocab': vocab,
            'map': original_to_idx,
            'inv_map': idx_to_original, # Standard dict
            'vocab_size': vocab_size
        }
        print(f"    Entity '{entity_name}': {vocab_size} unique values.")

    # --- Extract Relations ---
    print("  Extracting relations...")

    # Helper to safely get mapped index
    def get_idx(entity_type, value, maps):
        # Check if entity_type exists and value is not NaN before map lookup
        if entity_type in maps and pd.notna(value) and value in maps[entity_type]['map']:
            return maps[entity_type]['map'][value]
        return None

    # 1. PURCHASE (USERID -> ITEMID)
    if USERID in processed_data['entity_maps'] and ITEMID in processed_data['entity_maps']:
        purchase_relations = set() # Use set for efficient deduplication
        for _, row in df.iterrows():
            uid_idx = get_idx(USERID, row[entity_columns[USERID]], processed_data['entity_maps'])
            iid_idx = get_idx(ITEMID, row[entity_columns[ITEMID]], processed_data['entity_maps'])
            if uid_idx is not None and iid_idx is not None:
                purchase_relations.add((uid_idx, iid_idx))
        processed_data['relations'][PURCHASE] = list(purchase_relations)
        print(f"    Relation '{PURCHASE}': {len(processed_data['relations'][PURCHASE])} unique interactions.")

    # 2. BELONG_TO (ITEMID -> GENRES)
    if ITEMID in processed_data['entity_maps'] and GENRES in processed_data['entity_maps']:
        belong_to_relations = set()
        item_genre_df = df[[entity_columns[ITEMID], entity_columns[GENRES]]].drop_duplicates(subset=[entity_columns[ITEMID]]).dropna()
        for _, row in item_genre_df.iterrows():
            item_idx = get_idx(ITEMID, row[entity_columns[ITEMID]], processed_data['entity_maps'])
            if item_idx is not None:
                genres_str = str(row[entity_columns[GENRES]])
                for genre in genres_str.split():
                    if genre: # Check for empty string after split
                        genre_idx = get_idx(GENRES, genre, processed_data['entity_maps'])
                        if genre_idx is not None:
                            belong_to_relations.add((item_idx, genre_idx))
        processed_data['relations'][BELONG_TO] = list(belong_to_relations)
        print(f"    Relation '{BELONG_TO}': {len(processed_data['relations'][BELONG_TO])} unique item-genre links.")

    # 3. DESCRIBED_AS (TITLE -> ITEMID) - Assuming one primary title per item
    if TITLE in processed_data['entity_maps'] and ITEMID in processed_data['entity_maps']:
        described_as_relations = set()
        item_title_df = df[[entity_columns[ITEMID], entity_columns[TITLE]]].drop_duplicates(subset=[entity_columns[ITEMID]]).dropna()
        for _, row in item_title_df.iterrows():
             title_idx = get_idx(TITLE, row[entity_columns[TITLE]], processed_data['entity_maps'])
             item_idx = get_idx(ITEMID, row[entity_columns[ITEMID]], processed_data['entity_maps'])
             if title_idx is not None and item_idx is not None:
                 described_as_relations.add((title_idx, item_idx))
        processed_data['relations'][DESCRIBED_AS] = list(described_as_relations)
        print(f"    Relation '{DESCRIBED_AS}': {len(processed_data['relations'][DESCRIBED_AS])} unique title-item links.")

    # 4. MENTION (USERID -> TITLE) - Simplified
    if USERID in processed_data['entity_maps'] and TITLE in processed_data['entity_maps'] and ITEMID in processed_data['entity_maps']:
        mention_relations = set()
        # Need item-to-title mapping first
        item_to_title = {}
        item_title_df = df[[entity_columns[ITEMID], entity_columns[TITLE]]].drop_duplicates(subset=[entity_columns[ITEMID]]).dropna()
        for _, row in item_title_df.iterrows():
            item_idx = get_idx(ITEMID, row[entity_columns[ITEMID]], processed_data['entity_maps'])
            title_idx = get_idx(TITLE, row[entity_columns[TITLE]], processed_data['entity_maps'])
            if item_idx is not None and title_idx is not None:
                 item_to_title[item_idx] = title_idx

        # Now link user to title via item interaction
        # Use the purchase relations if they exist
        purchase_rel_data = processed_data.get('relations',{}).get(PURCHASE, [])
        for user_idx, item_idx in purchase_rel_data:
             if item_idx in item_to_title:
                 title_idx = item_to_title[item_idx]
                 mention_relations.add((user_idx, title_idx))

        processed_data['relations'][MENTION] = list(mention_relations)
        print(f"    Relation '{MENTION}': {len(processed_data['relations'][MENTION])} unique user-title links (derived).")


    # --- Calculate Tail Entity Distributions for Negative Sampling ---
    print("  Calculating tail entity distributions for negative sampling...")
    processed_data['distributions'] = {}

    for relation_name, relation_data in processed_data.get('relations', {}).items():
        print(f"    Calculating distribution for: {relation_name}")
        try:
            # Determine the tail entity type for this relation
            head_entity_type = None # Find the head type first
            for h_type, rels in KG_RELATION.items():
                if relation_name in rels:
                    head_entity_type = h_type
                    break
            if head_entity_type is None:
                print(f"      Warning: Could not find head entity for relation '{relation_name}'. Skipping distribution calculation.")
                continue

            tail_entity_type = get_entity_tail(head_entity_type, relation_name)
            tail_map_data = processed_data.get('entity_maps', {}).get(tail_entity_type, {})
            tail_vocab_size = tail_map_data.get('vocab_size', 0)

            if tail_vocab_size == 0:
                 print(f"      Warning: Tail vocab size is 0 for relation '{relation_name}'. Skipping distribution calculation.")
                 continue

            # Count frequencies of tail entities
            tail_counts = np.zeros(tail_vocab_size, dtype=np.float64)
            num_valid_triples = 0
            for _, tail_idx in relation_data: # We only need the tail index here
                 if 0 <= tail_idx < tail_vocab_size: # Check index validity
                     tail_counts[tail_idx] += 1
                     num_valid_triples += 1
                 # else: # Optional logging for invalid indices
                 #     print(f"      Debug: Invalid tail index {tail_idx} encountered for relation {relation_name}")

            if num_valid_triples == 0:
                 print(f"      Warning: No valid triples found for relation '{relation_name}'. Setting uniform distribution.")
                 # Fallback to uniform distribution if no data or all indices were invalid
                 distrib = np.ones(tail_vocab_size, dtype=np.float64)
            else:
                 print(f"      Counted {int(np.sum(tail_counts))} occurrences across {num_valid_triples} valid triples.")
                 # Apply smoothing (power 0.75) - standard practice
                 distrib = np.power(tail_counts, 0.75)

            # Normalize to get a probability distribution
            sum_distrib = np.sum(distrib)
            if sum_distrib > 0:
                normalized_distrib = distrib / sum_distrib
            else:
                 # Handle case where all counts were zero after smoothing (or initially)
                 print(f"      Warning: Sum of distribution is zero for '{relation_name}'. Using uniform.")
                 normalized_distrib = np.ones(tail_vocab_size, dtype=np.float64) / tail_vocab_size

            # Store the final distribution (as numpy array)
            processed_data['distributions'][relation_name] = normalized_distrib
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
    user_col = 'userID' # Standardized column name
    item_col = 'itemID' # Standardized column name
    user_items = defaultdict(list)
    processed_interactions = 0
    skipped_interactions = 0

    for _, row in df.iterrows():
        uid_orig = row[user_col]
        iid_orig = row[item_col]

        # Map original IDs to integer indices
        # Check for NaN before map lookup
        if pd.notna(uid_orig) and pd.notna(iid_orig) and uid_orig in user_map and iid_orig in item_map:
            user_idx = user_map[uid_orig]
            item_idx = item_map[iid_orig]
            user_items[user_idx].append(item_idx)
            processed_interactions += 1
        else:
             skipped_interactions += 1
             # Optional detailed logging:
             # if pd.isna(uid_orig) or pd.isna(iid_orig):
             #      print(f"Skipping label interaction due to NaN: User '{uid_orig}', Item '{iid_orig}'")
             # elif uid_orig not in user_map:
             #      print(f"Skipping label interaction: User '{uid_orig}' not in map.")
             # elif iid_orig not in item_map:
             #       print(f"Skipping label interaction: Item '{iid_orig}' not in map.")


    # Deduplicate items for each user
    final_user_items = {uid: sorted(list(set(items))) for uid, items in user_items.items()}

    print(f"  Generated labels for {len(final_user_items)} users.")
    print(f"  Processed {processed_interactions} interactions, skipped {skipped_interactions}.")
    save_labels(dataset_name, final_user_items, mode=mode)


def main():
    parser = argparse.ArgumentParser()
    # Updated dataset choices
    parser.add_argument('--dataset', type=str, default=MOVIELENS,
                        help=f'One of {{{MOVIELENS}, {AMAZONSALES}, {POSTRECOMMENDATIONS}}}.')
    # Add args for data loading if needed (e.g., num_rows, seed)
    parser.add_argument('--num_rows', type=int, default=None, help='Number of rows to load (optional).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling and splitting.')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of data for the test set.')
    parser.add_argument('--ratio', type=float, default=0.75, help='Fraction of data for the test set.')
    # Add flag to force reprocessing even if cached files exist
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocessing even if cached files exist.')

    args = parser.parse_args()

    # Set seed
    set_random_seed(args.seed)



    # --- 1. Load Raw Data using Custom Loader ---
    print(f"Loading raw data for dataset: {args.dataset}")
    # Define required columns based on new schema
    # Adjust based on exactly what's needed downstream, timestamp is good for splitting
    required_columns = ['userID', 'itemID', 'rating', 'timestamp', 'title', 'genres']
    # Load the dataframe
    try:
        # data_df = load_dataframe(dataset_name=args.dataset.lower(), # loader uses lowercase names
        #                         want_col=required_columns,
        #                         num_rows=args.num_rows,
        #                         seed=args.seed)
        data_df = load_dataframe(dataset_name=args.dataset.lower(), # loader uses lowercase names
                                want_col=required_columns,
                                num_rows=10000,
                                seed=42)
    except KeyError as e:
         print(f"\nError loading DataFrame: Missing required column(s) - {e}")
         print("Please ensure the merge_file generated by your custom loader contains:", required_columns)
         return # Exit if data loading fails fundamentally
    except Exception as e:
         print(f"\nError loading DataFrame: {e}")
         return

    print(f"Loaded DataFrame with shape: {data_df.shape}")
    print(data_df.head())


    # --- 2. Create and Cache Processed Dataset Object ---
    processed_dataset_file = TMP_DIR[args.dataset] + '/processed_dataset.pkl'
    processed_dataset = None # Initialize
    if not args.force_reprocess and os.path.exists(processed_dataset_file):
        print(f"Loading cached processed dataset from {processed_dataset_file}...")
        try:
            with open(processed_dataset_file, 'rb') as f:
                 processed_dataset = pickle.load(f)
            # Basic validation after loading
            if not isinstance(processed_dataset, dict) or 'entity_maps' not in processed_dataset or 'relations' not in processed_dataset:
                 print("Cached processed dataset seems invalid or incomplete. Reprocessing...")
                 processed_dataset = None # Invalidate cache
                 args.force_reprocess = True # Force reprocessing if cache is bad
        except Exception as e:
            print(f"Failed to load cached processed dataset ({e}). Reprocessing...")
            processed_dataset = None # Invalidate cache
            args.force_reprocess = True # Force reprocessing if loading fails

    # Reprocess if forced or loading failed/invalid
    if processed_dataset is None:
        print("Creating processed dataset object from DataFrame...")
        if not os.path.isdir(TMP_DIR[args.dataset]):
            try:
                os.makedirs(TMP_DIR[args.dataset])
            except OSError as e:
                 print(f"Error creating directory {TMP_DIR[args.dataset]}: {e}")
                 return # Cannot proceed without temp dir

        processed_dataset = create_processed_dataset(data_df)
        # Save the processed dataset object
        print(f"Saving processed dataset object to {processed_dataset_file}...")
        try:
            with open(processed_dataset_file, 'wb') as f:
                pickle.dump(processed_dataset, f)
        except Exception as e:
             print(f"Error saving processed dataset to {processed_dataset_file}: {e}")
             # Continue without saving, but maybe warn user?


    # --- 3. Train/Test Split ---
    print("Splitting data into train/test sets...")
    # Simple random split for now, consider time-based split if 'timestamp' is reliable
    try:
        train_df, test_df = python_stratified_split(
            data_df,
            ratio=args.ratio,      # Corresponds to train set size (e.g., 0.75 for 75% train)
            col_user='userID',     # Pass the string name 'userID'
            col_item='itemID',     # Pass the string name 'itemID'
            seed=args.seed
        )
        print("Stratified split successful.")

    except Exception as e:
         print('####################################################################################################################')
         print(f"Stratified split failed ({e}), using simple random split.")
         train_df, test_df = train_test_split(
            data_df,
            test_size=args.test_split,
            random_state=args.seed,
        )
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

    # --- 4. Generate and Cache Knowledge Graph ---
    kg_file = TMP_DIR[args.dataset] + '/kg.pkl'
    if not args.force_reprocess and os.path.exists(kg_file):
        print(f"Loading cached knowledge graph from {kg_file}...")
        # Optional: Load and validate KG if needed
        # try:
        #    kg = load_kg(args.dataset)
        # except Exception as e:
        #    print(f"Failed to load KG cache ({e}). Will regenerate.")
        #    args.force_reprocess = True # Force KG regen if cache load fails
    # Regenerate if forced or cache doesn't exist / failed to load
    if args.force_reprocess or not os.path.exists(kg_file):
        print('Creating knowledge graph from processed dataset...')
        if processed_dataset is None:
             print("Error: Cannot create Knowledge Graph because processed_dataset is None.")
             return
        try:
            kg = KnowledgeGraph(processed_dataset) # Use the adapted KG class
            print("Computing node degrees...")
            kg.compute_degrees() # Compute degrees after graph creation
            print(f"Saving knowledge graph to {kg_file}...")
            save_kg(args.dataset, kg) # Saves to kg.pkl
        except Exception as e:
             print(f"Error during Knowledge Graph creation/saving: {e}")
             # Decide how to handle this - maybe exit?

    # --- 5. Generate Train/Test Labels ---
    # Check if labels exist, regenerate if force_reprocess is True
    train_label_file, test_label_file = LABELS[args.dataset]
    if args.force_reprocess or not os.path.exists(train_label_file) or not os.path.exists(test_label_file):
        print('Generating train/test labels from split DataFrames.')
        # Pass the necessary maps from the processed_dataset
        # Make sure maps exist before accessing them
        if processed_dataset and USERID in processed_dataset.get('entity_maps', {}) and ITEMID in processed_dataset.get('entity_maps', {}):
            user_map = processed_dataset['entity_maps'][USERID]['map']
            item_map = processed_dataset['entity_maps'][ITEMID]['map']
            try:
                generate_labels_from_df(args.dataset, train_df, user_map, item_map, 'train')
                generate_labels_from_df(args.dataset, test_df, user_map, item_map, 'test')
            except Exception as e:
                 print(f"Error generating labels: {e}")
        else:
            print("Error: User or Item map not found in processed_dataset. Cannot generate labels.")

    else:
        print("Train/test labels already exist. Skipping generation.")

    print("Preprocessing finished.")


if __name__ == '__main__':
    main()