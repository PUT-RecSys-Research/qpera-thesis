import pickle
import pandas as pd
from rl_utils import *

class RLRecommenderDecoder:
    """
    Decodes the output of the RL agent's evaluation phase into human-readable
    recommendations and a DataFrame compatible with metrics.py.
    """
    def __init__(self, dataset_name: str):
        """
        Initializes the decoder by loading necessary mappings.

        Args:
            dataset_name (str): The name of the dataset (e.g., 'movielens').
        """
        processed_dataset_file = TMP_DIR[dataset_name] + '/processed_dataset.pkl'
        print(f"Decoder: Loading processed dataset from {processed_dataset_file}")
        try:
            with open(processed_dataset_file, 'rb') as f:
                processed_dataset = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: Processed dataset file not found at {processed_dataset_file}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to load processed dataset: {e}")
            raise

        maps = processed_dataset.get('entity_maps', {})
        self.item_map = maps.get(ITEMID, {}).get('map', {})
        self.item_inv_map = maps.get(ITEMID, {}).get('inv_map', {})
        self.title_inv_map = maps.get(TITLE, {}).get('inv_map', {})
        self.genre_inv_map = maps.get(GENRES, {}).get('inv_map', {})
        self.user_map = maps.get(USERID, {}).get('map', {})
        self.user_inv_map = maps.get(USERID, {}).get('inv_map', {})

        relations = processed_dataset.get('relations', {})
        self.item_to_title_idx = {}
        desc_as = relations.get(DESCRIBED_AS, [])
        for title_idx, item_idx in desc_as:
            self.item_to_title_idx[item_idx] = title_idx

        self.item_to_genre_idxs = {}
        belongs = relations.get(BELONG_TO, [])
        for item_idx, genre_idx in belongs:
            if item_idx not in self.item_to_genre_idxs:
                self.item_to_genre_idxs[item_idx] = []
            self.item_to_genre_idxs[item_idx].append(genre_idx)

        print("Decoder initialized successfully.")


    def get_item_details(self, item_idx: int) -> dict:
        """Retrieves details for a single item index."""
        original_id = self.item_inv_map.get(item_idx, f"UNKNOWN_IDX_{item_idx}")

        title_idx = self.item_to_title_idx.get(item_idx)
        title = self.title_inv_map.get(title_idx, "UNKNOWN_TITLE") if title_idx is not None else "UNKNOWN_TITLE"

        genre_idxs = self.item_to_genre_idxs.get(item_idx, [])
        genres = [self.genre_inv_map.get(g_idx, "UNKNOWN_GENRE") for g_idx in genre_idxs]

        return {
            'item_idx': item_idx,
            'original_id': original_id,
            'title': title,
            'genres': sorted(list(set(genres)))
        }
    
    def get_user_details(self, user_idx: int) -> dict:
        """Retrieves details for a single user index."""
        original_user_id_val = self.user_inv_map.get(user_idx, f"UNKNOWN_IDX_{user_idx}")
        return {
            'user_idx': user_idx,
            'original_userID': original_user_id_val
        }

    def decode(self, dataset_name, pred_labels: dict, k: int = 10) -> tuple[dict, pd.DataFrame]:
        """
        Decodes the predicted item indices for users.

        Args:
            pred_labels (dict): Dictionary {user_idx: [item_idx_list_ascending_score]}.
                                The list MUST be sorted lowest score first, as generated
                                by the original evaluate_paths function.
            k (int): The number of top recommendations to consider.

        Returns:
            tuple:
                - human_readable_recs (dict): {user_idx: [{'item_idx':.., 'original_id':.., 'title':.., 'genres':.., 'rank':..}, ...]}
                                             Sorted highest rank first.
                - rating_pred_df (pd.DataFrame): DataFrame with columns ['userID', 'itemID', 'prediction']
                                                  compatible with metrics.py. itemID here is the item_idx.
        """
        human_readable_recs = {}
        pred_df_data = []

        print(f"Decoder: Decoding recommendations for {len(pred_labels)} users...")

        for user_idx, ascending_item_idxs in pred_labels.items():
            
            # 1. Reverse the list to get highest score first
            user_details = self.get_user_details(user_idx)
            original_userID_for_human_recs = user_details['original_userID']


            # 2. Keep only top K
            ranked_item_idxs = ascending_item_idxs[::-1]
            top_k_item_idxs = ranked_item_idxs[:k]

            # 3. Generate human-readable output and DataFrame data
            if not top_k_item_idxs:
                 print(f"Warning: No recommendations found for user_idx {user_idx} (Original ID: {original_userID_for_human_recs})")
                 human_readable_recs[user_idx] = {'original_userID': original_userID_for_human_recs, 'recommendations': []}
                 continue

            max_rank_score = len(top_k_item_idxs)
            current_user_recommendations_list = [] # For human_readable_recs value
            for rank, item_idx in enumerate(top_k_item_idxs, 1):
                item_details = self.get_item_details(item_idx)
                item_details['rank'] = rank
                current_user_recommendations_list.append(item_details)

                pred_score = float(max_rank_score - rank + 1)
                pred_df_data.append({
                    USERID: user_idx,
                    'userID': original_userID_for_human_recs,
                    ITEMID: item_idx,
                    'itemID': item_details['original_id'],
                    PREDICTION: pred_score
                })
            
            # 4. Create DataFrame
            human_readable_recs[user_idx] = {
                'original_userID': original_userID_for_human_recs,
                'recommendations': current_user_recommendations_list
            }

        rating_pred_df = pd.DataFrame(pred_df_data)
        if not rating_pred_df.empty:
             # Columns for internal integer indices (keys are from rl_utils.py constants)
            if USERID in rating_pred_df.columns: # e.g. 'user_id'
                 rating_pred_df[USERID] = rating_pred_df[USERID].astype(int)
            if ITEMID in rating_pred_df.columns: # e.g. 'item_id'
                 rating_pred_df[ITEMID] = rating_pred_df[ITEMID].astype(int)
            if PREDICTION in rating_pred_df.columns:
                 rating_pred_df[PREDICTION] = rating_pred_df[PREDICTION].astype(float)

             # Columns for original IDs (keys are literal strings 'userID', 'itemID')
             # These should be strings for AmazonSales, integers for Movielens (if present and different)
            if 'userID' in rating_pred_df.columns:
                if dataset_name == AMAZONSALES or dataset_name == POSTRECOMMENDATIONS: # UserIDs are strings
                    rating_pred_df['userID'] = rating_pred_df['userID'].astype(str)
                else: # Movielens userID original == int
                    try:
                        rating_pred_df['userID'] = rating_pred_df['userID'].astype(int)
                    except ValueError:
                        print(f"Warning (Decoder): Could not cast 'userID' (original) to int for {dataset_name}. Keeping as object/string.")
                        rating_pred_df['userID'] = rating_pred_df['userID'].astype(str)

            if 'itemID' in rating_pred_df.columns:
                if dataset_name == AMAZONSALES: # Amazon itemIDs are strings
                    rating_pred_df['itemID'] = rating_pred_df['itemID'].astype(str)
                # For PostRecommendations and Movielens, original itemIDs are integers
                elif dataset_name == POSTRECOMMENDATIONS or dataset_name == MOVIELENS: 
                    try:
                        # item_details['original_id'] should be providing integers for these datasets
                        rating_pred_df['itemID'] = rating_pred_df['itemID'].astype(int)
                    except ValueError:
                        print(f"Warning (Decoder): Could not cast 'itemID' (original) to int for {dataset_name}. Keeping as object/string.")
                        rating_pred_df['itemID'] = rating_pred_df['itemID'].astype(str)

        print(f"Decoder: Finished decoding. Generated DataFrame with {len(rating_pred_df)} rows.")
        if not rating_pred_df.empty:
            print(f"Decoder: rating_pred_df dtypes:\n{rating_pred_df.dtypes.to_string()}")
        return human_readable_recs, rating_pred_df