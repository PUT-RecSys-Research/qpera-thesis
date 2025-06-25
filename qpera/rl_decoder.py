from __future__ import absolute_import, division, print_function

import pickle
from typing import Any, Dict, List, Tuple

import pandas as pd

from .rl_utils import AMAZONSALES, BELONG_TO, DESCRIBED_AS, GENRES, ITEMID, MOVIELENS, POSTRECOMMENDATIONS, PREDICTION, TITLE, TMP_DIR, USERID


class RLRecommenderDecoder:
    """
    Decodes the output of the RL agent's evaluation phase into human-readable
    recommendations and a DataFrame compatible with metrics.py.

    This class handles the conversion from internal indices to original IDs
    and provides detailed item information including titles and genres.
    """

    def __init__(self, dataset_name: str):
        """
        Initialize the decoder by loading necessary mappings.

        Args:
            dataset_name: The name of the dataset (e.g., 'movielens')

        Raises:
            FileNotFoundError: If processed dataset file is not found
            Exception: If dataset loading fails
        """
        processed_dataset_file = f"{TMP_DIR[dataset_name]}/processed_dataset.pkl"
        print(f"Decoder: Loading processed dataset from {processed_dataset_file}")

        try:
            with open(processed_dataset_file, "rb") as f:
                processed_dataset = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: Processed dataset file not found at {processed_dataset_file}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to load processed dataset: {e}")
            raise

        # Load entity mappings
        maps = processed_dataset.get("entity_maps", {})
        self.item_map = maps.get(ITEMID, {}).get("map", {})
        self.item_inv_map = maps.get(ITEMID, {}).get("inv_map", {})
        self.title_inv_map = maps.get(TITLE, {}).get("inv_map", {})
        self.genre_inv_map = maps.get(GENRES, {}).get("inv_map", {})
        self.user_map = maps.get(USERID, {}).get("map", {})
        self.user_inv_map = maps.get(USERID, {}).get("inv_map", {})

        # Build relation mappings
        relations = processed_dataset.get("relations", {})
        self._build_item_title_mapping(relations)
        self._build_item_genre_mapping(relations)

        print("Decoder initialized successfully.")

    def _build_item_title_mapping(self, relations: Dict[str, List[Tuple]]) -> None:
        """Build mapping from item indices to title indices."""
        self.item_to_title_idx = {}
        desc_as = relations.get(DESCRIBED_AS, [])
        for title_idx, item_idx in desc_as:
            self.item_to_title_idx[item_idx] = title_idx

    def _build_item_genre_mapping(self, relations: Dict[str, List[Tuple]]) -> None:
        """Build mapping from item indices to genre indices."""
        self.item_to_genre_idxs = {}
        belongs = relations.get(BELONG_TO, [])
        for item_idx, genre_idx in belongs:
            if item_idx not in self.item_to_genre_idxs:
                self.item_to_genre_idxs[item_idx] = []
            self.item_to_genre_idxs[item_idx].append(genre_idx)

    def get_item_details(self, item_idx: int) -> Dict[str, Any]:
        """
        Retrieve details for a single item index.

        Args:
            item_idx: Internal item index

        Returns:
            Dictionary containing item details (original_id, title, genres, etc.)
        """
        original_id = self.item_inv_map.get(item_idx, f"UNKNOWN_IDX_{item_idx}")

        # Get title
        title_idx = self.item_to_title_idx.get(item_idx)
        title = "UNKNOWN_TITLE"
        if title_idx is not None:
            title = self.title_inv_map.get(title_idx, "UNKNOWN_TITLE")

        # Get genres
        genre_idxs = self.item_to_genre_idxs.get(item_idx, [])
        genres = [self.genre_inv_map.get(g_idx, "UNKNOWN_GENRE") for g_idx in genre_idxs]

        return {
            "item_idx": item_idx,
            "original_id": original_id,
            "title": title,
            "genres": sorted(list(set(genres))),
        }

    def get_user_details(self, user_idx: int) -> Dict[str, Any]:
        """
        Retrieve details for a single user index.

        Args:
            user_idx: Internal user index

        Returns:
            Dictionary containing user details
        """
        original_user_id_val = self.user_inv_map.get(user_idx, f"UNKNOWN_IDX_{user_idx}")
        return {"user_idx": user_idx, "original_userID": original_user_id_val}

    def decode(self, dataset_name: str, pred_labels: Dict[int, List[int]], k: int = 10) -> Tuple[Dict[int, Dict], pd.DataFrame]:
        """
        Decode predicted item indices for users into human-readable recommendations.

        Args:
            dataset_name: Name of the dataset
            pred_labels: Dictionary {user_idx: [item_idx_list_ascending_score]}.
                        The list MUST be sorted lowest score first, as generated
                        by the original evaluate_paths function.
            k: The number of top recommendations to consider

        Returns:
            Tuple containing:
            - human_readable_recs: Dictionary with user recommendations
            - rating_pred_df: DataFrame compatible with metrics.py
        """
        human_readable_recs = {}
        pred_df_data = []

        print(f"Decoder: Decoding recommendations for {len(pred_labels)} users...")

        for user_idx, ascending_item_idxs in pred_labels.items():
            user_recs = self._process_user_recommendations(user_idx, ascending_item_idxs, k, pred_df_data)
            human_readable_recs[user_idx] = user_recs

        # Create and format DataFrame
        rating_pred_df = pd.DataFrame(pred_df_data)
        if not rating_pred_df.empty:
            rating_pred_df = self._format_dataframe_columns(rating_pred_df, dataset_name)

        print(f"Decoder: Finished decoding. Generated DataFrame with {len(rating_pred_df)} rows.")
        if not rating_pred_df.empty:
            print(f"Decoder: rating_pred_df dtypes:\n{rating_pred_df.dtypes.to_string()}")

        return human_readable_recs, rating_pred_df

    def _process_user_recommendations(self, user_idx: int, ascending_item_idxs: List[int], k: int, pred_df_data: List[Dict]) -> Dict[str, Any]:
        """Process recommendations for a single user."""
        user_details = self.get_user_details(user_idx)
        original_userID_for_human_recs = user_details["original_userID"]

        # Reverse list to get highest score first and take top K
        ranked_item_idxs = ascending_item_idxs[::-1]
        top_k_item_idxs = ranked_item_idxs[:k]

        # Handle empty recommendations
        if not top_k_item_idxs:
            print(f"Warning: No recommendations found for user_idx {user_idx} (Original ID: {original_userID_for_human_recs})")
            return {
                "original_userID": original_userID_for_human_recs,
                "recommendations": [],
            }

        # Generate recommendations
        max_rank_score = len(top_k_item_idxs)
        current_user_recommendations_list = []

        for rank, item_idx in enumerate(top_k_item_idxs, 1):
            item_details = self.get_item_details(item_idx)
            item_details["rank"] = rank
            current_user_recommendations_list.append(item_details)

            # Add to DataFrame data
            pred_score = float(max_rank_score - rank + 1)
            pred_df_data.append(
                {
                    USERID: user_idx,
                    "userID": original_userID_for_human_recs,
                    ITEMID: item_idx,
                    "itemID": item_details["original_id"],
                    PREDICTION: pred_score,
                }
            )

        return {
            "original_userID": original_userID_for_human_recs,
            "recommendations": current_user_recommendations_list,
        }

    def _format_dataframe_columns(self, rating_pred_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Format DataFrame columns with appropriate data types."""
        # Format internal index columns
        if USERID in rating_pred_df.columns:
            rating_pred_df[USERID] = rating_pred_df[USERID].astype(int)
        if ITEMID in rating_pred_df.columns:
            rating_pred_df[ITEMID] = rating_pred_df[ITEMID].astype(int)
        if PREDICTION in rating_pred_df.columns:
            rating_pred_df[PREDICTION] = rating_pred_df[PREDICTION].astype(float)

        # Format original ID columns based on dataset
        self._format_user_id_column(rating_pred_df, dataset_name)
        self._format_item_id_column(rating_pred_df, dataset_name)

        return rating_pred_df

    def _format_user_id_column(self, rating_pred_df: pd.DataFrame, dataset_name: str) -> None:
        """Format userID column based on dataset requirements."""
        if "userID" not in rating_pred_df.columns:
            return

        if dataset_name in [AMAZONSALES, POSTRECOMMENDATIONS]:
            # UserIDs are strings for these datasets
            rating_pred_df["userID"] = rating_pred_df["userID"].astype(str)
        else:
            # Movielens userID original == int
            try:
                rating_pred_df["userID"] = rating_pred_df["userID"].astype(int)
            except ValueError:
                print(f"Warning (Decoder): Could not cast 'userID' to int for {dataset_name}. Keeping as string.")
                rating_pred_df["userID"] = rating_pred_df["userID"].astype(str)

    def _format_item_id_column(self, rating_pred_df: pd.DataFrame, dataset_name: str) -> None:
        """Format itemID column based on dataset requirements."""
        if "itemID" not in rating_pred_df.columns:
            return

        if dataset_name == AMAZONSALES:
            # Amazon itemIDs are strings
            rating_pred_df["itemID"] = rating_pred_df["itemID"].astype(str)
        elif dataset_name in [POSTRECOMMENDATIONS, MOVIELENS]:
            # Original itemIDs are integers for these datasets
            try:
                rating_pred_df["itemID"] = rating_pred_df["itemID"].astype(int)
            except ValueError:
                print(f"Warning (Decoder): Could not cast 'itemID' to int for {dataset_name}. Keeping as string.")
                rating_pred_df["itemID"] = rating_pred_df["itemID"].astype(str)
