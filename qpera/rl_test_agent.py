from __future__ import absolute_import, division, print_function

import argparse
import os
import pickle
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from recommenders.evaluation.python_evaluation import (
    mae,
    ndcg_at_k,
    rmse,
)
from tqdm import tqdm

from . import log_mlflow
from .metrics import (
    intra_list_dissimilarity,
    intra_list_similarity_score,
    item_coverage,
    mrr,
    personalization_score,
    precision_at_k,
    recall_at_k,
    user_coverage,
)
from .rl_decoder import RLRecommenderDecoder
from .rl_kg_env import BatchKGEnvironment
from .rl_prediction import calculate_predictons
from .rl_train_agent import ActorCritic
from .rl_utils import (
    AMAZONSALES,
    ITEMID,
    KG_RELATION,
    MOVIELENS,
    POSTRECOMMENDATIONS,
    RATING,
    SELF_LOOP,
    TMP_DIR,
    USERID,
    WATCHED,
    load_embed,
    load_labels,
    set_random_seed,
)


def batch_beam_search(
    env: BatchKGEnvironment, model: ActorCritic, uids: List[int], device: torch.device, topk: List[int] = [25, 5, 1]
) -> Tuple[List[List[Tuple]], List[float]]:
    """
    Perform batch beam search to generate recommendation paths.

    Args:
        env: Batch knowledge graph environment
        model: Trained actor-critic model
        uids: List of user IDs to generate paths for
        device: PyTorch device for computation
        topk: Top-k values for each hop in beam search

    Returns:
        Tuple of (paths, probabilities) for all generated paths
    """

    def _batch_acts_to_masks(batch_acts: List[List[Tuple]]) -> np.ndarray:
        """Convert batch actions to boolean masks."""
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=bool)
            act_mask[:num_acts] = True
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    # Initialize beam search
    state_pool = env.reset(uids)
    path_pool = env._batch_path
    probs_pool = [[] for _ in uids]
    model.eval()

    # Perform beam search for specified number of hops
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)
        actmask_pool = _batch_acts_to_masks(acts_pool)
        actmask_tensor = torch.BoolTensor(actmask_pool).to(device)

        # Get action probabilities from model
        with torch.no_grad():
            probs, _ = model((state_tensor, actmask_tensor))

        probs[~actmask_tensor] = -float("inf")
        probs = F.softmax(probs, dim=1)

        # Handle NaN values in probabilities
        if torch.isnan(probs).any():
            probs = _handle_nan_probabilities(probs, actmask_tensor)

        # Select top-k actions for current hop
        current_k = min(topk[hop], probs.shape[1])
        try:
            topk_probs, topk_idxs = torch.topk(probs, current_k, dim=1)
        except RuntimeError as e:
            print(f"Error during topk: {e}. Check k value ({current_k}) vs dimensions.")
            raise e

        topk_idxs = topk_idxs.cpu().numpy()
        topk_probs = topk_probs.cpu().numpy()

        # Expand paths based on top-k actions
        new_path_pool, new_probs_pool = _expand_paths(path_pool, probs_pool, acts_pool, actmask_pool, topk_idxs, topk_probs)

        path_pool = new_path_pool
        probs_pool = new_probs_pool

        if not path_pool:
            print("Warning: Beam search resulted in an empty path pool. Terminating early.")
            break

        # Prepare for next hop
        if hop < env.max_num_nodes - 2:
            try:
                state_pool = env._batch_get_state(path_pool)
            except IndexError as e:
                print(f"Error getting state at hop {hop + 1}: {e}. Path pool might be malformed.")
                print("Example path causing error:", path_pool[0] if path_pool else "None")
                break

    # Calculate final probabilities
    final_probs = [reduce(lambda x, y: x * y, p_list) if p_list else 0.0 for p_list in probs_pool]

    return path_pool, final_probs


def _handle_nan_probabilities(probs: torch.Tensor, actmask_tensor: torch.Tensor) -> torch.Tensor:
    """Handle NaN values in probability tensor."""
    print("Warning: NaNs detected in probabilities after softmax!")
    nan_rows = torch.isnan(probs).any(dim=1)

    for r_idx in torch.where(nan_rows)[0]:
        num_valid_actions = actmask_tensor[r_idx].sum().item()
        if num_valid_actions > 0:
            probs[r_idx, actmask_tensor[r_idx]] = 1.0 / num_valid_actions
            probs[r_idx, ~actmask_tensor[r_idx]] = 0.0
        else:
            probs[r_idx] = 0.0

    return probs


def _expand_paths(
    path_pool: List[List[Tuple]],
    probs_pool: List[List[float]],
    acts_pool: List[List[Tuple]],
    actmask_pool: np.ndarray,
    topk_idxs: np.ndarray,
    topk_probs: np.ndarray,
) -> Tuple[List[List[Tuple]], List[List[float]]]:
    """Expand paths based on top-k actions."""
    new_path_pool, new_probs_pool = [], []

    for row in range(topk_idxs.shape[0]):
        path = path_pool[row]

        for idx_in_topk, p in zip(topk_idxs[row], topk_probs[row]):
            if idx_in_topk >= len(acts_pool[row]):
                continue

            if not actmask_pool[row][idx_in_topk]:
                continue

            if p <= 0:
                continue

            relation, next_node_id = acts_pool[row][idx_in_topk]

            # Determine next node type
            if relation == SELF_LOOP:
                next_node_type = path[-1][1]
            else:
                current_node_type = path[-1][1]
                if current_node_type not in KG_RELATION or relation not in KG_RELATION[current_node_type]:
                    print(f"Warning: Invalid relation '{relation}' for node type '{current_node_type}'. Skipping path extension.")
                    continue
                next_node_type = KG_RELATION[current_node_type][relation]

            # Create new path
            new_path = path + [(relation, next_node_type, next_node_id)]
            new_path_pool.append(new_path)
            new_probs_pool.append(probs_pool[row] + [p])

    return new_path_pool, new_probs_pool


def predict_paths(policy_file: str, path_file: str, args: argparse.Namespace) -> Optional[ActorCritic]:
    """
    Generate prediction paths using trained model and beam search.

    Args:
        policy_file: Path to trained model file
        path_file: Path to save generated paths
        args: Configuration arguments

    Returns:
        Loaded model if successful, None otherwise
    """
    print("Predicting paths using beam search...")

    # Initialize environment
    env = BatchKGEnvironment(
        args.dataset,
        args.max_acts,
        max_path_len=args.max_path_len,
        state_history=args.state_history,
    )

    # Load trained model
    model = _load_trained_model(policy_file, env, args)
    if model is None:
        return None

    model.eval()

    # Get test users
    test_labels = load_labels(args.dataset, "test")
    test_uids = list(test_labels.keys())

    # Generate paths in batches
    all_paths, all_probs = _generate_paths_in_batches(env, model, test_uids, args.device, args.topk)

    if not all_paths:
        print("ERROR: No paths were generated during prediction. Check beam search logic and model loading.")
        return None

    # Save predictions
    _save_predictions(all_paths, all_probs, path_file)

    return model


def _load_trained_model(policy_file: str, env: BatchKGEnvironment, args: argparse.Namespace) -> Optional[ActorCritic]:
    """Load trained model from file."""
    try:
        pretrain_sd = torch.load(policy_file, map_location=args.device)
        print(f"Successfully loaded policy file: {policy_file}")
    except FileNotFoundError:
        print(f"ERROR: Policy file not found at {policy_file}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load policy file {policy_file}: {e}")
        return None

    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)

    try:
        model.load_state_dict(pretrain_sd)
        print("Successfully loaded state dict into the model.")
        return model
    except RuntimeError as e:
        print(f"ERROR: Failed to load state dict. Mismatched keys or layers?: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading the state dict: {e}")
        return None


def _generate_paths_in_batches(
    env: BatchKGEnvironment, model: ActorCritic, test_uids: List[int], device: torch.device, topk: List[int], batch_size: int = 16
) -> Tuple[List[List[Tuple]], List[float]]:
    """Generate paths for all test users in batches."""
    all_paths, all_probs = [], []
    start_idx = 0

    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]

        try:
            with torch.no_grad():
                paths, probs = batch_beam_search(env, model, batch_uids, device, topk=topk)
            all_paths.extend(paths)
            all_probs.extend(probs)
        except Exception as e:
            print(f"\nERROR during beam search for batch starting at index {start_idx}: {e}")
            pass

        start_idx = end_idx
        pbar.update(len(batch_uids))

    pbar.close()
    return all_paths, all_probs


def _save_predictions(all_paths: List[List[Tuple]], all_probs: List[float], path_file: str) -> None:
    """Save predictions to file."""
    predicts = {"paths": all_paths, "probs": all_probs}
    print(f"Saving {len(all_paths)} paths and probabilities to {path_file}")

    try:
        with open(path_file, "wb") as f:
            pickle.dump(predicts, f)
    except Exception as e:
        print(f"ERROR: Failed to save predictions to {path_file}: {e}")


def run_evaluation(
    path_file: str,
    train_labels: Dict[int, List[Tuple]],
    test_labels: Dict[int, List[Tuple]],
    TOP_K: int,
    data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[int, Dict]]:
    """
    Run evaluation using decoded recommendations and calculate metrics.

    Args:
        path_file: File containing predicted paths
        train_labels: Training labels
        test_labels: Test labels
        TOP_K: Number of top recommendations
        data: Full dataset
        train: Training data
        test: Test data
        args: Configuration arguments

    Returns:
        Tuple of (metrics, prediction_dataframe, human_readable_recommendations)
    """
    print("Starting evaluation using Decoder and metrics.py...")
    k = TOP_K

    # Load path predictions
    pred_paths_raw, pred_probs_raw = _load_path_predictions(path_file)
    if pred_paths_raw is None:
        return {}, pd.DataFrame(), {}

    # Load embeddings for scoring
    base_scores, median_score, scale_factor = _load_embeddings_and_scores(args.dataset)

    # Process paths to get candidates
    best_path_candidates = _process_paths_to_candidates(pred_paths_raw, pred_probs_raw, test_labels, train_labels, base_scores)

    # Generate ranked recommendations
    pred_labels, all_users_top_k_full_candidates = _generate_ranked_recommendations(best_path_candidates, k, base_scores, train_labels, args)

    # Calculate predictions using sigmoid
    top, top_k = _calculate_sigmoid_predictions(all_users_top_k_full_candidates, median_score, scale_factor, k)

    # Map IDs for specific datasets
    top, top_k = _map_ids_for_dataset(top, top_k, args.dataset)

    # Standardize data types
    train, test, top, top_k = _standardize_dataframe_types(train, test, top, top_k, args.dataset)

    # Decode recommendations
    decoder = RLRecommenderDecoder(args.dataset)
    human_recs, rating_pred_df = decoder.decode(args.dataset, pred_labels, k=k)

    # Prepare ground truth
    rating_true_df = _prepare_ground_truth(test_labels)

    # Filter predictions to remove training items
    top_filtered, top_k_filtered = _filter_training_items(top, top_k, train)

    # Calculate metrics
    metrics = _calculate_all_metrics(test, top_filtered, top_k_filtered, data, train, TOP_K)

    return metrics, rating_pred_df, human_recs


def _load_path_predictions(path_file: str) -> Tuple[Optional[List], Optional[List]]:
    """Load predicted paths from file."""
    print(f"Loading predicted paths from {path_file}")
    try:
        results = pickle.load(open(path_file, "rb"))
        return results["paths"], results["probs"]
    except FileNotFoundError:
        print(f"ERROR: Path file not found: {path_file}")
        return None, None
    except Exception as e:
        print(f"ERROR: Failed to load path file {path_file}: {e}")
        return None, None


def _load_embeddings_and_scores(dataset: str) -> Tuple[Optional[np.ndarray], float, float]:
    """Load embeddings and compute base scores."""
    try:
        embeds = load_embed(dataset)
        user_embeds = embeds[USERID]
        purchase_embeds = embeds[WATCHED][0]
        product_embeds = embeds[ITEMID]
        base_scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)

        median_score = np.median(base_scores)
        scale_factor = np.std(base_scores)

        return base_scores, median_score, scale_factor

    except Exception as e:
        print(f"Warning: Could not load embeddings or compute base scores: {e}. Relying solely on path probability.")
        return None, 0.0, 1.0


def _process_paths_to_candidates(
    pred_paths_raw: List[List[Tuple]],
    pred_probs_raw: List[float],
    test_labels: Dict[int, List[Tuple]],
    train_labels: Dict[int, List[Tuple]],
    base_scores: Optional[np.ndarray],
) -> Dict[int, List[Tuple]]:
    """Process paths and probabilities to get candidate items per user."""
    print("Processing paths to select candidate items...")

    pred_paths_by_user = {uid: {} for uid in test_labels}

    # Group paths by user and item
    for path, path_prob in zip(pred_paths_raw, pred_probs_raw):
        if not path or path[-1][1] != ITEMID:
            continue

        uid = path[0][2]
        if uid not in pred_paths_by_user:
            continue

        pid = path[-1][2]
        path_score = base_scores[uid][pid] if base_scores is not None else 0.0

        if pid not in pred_paths_by_user[uid]:
            pred_paths_by_user[uid][pid] = []
        pred_paths_by_user[uid][pid].append((path_score, path_prob, path))

    # Select best path per user-item pair and filter training items
    print("Selecting best path per user-item pair and filtering training items...")
    best_path_candidates = {}

    for uid, item_paths_dict in pred_paths_by_user.items():
        raw_train_list = train_labels.get(uid, [])
        train_pids = set(item_idx for item_idx, _ in raw_train_list)
        user_candidates = []

        for pid, path_tuples in item_paths_dict.items():
            if pid in train_pids:
                continue

            sorted_paths_for_item = sorted(path_tuples, key=lambda x: (x[1], x[0]), reverse=True)
            best_score, best_prob, _ = sorted_paths_for_item[0]
            user_candidates.append((best_score, best_prob, pid))

        best_path_candidates[uid] = user_candidates

    return best_path_candidates


def _generate_ranked_recommendations(
    best_path_candidates: Dict[int, List[Tuple]],
    k: int,
    base_scores: Optional[np.ndarray],
    train_labels: Dict[int, List[Tuple]],
    args: argparse.Namespace,
    sort_by: str = "score",
) -> Tuple[Dict[int, List[int]], Dict[int, List[Tuple]]]:
    """Generate final ranked recommendations."""
    print(f"Generating final top-{k} ranked list for {len(best_path_candidates)} users...")

    pred_labels = {}
    all_users_top_k_full_candidates = {}

    for uid, candidates in best_path_candidates.items():
        if sort_by == "score":
            sorted_candidates = sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)
        else:
            sorted_candidates = sorted(candidates, key=lambda x: (x[1], x[0]), reverse=True)

        top_k_full_info = sorted_candidates[:k]
        all_users_top_k_full_candidates[uid] = top_k_full_info

        top_k_pids = [pid for _, _, pid in sorted_candidates[:k]]

        # Add products if requested and available
        if args.add_products and len(top_k_pids) < k and base_scores is not None:
            top_k_pids = _add_additional_products(uid, top_k_pids, k, base_scores, train_labels)

        pred_labels[uid] = top_k_pids[::-1]  # Reverse for lowest score first

    return pred_labels, all_users_top_k_full_candidates


def _add_additional_products(uid: int, top_k_pids: List[int], k: int, base_scores: np.ndarray, train_labels: Dict[int, List[Tuple]]) -> List[int]:
    """Add additional products to reach k recommendations."""
    train_pids_tuples = train_labels.get(uid, [])
    train_pids = set(item_idx for item_idx, _ in train_pids_tuples)
    cand_pids_from_scores = np.argsort(base_scores[uid])
    num_needed = k - len(top_k_pids)
    filled_count = 0

    for cand_pid in cand_pids_from_scores[::-1]:
        if cand_pid not in train_pids and cand_pid not in top_k_pids:
            top_k_pids.append(cand_pid)
            filled_count += 1
            if filled_count >= num_needed:
                break

    return top_k_pids


def _calculate_sigmoid_predictions(
    all_users_top_k_full_candidates: Dict[int, List[Tuple]], median_score: float, scale_factor: float, k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate sigmoid-based predictions."""
    print(f"DEBUG: Calling calculate_predictons with all_users_top_k_full_candidates for {len(all_users_top_k_full_candidates)} users.")
    top, top_k = calculate_predictons(all_users_top_k_full_candidates, median_score, scale_factor, k)
    print(f"DEBUG: calculate_predictons returned 'top' with {len(top)} rows and 'top_k' with {len(top_k)} rows.")
    return top, top_k


def _map_ids_for_dataset(top: pd.DataFrame, top_k: pd.DataFrame, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map integer IDs to original IDs for specific datasets."""
    if dataset == AMAZONSALES:
        return _map_ids_amazonsales(top, top_k, dataset)
    elif dataset == POSTRECOMMENDATIONS:
        return _map_ids_postrecommendations(top, top_k, dataset)
    else:
        return top, top_k


def _map_ids_amazonsales(top: pd.DataFrame, top_k: pd.DataFrame, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map IDs for Amazon Sales dataset."""
    print("DEBUG: Mapping integer indices in 'top' and 'top_k' to original string IDs for AmazonSales.")

    try:
        processed_dataset_file_map = TMP_DIR[dataset] + "/processed_dataset.pkl"
        with open(processed_dataset_file_map, "rb") as f_map:
            processed_dataset_for_map = pickle.load(f_map)

        user_inv_map = processed_dataset_for_map.get("entity_maps", {}).get(USERID, {}).get("inv_map", {})
        item_inv_map = processed_dataset_for_map.get("entity_maps", {}).get(ITEMID, {}).get("inv_map", {})

        if not user_inv_map or not item_inv_map:
            print("ERROR: user_inv_map or item_inv_map is empty. Cannot map IDs for AmazonSales.")
            return top, top_k

        # Map top dataframe
        if not top.empty and "userID" in top.columns and "itemID" in top.columns:
            top["userID"] = top["userID"].astype(int).map(user_inv_map)
            top["itemID"] = top["itemID"].astype(int).map(item_inv_map)
            top.dropna(subset=["userID", "itemID"], inplace=True)

        # Map top_k dataframe
        if not top_k.empty and "userID" in top_k.columns and "itemID" in top_k.columns:
            top_k["userID"] = top_k["userID"].astype(int).map(user_inv_map)
            top_k["itemID"] = top_k["itemID"].astype(int).map(item_inv_map)
            top_k.dropna(subset=["userID", "itemID"], inplace=True)

        print(f"DEBUG: After mapping, 'top' has {len(top)} rows, 'top_k' has {len(top_k)} rows.")

    except Exception as e_map:
        print(f"ERROR during ID mapping for AmazonSales: {e_map}")

    return top, top_k


def _map_ids_postrecommendations(top: pd.DataFrame, top_k: pd.DataFrame, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map IDs for Post Recommendations dataset."""
    print("DEBUG: Mapping integer indices in 'top' and 'top_k' to original string IDs for Postrecommendation.")

    try:
        processed_dataset_file_map = TMP_DIR[dataset] + "/processed_dataset.pkl"
        with open(processed_dataset_file_map, "rb") as f_map:
            processed_dataset_for_map = pickle.load(f_map)

        user_inv_map = processed_dataset_for_map.get("entity_maps", {}).get(USERID, {}).get("inv_map", {})
        item_inv_map = processed_dataset_for_map.get("entity_maps", {}).get(ITEMID, {}).get("inv_map", {})

        if not user_inv_map or not item_inv_map:
            print("ERROR: user_inv_map or item_inv_map is empty. Cannot map IDs for Postrecommendation.")
            return top, top_k

        # Map top dataframe
        if not top.empty and "userID" in top.columns:
            top["userID"] = top["userID"].astype(int).map(user_inv_map)
            if "itemID" in top.columns and item_inv_map:
                top["itemID"] = top["itemID"].astype(int).map(item_inv_map)
            elif "itemID" in top.columns:
                top["itemID"] = top["itemID"].astype(int)
            top.dropna(subset=["userID"], inplace=True)

        # Map top_k dataframe
        if not top_k.empty and "userID" in top_k.columns:
            top_k["userID"] = top_k["userID"].astype(int).map(user_inv_map)
            if "itemID" in top_k.columns and item_inv_map:
                top_k["itemID"] = top_k["itemID"].astype(int).map(item_inv_map)
            elif "itemID" in top_k.columns:
                top_k["itemID"] = top_k["itemID"].astype(int)
            top_k.dropna(subset=["userID"], inplace=True)

        print(f"DEBUG: After mapping, 'top' has {len(top)} rows, 'top_k' has {len(top_k)} rows.")

    except Exception as e_map:
        print(f"ERROR during ID mapping for PostRecommendations: {e_map}")

    return top, top_k


def _standardize_dataframe_types(
    train: pd.DataFrame, test: pd.DataFrame, top: pd.DataFrame, top_k: pd.DataFrame, dataset: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize data types for all DataFrames."""
    print(f"\n--- Ensuring Consistent DataFrame Dtypes for {dataset} ---")

    # Define target data types by dataset
    if dataset == MOVIELENS:
        id_cols_dtypes = {"userID": int, "itemID": int}
    elif dataset == AMAZONSALES:
        id_cols_dtypes = {"userID": str, "itemID": str}
    elif dataset == POSTRECOMMENDATIONS:
        id_cols_dtypes = {"userID": str, "itemID": int}
    else:
        id_cols_dtypes = {"userID": int, "itemID": int}

    numeric_columns = {"rating": float, "prediction": float}

    # Apply standardization to all DataFrames
    dfs_to_standardize = [train, test, top, top_k]
    for df_item in dfs_to_standardize:
        if df_item is None or df_item.empty:
            continue

        # Standardize ID columns
        for col_name, target_type in id_cols_dtypes.items():
            if col_name in df_item.columns:
                _convert_column_type(df_item, col_name, target_type, dataset)

        # Standardize numeric columns
        for col_name, target_type in numeric_columns.items():
            if col_name in df_item.columns:
                _convert_column_type(df_item, col_name, target_type, dataset)

    return train, test, top, top_k


def _convert_column_type(df: pd.DataFrame, col_name: str, target_type: type, dataset: str) -> None:
    """Convert column to target type with error handling."""
    current_dtype = df[col_name].dtype

    if target_type is str:
        if not (current_dtype == "object" or isinstance(current_dtype, pd.StringDtype) or current_dtype == "string"):
            print(f"Converting column '{col_name}' to string. Original dtype: {current_dtype}")
            df[col_name] = df[col_name].astype(str)
    elif target_type is int:
        if not pd.api.types.is_integer_dtype(current_dtype):
            print(f"Converting column '{col_name}' to integer. Original dtype: {current_dtype}")
            try:
                df[col_name] = pd.to_numeric(df[col_name], errors="raise").astype(int)
            except ValueError as e:
                print(f"ERROR: Could not convert column '{col_name}' to int for {dataset}: {e}.")
    elif target_type is float:
        if not pd.api.types.is_numeric_dtype(df[col_name]) or df[col_name].dtype != target_type:
            print(f"Converting column '{col_name}' to {target_type}. Original dtype: {df[col_name].dtype}")
            try:
                df[col_name] = df[col_name].astype(target_type)
            except ValueError as e:
                print(f"Warning: Could not convert column '{col_name}' to {target_type}: {e}. Trying pd.to_numeric with coercion.")
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")


def _prepare_ground_truth(test_labels: Dict[int, List[Tuple]]) -> pd.DataFrame:
    """Prepare ground truth DataFrame for metrics calculation."""
    print("Preparing ground truth DataFrame...")

    true_data = []
    for user_idx, item_rating_tuples in test_labels.items():
        for item_idx, rating_val in item_rating_tuples:
            true_data.append({USERID: user_idx, ITEMID: item_idx, RATING: rating_val})

    rating_true_df = pd.DataFrame(true_data)
    if not rating_true_df.empty:
        rating_true_df[USERID] = rating_true_df[USERID].astype(int)
        rating_true_df[ITEMID] = rating_true_df[ITEMID].astype(int)
        rating_true_df[RATING] = rating_true_df[RATING].astype(float)

    return rating_true_df


def _filter_training_items(top: pd.DataFrame, top_k: pd.DataFrame, train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter predictions to remove items already in training set."""
    print("Filtering predictions to remove items already in the `train` DataFrame...")

    if train.empty or (top.empty and top_k.empty):
        print("Train DataFrame is empty or prediction DataFrames are empty, skipping filtering.")
        return top.copy(), top_k.copy()

    train_interactions = train.set_index(["userID", "itemID"]).index

    # Filter top dataframe
    if not top.empty:
        top_interactions = pd.MultiIndex.from_frame(top[["userID", "itemID"]])
        mask_top = ~top_interactions.isin(train_interactions)
        top_filtered = top[mask_top].copy()
        print(f"Original 'top' size: {len(top)}, Filtered 'top' size: {len(top_filtered)}")
    else:
        top_filtered = pd.DataFrame(columns=top.columns if not top.empty else [])

    # Filter top_k dataframe
    if not top_k.empty:
        top_k_interactions = pd.MultiIndex.from_frame(top_k[["userID", "itemID"]])
        mask_top_k = ~top_k_interactions.isin(train_interactions)
        top_k_filtered = top_k[mask_top_k].copy()
        print(f"Original 'top_k' size: {len(top_k)}, Filtered 'top_k_filtered' size: {len(top_k_filtered)}")
    else:
        top_k_filtered = pd.DataFrame(columns=top_k.columns if not top_k.empty else [])

    return top_filtered, top_k_filtered


def _calculate_all_metrics(
    test: pd.DataFrame, top_filtered: pd.DataFrame, top_k_filtered: pd.DataFrame, data: pd.DataFrame, train: pd.DataFrame, TOP_K: int
) -> Dict[str, Optional[float]]:
    """Calculate all evaluation metrics."""
    print(f"\n--- Calculating Metrics @{TOP_K} using metrics.py ---")

    if top_k_filtered.empty or test.empty:
        print("Cannot calculate metrics: Prediction or Ground Truth DataFrame is empty.")
        return {}

    # Calculate individual metrics with error handling
    metrics = {}

    metric_functions = [
        (
            "precision",
            lambda: precision_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1),
        ),
        (
            "precision_at_k",
            lambda: precision_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K),
        ),
        ("recall", lambda: recall_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)),
        (
            "recall_at_k",
            lambda: recall_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K),
        ),
        (
            "ndcg_at_k",
            lambda: ndcg_at_k(
                test, top_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", relevancy_method="top_k", k=1
            ),
        ),
        ("mae", lambda: mae(test, top_k_filtered)),
        ("rmse", lambda: rmse(test, top_k_filtered)),
        ("mrr", lambda: mrr(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")),
        ("user_coverage", lambda: user_coverage(test, top_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")),
        ("item_coverage", lambda: item_coverage(test, top_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")),
        ("intra_list_similarity", lambda: intra_list_similarity_score(data, top_k_filtered, feature_cols=["genres"])),
        ("intra_list_dissimilarity", lambda: intra_list_dissimilarity(data, top_k_filtered, feature_cols=["genres"])),
        ("personalization", lambda: personalization_score(train, top_filtered)),
    ]

    for metric_name, metric_func in metric_functions:
        try:
            metrics[metric_name] = metric_func()
        except Exception as e:
            metrics[metric_name] = None
            print(f"Error calculating {metric_name}: {e}")

    # Print results
    _print_metric_results(metrics)

    return metrics


def _print_metric_results(metrics: Dict[str, Optional[float]]) -> None:
    """Print formatted metric results."""

    def format_metric(metric):
        return f"{metric:.4f}" if isinstance(metric, (float, int)) else "N/A"

    print(
        "Precision:\t" + format_metric(metrics.get("precision")),
        "Precision@K:\t" + format_metric(metrics.get("precision_at_k")),
        "Recall:\t" + format_metric(metrics.get("recall")),
        "Recall@K:\t" + format_metric(metrics.get("recall_at_k")),
        "MAE:\t" + format_metric(metrics.get("mae")),
        "RMSE:\t" + format_metric(metrics.get("rmse")),
        "NDCG:\t" + format_metric(metrics.get("ndcg_at_k")),
        "MRR:\t" + format_metric(metrics.get("mrr")),
        "User coverage:\t" + format_metric(metrics.get("user_coverage")),
        "Item coverage:\t" + format_metric(metrics.get("item_coverage")),
        "Personalization:\t" + format_metric(metrics.get("personalization")),
        "Intra-list similarity:\t" + format_metric(metrics.get("intra_list_similarity")),
        "Intra-list dissimilarity:\t" + format_metric(metrics.get("intra_list_dissimilarity")),
        sep="\n",
    )


def test(
    TOP_K: int,
    want_col: List[str],
    num_rows: Optional[int],
    ratio: float,
    data_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    privacy: bool,
    fraction_to_hide: float,
    personalization: bool,
    fraction_to_change: float,
    args: argparse.Namespace,
) -> None:
    """
    Main testing function that coordinates path prediction and evaluation.

    Args:
        TOP_K: Number of top recommendations
        want_col: Required columns
        num_rows: Number of rows used
        ratio: Train/test split ratio
        data_df: Full dataset
        train_df: Training data
        test_df: Test data
        privacy: Privacy flag
        fraction_to_hide: Fraction hidden for privacy
        personalization: Personalization flag
        fraction_to_change: Fraction changed for personalization
        args: Configuration arguments
    """
    policy_file = args.log_dir + "/policy_model_epoch_{}.ckpt".format(args.epochs)
    path_file = args.log_dir + "/policy_paths_epoch{}.pkl".format(args.epochs)

    if not os.path.exists(policy_file):
        print(f"ERROR: Policy file not found: {policy_file}")
        print("Please ensure the agent has been trained for the specified number of epochs.")
        return

    # Generate paths if needed
    if args.run_path or not os.path.exists(path_file):
        print(f"Running path prediction (run_path={args.run_path}, file_exists={os.path.exists(path_file)})...")
        model = predict_paths(policy_file, path_file, args)
    else:
        print(f"Skipping path prediction, using existing file: {path_file}")
        model = None

    # Run evaluation if requested
    if args.run_eval:
        if not os.path.exists(path_file):
            print(f"ERROR: Cannot run evaluation because path file does not exist: {path_file}")
            return

        print("Loading data labels for evaluation...")
        try:
            train_labels = load_labels(args.dataset, "train")
            test_labels = load_labels(args.dataset, "test")
        except FileNotFoundError as e:
            print(f"ERROR: Could not load label file: {e}. Make sure preprocessing was completed.")
            return
        except Exception as e:
            print(f"ERROR loading labels: {e}")
            return

        metrics, rating_pred_df, human_recs = run_evaluation(path_file, train_labels, test_labels, TOP_K, data_df, train_df, test_df, args)

        # Prepare hyperparameters for logging
        rl_hyperparams = {
            "dataset": args.dataset,
            "want_col": want_col,
            "num_rows": num_rows,
            "ratio": ratio,
            "seed": args.seed,
            "epochs_loaded": args.epochs,
            "max_acts": args.max_acts,
            "max_path_len": args.max_path_len,
            "gamma": args.gamma,
            "state_history": args.state_history,
            "hidden_sizes": args.hidden,
        }

        # Log results to MLflow
        log_mlflow.log_mlflow(
            args.dataset,
            rating_pred_df,
            metrics,
            num_rows,
            args.seed,
            model,
            "RL",
            rl_hyperparams,
            data_df,
            train_df,
            privacy=privacy,
            fraction_to_hide=fraction_to_hide,
            personalization=personalization,
            fraction_to_change=fraction_to_change,
        )

    else:
        print("Skipping evaluation.")


def test_agent_rl(
    dataset: str,
    TOP_K: int,
    want_col: List[str],
    num_rows: Optional[int],
    ratio: float,
    seed: int,
    data_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    privacy: bool,
    fraction_to_hide: float,
    personalization: bool,
    fraction_to_change: float,
) -> None:
    """
    Main entry point for RL agent testing.

    Args:
        dataset: Dataset name
        TOP_K: Number of top recommendations
        want_col: Required columns
        num_rows: Number of rows to use
        ratio: Train/test split ratio
        seed: Random seed
        data_df: Full dataset
        train_df: Training data
        test_df: Test data
        privacy: Privacy modification flag
        fraction_to_hide: Fraction of data to hide
        personalization: Personalization modification flag
        fraction_to_change: Fraction of data to change
    """
    # Parse command line arguments
    boolean = lambda x: (str(x).lower() == "true")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=dataset, help="Dataset name (set automatically).")
    parser.add_argument("--name", type=str, default="train_agent", help="directory name matching training.")
    parser.add_argument("--seed", type=int, default=seed, help="random seed.")
    parser.add_argument("--gpu", type=str, default="0", help="gpu device.")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch number of the model to load.")
    parser.add_argument("--max_acts", type=int, default=250, help="Max number of actions (must match training).")
    parser.add_argument("--max_path_len", type=int, default=3, help="Max path length (must match training).")
    parser.add_argument("--gamma", type=float, default=0.99, help="reward discount factor (used in model definition).")
    parser.add_argument("--state_history", type=int, default=1, help="state history length (must match training).")
    parser.add_argument("--hidden", type=int, nargs="*", default=[512, 256], help="hidden layer sizes (must match training).")
    parser.add_argument("--add_products", type=boolean, default=False, help="Add predicted products up to 10")
    parser.add_argument("--topk", type=int, nargs="*", default=[25, 5, 1], help="Beam search topk width per hop.")
    parser.add_argument("--run_path", type=boolean, default=True, help="Generate predicted path? (takes long time)")
    parser.add_argument("--run_eval", type=boolean, default=True, help="Run evaluation?")
    args = parser.parse_args()

    # Set up device and directories
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    args.log_dir = TMP_DIR[args.dataset] + "/" + args.name
    if not os.path.isdir(args.log_dir):
        print(f"Warning: Log directory {args.log_dir} not found. Ensure '--name' matches training.")

    # Set random seed and run test
    set_random_seed(args.seed)
    test(
        TOP_K,
        want_col,
        num_rows,
        ratio,
        data_df,
        train_df,
        test_df,
        privacy,
        fraction_to_hide,
        personalization,
        fraction_to_change,
        args,
    )
