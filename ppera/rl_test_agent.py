from __future__ import absolute_import, division, print_function

import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from functools import reduce

from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, mae, rmse, novelty, historical_item_novelty, user_item_serendipity, user_serendipity, serendipity, catalog_coverage, distributional_coverage
from metrics import precision_at_k, recall_at_k, f1, mrr, accuracy, user_coverage, item_coverage, intra_list_similarity_score, intra_list_dissimilarity, personalization_score


import metrics
import log_mlflow

from rl_prediction import calculate_predictons
from rl_kg_env import BatchKGEnvironment
from rl_decoder import RLRecommenderDecoder
from rl_train_agent import ActorCritic
from rl_utils import *

def batch_beam_search(env, model, uids, device, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=bool) 
            act_mask[:num_acts] = True
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)
    path_pool = env._batch_path
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)
        actmask_pool = _batch_acts_to_masks(acts_pool)
        actmask_tensor = torch.BoolTensor(actmask_pool).to(device)
        with torch.no_grad():
            probs, _ = model((state_tensor, actmask_tensor))

        probs[~actmask_tensor] = -float('inf')
        probs = F.softmax(probs, dim=1)


        if torch.isnan(probs).any():
            print("Warning: NaNs detected in probabilities after softmax!")
            nan_rows = torch.isnan(probs).any(dim=1)
            for r_idx in torch.where(nan_rows)[0]:
                num_valid_actions = actmask_tensor[r_idx].sum().item()
                if num_valid_actions > 0:
                    probs[r_idx, actmask_tensor[r_idx]] = 1.0 / num_valid_actions
                    probs[r_idx, ~actmask_tensor[r_idx]] = 0.0
                else:
                    probs[r_idx] = 0.0

        current_k = min(topk[hop], probs.shape[1])
        try:
             topk_probs, topk_idxs = torch.topk(probs, current_k, dim=1)
        except RuntimeError as e:
            print(f"Error during topk: {e}. Check k value ({current_k}) vs dimensions.")
            raise e


        topk_idxs = topk_idxs.cpu().numpy()
        topk_probs = topk_probs.cpu().numpy()

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

                 if relation == SELF_LOOP:
                     next_node_type = path[-1][1]
                 else:
                    current_node_type = path[-1][1]
                    if current_node_type not in KG_RELATION or relation not in KG_RELATION[current_node_type]:
                         print(f"Warning: Invalid relation '{relation}' for node type '{current_node_type}'. Skipping path extension.")
                         continue
                    next_node_type = KG_RELATION[current_node_type][relation]

                 new_path = path + [(relation, next_node_type, next_node_id)]
                 new_path_pool.append(new_path)
                 new_probs_pool.append(probs_pool[row] + [p])

        path_pool = new_path_pool
        probs_pool = new_probs_pool

        if not path_pool:
             print("Warning: Beam search resulted in an empty path pool. Terminating early.")
             break

        if hop < env.max_num_nodes - 2:
            try:
                state_pool = env._batch_get_state(path_pool)
            except IndexError as e:
                 print(f"Error getting state at hop {hop+1}: {e}. Path pool might be malformed.")
                 print("Example path causing error:", path_pool[0] if path_pool else "None")
                 break


    final_probs = [reduce(lambda x, y: x * y, p_list) if p_list else 0.0 for p_list in probs_pool]
    return path_pool, final_probs


def predict_paths(policy_file, path_file, args):
    print('Predicting paths using beam search...')
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    try:
        pretrain_sd = torch.load(policy_file, map_location=args.device)
        print(f"Successfully loaded policy file: {policy_file}")
    except FileNotFoundError:
        print(f"ERROR: Policy file not found at {policy_file}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load policy file {policy_file}: {e}")
        return

    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    try:
        model.load_state_dict(pretrain_sd)
        print("Successfully loaded state dict into the model.")
    except RuntimeError as e:
        print(f"ERROR: Failed to load state dict. Mismatched keys or layers?: {e}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading the state dict: {e}")
        return

    model.eval()

    test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        try:
             with torch.no_grad():
                 paths, probs = batch_beam_search(env, model, batch_uids, args.device, topk=args.topk)
             all_paths.extend(paths)
             all_probs.extend(probs)
        except Exception as e:
             print(f"\nERROR during beam search for batch starting at index {start_idx}: {e}")
             pass

        start_idx = end_idx
        pbar.update(len(batch_uids))
    pbar.close()

    if not all_paths:
         print("ERROR: No paths were generated during prediction. Check beam search logic and model loading.")
         return

    predicts = {'paths': all_paths, 'probs': all_probs}
    print(f"Saving {len(all_paths)} paths and probabilities to {path_file}")
    try:
        with open(path_file, 'wb') as f:
            pickle.dump(predicts, f)
    except Exception as e:
         print(f"ERROR: Failed to save predictions to {path_file}: {e}")

    return model


def run_evaluation(path_file, train_labels, test_labels, TOP_K, data, train, test, args):

    print("Starting evaluation using Decoder and metrics.py...")
    k = TOP_K

    # 1. Load path predictions
    print(f"Loading predicted paths from {path_file}")
    try:
        results = pickle.load(open(path_file, 'rb'))
        pred_paths_raw = results['paths']
        pred_probs_raw = results['probs']
    except FileNotFoundError:
        print(f"ERROR: Path file not found: {path_file}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load path file {path_file}: {e}")
        return

    # 2. Load Embeddings (for path scoring if needed, although probs are now primary)
    try:
        embeds = load_embed(args.dataset)
        user_embeds = embeds[USERID]
        purchase_embeds = embeds[WATCHED][0]
        product_embeds = embeds[ITEMID]
        base_scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)

        median_score = np.median(base_scores)
        scale_factor = np.std(base_scores)
        # print(f"Base scores loaded. Median: {median_score}, Scale Factor: {scale_factor}")


    except Exception as e:
         print(f"Warning: Could not load embeddings or compute base scores: {e}. Relying solely on path probability.")
         base_scores = None


    # 3. Process paths and probabilities to get candidate items per user
    print("Processing paths to select candidate items...")
    pred_paths_by_user = {uid: {} for uid in test_labels}

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

    # 5. Generate final ranked list (pred_labels format: lowest score first)
    print(f"Generating final top-{k} ranked list for {len(best_path_candidates)} users...")
    pred_labels = {}
    sort_by = 'score'

    all_users_top_k_full_candidates  = {}

    for uid, candidates in best_path_candidates.items():
        if sort_by == 'score':
            sorted_candidates = sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)
        else:
            sorted_candidates = sorted(candidates, key=lambda x: (x[1], x[0]), reverse=True)
        
        # print(f"User {uid}: Sorted Candidates: {sorted_candidates}") #TODO: remove
        
        top_k_full_info = sorted_candidates[:k]
        all_users_top_k_full_candidates[uid] = top_k_full_info


        top_k_pids = [pid for _, _, pid in sorted_candidates[:k]] 
        # print(f"User {uid}: Top K PIDs: {top_k_pids}") #TODO: remove


        if args.add_products and len(top_k_pids) < k and base_scores is not None:
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

        pred_labels[uid] = top_k_pids[::-1]


    print('-------------------------------------------- Sorted Candidates --------------------------------------------')
    
    print(f"DEBUG: Calling calculate_predictons with all_users_top_k_full_candidates for {len(all_users_top_k_full_candidates)} users.")
    top, top_k = calculate_predictons(all_users_top_k_full_candidates, median_score, scale_factor, k)
    print(f"DEBUG: calculate_predictons returned 'top' with {len(top)} rows and 'top_k' with {len(top_k)} rows.")

    # Initialize user_inv_map and item_inv_map to be available for mapping if needed
    user_inv_map = {}
    item_inv_map = {}
    user_map = {} # Will also load user_map (original_id -> int_index)

    # --- STEP 1: Map integer IDs in top/top_k to ORIGINAL STRING IDs if AMAZONSALES ---
    if args.dataset == AMAZONSALES:
        print(f"DEBUG: Mapping integer indices in 'top' and 'top_k' to original string IDs for AmazonSales.")
        try:
            print("DEBUG: Loading inv_maps and map for ID conversion of top/top_k.")
            processed_dataset_file_map = TMP_DIR[args.dataset] + '/processed_dataset.pkl'
            with open(processed_dataset_file_map, 'rb') as f_map:
                processed_dataset_for_map = pickle.load(f_map) # Renamed to avoid conflict if 'processed_dataset' is used later
            user_inv_map = processed_dataset_for_map.get('entity_maps', {}).get(USERID, {}).get('inv_map', {})
            item_inv_map = processed_dataset_for_map.get('entity_maps', {}).get(ITEMID, {}).get('inv_map', {})
            user_map = processed_dataset_for_map.get('entity_maps', {}).get(USERID, {}).get('map', {})


            if not user_inv_map or not item_inv_map:
                print("ERROR: user_inv_map or item_inv_map is empty. Cannot map IDs for AmazonSales.")
            else:
                if not top.empty and 'userID' in top.columns and 'itemID' in top.columns:
                    top['userID'] = top['userID'].astype(int).map(user_inv_map)
                    top['itemID'] = top['itemID'].astype(int).map(item_inv_map)
                    top.dropna(subset=['userID', 'itemID'], inplace=True)
                
                if not top_k.empty and 'userID' in top_k.columns and 'itemID' in top_k.columns:
                    top_k['userID'] = top_k['userID'].astype(int).map(user_inv_map)
                    top_k['itemID'] = top_k['itemID'].astype(int).map(item_inv_map)
                    top_k.dropna(subset=['userID', 'itemID'], inplace=True)
                
                print(f"DEBUG: After mapping, 'top' has {len(top)} rows, 'top_k' has {len(top_k)} rows.")
                if not top_k.empty: print(f"DEBUG: Sample of 'top_k' after mapping IDs:\n{top_k.head().to_string()}")

        except FileNotFoundError:
            print(f"ERROR: Could not load {TMP_DIR[args.dataset]}/processed_dataset.pkl for ID mapping.")
        except Exception as e_map:
            print(f"ERROR during ID mapping for AmazonSales: {e_map}")
            import traceback
            traceback.print_exc()

    elif args.dataset == POSTRECOMMENDATIONS:
        print(f"DEBUG: Mapping integer indices in 'top' and 'top_k' to original string IDs for Postrecommendation.")
        try:
            print("DEBUG: Loading inv_maps and map for ID conversion of top/top_k.")
            processed_dataset_file_map = TMP_DIR[args.dataset] + '/processed_dataset.pkl'
            with open(processed_dataset_file_map, 'rb') as f_map:
                processed_dataset_for_map = pickle.load(f_map) # Renamed to avoid conflict if 'processed_dataset' is used later
            user_inv_map = processed_dataset_for_map.get('entity_maps', {}).get(USERID, {}).get('inv_map', {})
            item_inv_map = processed_dataset_for_map.get('entity_maps', {}).get(ITEMID, {}).get('inv_map', {})
            user_map = processed_dataset_for_map.get('entity_maps', {}).get(USERID, {}).get('map', {})


            if not user_inv_map or not item_inv_map:
                print("ERROR: user_inv_map or item_inv_map is empty. Cannot map IDs for Postrecommendation.")
            else:
                if not top.empty and 'userID' in top.columns:
                    # UserID: int_index -> original_string_id
                    top['userID'] = top['userID'].astype(int).map(user_inv_map)
                    # ItemID: int_index -> original_int_id (item_inv_map keys are int, values are int)
                    # So, if calculate_predictons outputs int_indices for items, and original itemIDs are also ints,
                    # and item_inv_map correctly maps int_index -> original_int_id,
                    # then the itemID column from calculate_predictons might already BE the original int_id if your
                    # internal item indices ARE the original item IDs.
                    # OR, if internal item indices are 0-N and original are different ints, then map is needed.
                    if 'itemID' in top.columns and item_inv_map: # Check if item_inv_map loaded
                         top['itemID'] = top['itemID'].astype(int).map(item_inv_map)
                    elif 'itemID' in top.columns: # item_inv_map not loaded/empty, assume itemID from calc_pred is original int
                         top['itemID'] = top['itemID'].astype(int) # Just ensure it's int
                    
                    top.dropna(subset=['userID', 'itemID' if 'itemID' in top.columns and item_inv_map else 'userID'], inplace=True) # Adjust dropna
                
                if not top_k.empty and 'userID' in top_k.columns:
                    top_k['userID'] = top_k['userID'].astype(int).map(user_inv_map)
                    if 'itemID' in top_k.columns and item_inv_map:
                        top_k['itemID'] = top_k['itemID'].astype(int).map(item_inv_map) # Map int_index to original_int_id
                    elif 'itemID' in top_k.columns:
                        top_k['itemID'] = top_k['itemID'].astype(int)

                    top_k.dropna(subset=['userID', 'itemID' if 'itemID' in top_k.columns and item_inv_map else 'userID'], inplace=True)
                
                print(f"DEBUG: After mapping, 'top' has {len(top)} rows, 'top_k' has {len(top_k)} rows.")
                if not top_k.empty: print(f"DEBUG: Sample of 'top_k' after mapping IDs:\n{top_k.head().to_string()}")

        except FileNotFoundError:
            print(f"ERROR: Could not load {TMP_DIR[args.dataset]}/processed_dataset.pkl for ID mapping.")
        except Exception as e_map:
            print(f"ERROR during ID mapping for AmazonSales: {e_map}")
            import traceback
            traceback.print_exc()
    
    # --- STEP 2: General DType Standardization for all relevant DataFrames ---
    print(f"\n--- Ensuring Consistent DataFrame Dtypes for {args.dataset} ---")
    # ... (Dtypes BEFORE standardization print loop - this can stay) ...

    dfs_to_standardize_list = []
    if train is not None and not train.empty: dfs_to_standardize_list.append(train)
    if test is not None and not test.empty: dfs_to_standardize_list.append(test)
    if 'top' in locals() and top is not None and not top.empty: dfs_to_standardize_list.append(top)
    if 'top_k' in locals() and top_k is not None and not top_k.empty: dfs_to_standardize_list.append(top_k)

    id_columns = ['userID', 'itemID']
    numeric_columns = {'rating': float, 'prediction': float}

    id_cols_dtypes = {} # store target dtypes {col_name: target_type}
    if args.dataset == MOVIELENS:
        print(f"Target DTypes for {args.dataset}: userID=int, itemID=int")
        id_cols_dtypes = {'userID': int, 'itemID': int}
    elif args.dataset == AMAZONSALES:
        print(f"Target DTypes for {args.dataset}: userID=str, itemID=str")
        id_cols_dtypes = {'userID': str, 'itemID': str}
    elif args.dataset == POSTRECOMMENDATIONS:
        print(f"Target DTypes for {args.dataset}: userID=str, itemID=int")
        id_cols_dtypes = {'userID': str, 'itemID': int}
    
    numeric_columns = {'rating': float, 'prediction': float} # These are generally consistent

    for df_item in dfs_to_standardize_list:
        if df_item is None or df_item.empty: continue
        
        for col_name, target_type in id_cols_dtypes.items():
            if col_name in df_item.columns:
                current_dtype = df_item[col_name].dtype
                if target_type == str:
                    if not (current_dtype == 'object' or isinstance(current_dtype, pd.StringDtype) or current_dtype == 'string'):
                        print(f"Converting column '{col_name}' in DataFrame to string. Original dtype: {current_dtype}")
                        df_item[col_name] = df_item[col_name].astype(str)
                elif target_type == int:
                    if not pd.api.types.is_integer_dtype(current_dtype):
                        print(f"Converting column '{col_name}' in DataFrame to integer. Original dtype: {current_dtype}")
                        try:
                            # Important: If converting from object/string to int, ensure it's purely numeric first
                            df_item[col_name] = pd.to_numeric(df_item[col_name], errors='raise').astype(int)
                        except ValueError as e:
                            print(f"ERROR: Could not convert column '{col_name}' to int for {args.dataset}: {e}.")
                            print(f"Sample offending values in {col_name}: {df_item[pd.to_numeric(df_item[col_name], errors='coerce').isna() & df_item[col_name].notna()][col_name].unique()[:5]}")
        
        for col_name, target_type_val in numeric_columns.items():
            if col_name in df_item.columns:
                if not pd.api.types.is_numeric_dtype(df_item[col_name]) or df_item[col_name].dtype != target_type_val:
                    print(f"Converting column '{col_name}' in DataFrame to {target_type_val}. Original dtype: {df_item[col_name].dtype}")
                    try:
                        df_item[col_name] = df_item[col_name].astype(target_type_val)
                    except ValueError as e:
                        print(f"Warning: Could not convert column '{col_name}' to {target_type_val}: {e}. Trying pd.to_numeric with coercion.")
                        df_item[col_name] = pd.to_numeric(df_item[col_name], errors='coerce')


    print("--- Dtypes AFTER standardization ---")


    # --- Debug: Print Dtypes After Standardization ---
    print("--- Dtypes AFTER standardization ---")
    dfs_to_check_after = {'train': train, 'test': test}
    if 'top' in locals() and top is not None and not top.empty:
        dfs_to_check_after['top'] = top
    if 'top_k' in locals() and top_k is not None and not top_k.empty:
        dfs_to_check_after['top_k'] = top_k
        
    for name, df_val in dfs_to_check_after.items():
        if df_val is not None and not df_val.empty:
            print(f"Dtypes for '{name}':\n{df_val.dtypes.to_string()}")
        else:
            print(f"DataFrame '{name}' is empty or None after standardization.")
    print("------------------------------------")
    


    # 6. Instantiate Decoder
    print("Instantiating decoder...")
    decoder = RLRecommenderDecoder(args.dataset)

    # 7. Decode Recommendations
    print("Decoding recommendations...")
    human_recs, rating_pred_df = decoder.decode(args.dataset, pred_labels, k=k)

    print("\n--- Sample Human-Readable Recommendations ---")
    # zmienic count bo i tak zwraca top_k
    # count = 0
    # for user_idx, user_rec_data in human_recs.items():
    #     original_uid = user_rec_data['original_userID']
    #     recs_list = user_rec_data['recommendations']
    #     print(f"User Index: {user_idx} (Original UserID: {original_uid})")
    #     if not recs_list:
    #         print("  No recommendations.")
    #     for r in recs_list:
    #         print(f"  Rank {r['rank']}: ID={r['original_id']} (Idx={r['item_idx']}), Title='{r['title']}', Genres={r['genres']}")
    #     count += 1
    #     if count >= 5:
    #         break
    # print("--------------------------------------------")


    # 8. Prepare Ground Truth DataFrame for metrics.py
    print("Preparing ground truth DataFrame...")
    true_data = []
    for user_idx, item_rating_tuples in test_labels.items():
        for item_idx, rating_val in item_rating_tuples: # Unpack the tuple
            true_data.append({
                USERID: user_idx,
                ITEMID: item_idx,
                RATING: rating_val
            })
    rating_true_df = pd.DataFrame(true_data)
    if not rating_true_df.empty:
        rating_true_df[USERID] = rating_true_df[USERID].astype(int)
        rating_true_df[ITEMID] = rating_true_df[ITEMID].astype(int)
        rating_true_df[RATING] = rating_true_df[RATING].astype(float)

    problematic_user_indices_for_pred_labels = set()
    if not test.empty and 'userID' in test.columns:
        # We need user_map (original_id -> int_index) if test['userID'] has original IDs
        # If test['userID'] already has int_indices (e.g. for Movielens after standardization to int)
        # then user_map isn't strictly needed here, but using it makes it robust.
        if not user_map and args.dataset in [AMAZONSALES, POSTRECOMMENDATIONS] : # user_map should have been loaded for AMAZONSALES
             print("WARNING: user_map not loaded for pred_labels check with AmazonSales, skipping this specific debug.")
        else:
            for test_user_id_orig_or_idx in test['userID'].unique():
                user_idx_internal = -1
                if args.dataset in [AMAZONSALES, POSTRECOMMENDATIONS]:
                    if user_map and test_user_id_orig_or_idx in user_map:
                        user_idx_internal = user_map[test_user_id_orig_or_idx]
                    else:
                        continue # Cannot map this original ID
                else: # For Movielens, test_user_id_orig_or_idx is already an int index
                    user_idx_internal = int(test_user_id_orig_or_idx)

                if user_idx_internal != -1 and (user_idx_internal not in pred_labels or not pred_labels[user_idx_internal]):
                    problematic_user_indices_for_pred_labels.add(user_idx_internal)
        
        if problematic_user_indices_for_pred_labels:
            print(f"DEBUG (pred_labels check): {len(problematic_user_indices_for_pred_labels)} users from test set have no items in 'pred_labels'. Sample indices: {list(problematic_user_indices_for_pred_labels)[:5]}")
            print("  This means these users had no valid candidate paths or items before 'calculate_predictons' or 'decoder'.")

    # --------------- ADD THIS FILTERING STEP ---------------
    print("Filtering predictions to remove items already in the `train` DataFrame (for metrics.py compatibility)...")
    if not train.empty and (not top.empty or not top_k.empty) :
        train_interactions = train.set_index(['userID', 'itemID']).index

        if not top.empty:
            top_interactions = pd.MultiIndex.from_frame(top[['userID', 'itemID']])
            mask_top = ~top_interactions.isin(train_interactions)
            top_filtered = top[mask_top].copy()
            print(f"Original 'top' size: {len(top)}, Filtered 'top' size: {len(top_filtered)}")
        else:
            top_filtered = pd.DataFrame(columns=top.columns)

        if not top_k.empty:
            top_k_interactions = pd.MultiIndex.from_frame(top_k[['userID', 'itemID']])
            mask_top_k = ~top_k_interactions.isin(train_interactions)
            top_k_filtered = top_k[mask_top_k].copy()
            print(f"Original 'top_k' size: {len(top_k)}, Filtered 'top_k_filtered' size: {len(top_k_filtered)}")
        else:
            top_k_filtered = pd.DataFrame(columns=top_k.columns)
    else:
        print("Train DataFrame is empty or prediction DataFrames are empty, skipping filtering.")
        top_filtered = top.copy()
        top_k_filtered = top_k.copy()
    # --------------- END OF FILTERING STEP ---------------
    known_missing_user_ids_set = set() # Will be populated if users are missing
    
    if not test.empty and 'userID' in test.columns:
        test_user_ids_original = set(test['userID'].unique()) # These are original IDs after standardization

        # Check top_k BEFORE filtering (but AFTER ID mapping for Amazon)
        if not top_k.empty and 'userID' in top_k.columns:
            top_k_users_original = set(top_k['userID'].unique())
            users_missing_from_top_k_raw = test_user_ids_original - top_k_users_original
            if users_missing_from_top_k_raw:
                print(f"  DEBUG (top_k raw): {len(users_missing_from_top_k_raw)} original user IDs from test are MISSING from 'top_k' (after mapping, before filtering). Sample: {list(users_missing_from_top_k_raw)[:5]}")
        else:
            print(f"  DEBUG (top_k raw): 'top_k' is empty or has no 'userID' column.")
            if not test.empty: users_missing_from_top_k_raw = test_user_ids_original # All are missing

        # Check top_k_filtered AFTER filtering
        if not top_k_filtered.empty and 'userID' in top_k_filtered.columns:
            top_k_filtered_users_original = set(top_k_filtered['userID'].unique())
            known_missing_user_ids_set = test_user_ids_original - top_k_filtered_users_original # This is the critical set for metrics
            if known_missing_user_ids_set:
                print(f"  IMPORTANT DEBUG (top_k_filtered): {len(known_missing_user_ids_set)} original user IDs from test are MISSING from 'top_k_filtered' (AFTER filtering). Sample: {list(known_missing_user_ids_set)[:5]}")
                # If users were in top_k_raw but not in top_k_filtered, filtering removed them
                users_removed_by_filtering = (test_user_ids_original & top_k_users_original) - top_k_filtered_users_original
                if users_removed_by_filtering:
                    print(f"    Of these, {len(users_removed_by_filtering)} users had all their predictions removed by the train filtering step. Sample: {list(users_removed_by_filtering)[:3]}")

        else:
            print(f"  DEBUG (top_k_filtered): 'top_k_filtered' is empty or has no 'userID' column.")
            if not test.empty: known_missing_user_ids_set = test_user_ids_original # All are missing
    
    # 9. Calculate Metrics using metrics.py
    print(f"\n--- Calculating Metrics @{k} using metrics.py ---")
    if rating_pred_df.empty or rating_true_df.empty:
         print("Cannot calculate metrics: Prediction or Ground Truth DataFrame is empty.")
         return
    # print(f"Top K: {top_k}") #TODO: remove
    # print(f"Top: {top}") #TODO: remove
    # print(f"Train: {train}") #TODO: remove
    # print(f"Test: {test}") #TODO: remove

    print(f"\n--- Calculating Metrics @{k} using metrics.py ---")
    # if top_k_filtered.empty:
    #     print("WARNING: top_k_filtered DataFrame is empty. Metrics relying on it will likely fail or be zero.")
    # # else: # Optional: print head if not empty
    #     # print("top_k_filtered head:\n", top_k_filtered.head())

    # if top_filtered.empty:
    #     print("WARNING: top_filtered DataFrame is empty. Metrics relying on it will likely fail or be zero.")
    # # else: # Optional: print head if not empty
    #     # print("top_filtered head:\n", top_filtered.head())

    # if test.empty:
    #     print("WARNING: test DataFrame is empty. Metrics cannot be calculated.")
    #     return # Or handle appropriately

    # # Check if there are users in test that are not in top_k_filtered
    # test_users = set(test['userID', 'itemID', 'prediction'].unique())
    # pred_users_top_k = set(top_k_filtered['userID', 'itemID', 'prediction'].unique()) if not top_k_filtered.empty else set()
    # users_in_test_not_in_top_k_preds = test_users - pred_users_top_k
    # if users_in_test_not_in_top_k_preds:
    #     print(f"WARNING: {len(users_in_test_not_in_top_k_preds)} users in 'test' have no predictions in 'top_k_filtered'. This might affect per-user metrics.")
    #     missing_users_series = pd.Series(list(users_in_test_not_in_top_k_preds))
    #     # print(f"Users without predictions in top_k_filtered: {list(users_in_test_not_in_top_k_preds)}") # Print a few
    #     missing_users_series.to_csv('users_without_predictions_in_top_k_filtered.csv', index=False) # Save to CSV for further analysis

    # # Similar check for top_filtered
    # pred_users_top = set(top_filtered['userID', 'itemID', 'prediction'].unique()) if not top_filtered.empty else set()
    # users_in_test_not_in_top_preds = test_users - pred_users_top
    # if users_in_test_not_in_top_preds:
    #     print(f"WARNING: {len(users_in_test_not_in_top_preds)} users in 'test' have no predictions in 'top_filtered'.")
    #     missing_users_series_top_k = pd.Series(list(users_in_test_not_in_top_preds))
    #     # print(f"Users without predictions in top_filtered: {list(users_in_test_not_in_top_preds)}") # Print a few
    #     missing_users_series_top_k.to_csv('users_without_predictions_in_top_filtered.csv', index=False) # Save to CSV for further analysis

        # ... (inside run_evaluation, before metric calculations) ...

    # Additional checks before metric calculation
    if test.empty:
        print("CRITICAL: 'test' DataFrame is empty. Cannot calculate metrics.")
        return {}, pd.DataFrame(), {}

    if top_k_filtered.empty:
        print(f"CRITICAL for {args.dataset}: top_k_filtered DataFrame is COMPLETELY empty.")
    if top_filtered.empty:
        print(f"CRITICAL for {args.dataset}: top_filtered DataFrame is COMPLETELY empty.")

    # Columns to use for identifying unique interactions/predictions
    # Ensure these columns actually exist in your 'test', 'top_k_filtered', and 'top_filtered' DataFrames
    # 'prediction' might not be in 'test' if 'test' only contains ground truth ratings.
    # Adjust 'id_cols_test' and 'id_cols_pred' accordingly.
    
    # For test data, we usually care about (userID, itemID) for ground truth
    id_cols_test = ['userID', 'itemID']
    # For prediction data, we have (userID, itemID, prediction)
    id_cols_pred = ['userID', 'itemID', 'prediction']


    # --- Check for (userID, itemID) from test NOT in top_k_filtered's (userID, itemID) ---
    # This tells you which ground truth interactions were not recommended at all.
    if all(col in test.columns for col in id_cols_test):
        # Create a MultiIndex of (userID, itemID) from the test set
        test_interactions_idx = test.set_index(id_cols_test).index
        
        if all(col in top_k_filtered.columns for col in id_cols_pred) and not top_k_filtered.empty:
            # Create a MultiIndex of (userID, itemID) from the predictions
            pred_interactions_top_k_idx = top_k_filtered.set_index(id_cols_test).index # Use same cols for comparison
            
            # Find which test interactions are NOT in the predictions' (userID, itemID)
            missing_test_interactions_mask = ~test_interactions_idx.isin(pred_interactions_top_k_idx)
            
            # Get the actual rows from 'test' that are missing
            missing_test_interactions_df = test[missing_test_interactions_mask]

            if not missing_test_interactions_df.empty:
                print(f"WARNING: {len(missing_test_interactions_df)} (userID, itemID) interactions from 'test' set are not found among (userID, itemID) in 'top_k_filtered'.")
                # filename = f'test_interactions_not_in_top_k_filtered_{args.dataset}.csv'
                # try:
                #     # Save only the relevant columns from these missing test interactions
                #     missing_test_interactions_df[id_cols_test + ['rating']].to_csv(filename, index=False, header=True)
                #     print(f"Saved {len(missing_test_interactions_df)} test interactions not found in 'top_k_filtered' to {filename}")
                # except Exception as e:
                #     print(f"Error saving missing test interactions to CSV {filename}: {e}")
                print(f"Sample of test interactions not in 'top_k_filtered' (max 5 rows):\n{missing_test_interactions_df[id_cols_test + ['rating']].head().to_string()}")
            elif not top_k_filtered.empty :
                 print(f"INFO: All (userID, itemID) interactions from 'test' seem to have a corresponding (userID, itemID) in 'top_k_filtered'. (This doesn't mean ratings/predictions match, just that the pairs exist).")
        else:
            print(f"Skipping check for test interactions not in top_k_filtered because top_k_filtered is empty or missing columns {id_cols_pred}.")
            # If top_k_filtered is empty, then ALL test interactions are missing
            if top_k_filtered.empty and not test.empty:
                 print(f"WARNING: Since top_k_filtered is empty, all {len(test)} test interactions are effectively not present in predictions.")
                #  filename = f'test_interactions_not_in_top_k_filtered_ (because_preds_empty)_{args.dataset}.csv'
                #  try:
                #     test[id_cols_test + ['rating']].to_csv(filename, index=False, header=True)
                #     print(f"Saved all {len(test)} test interactions to {filename}")
                #  except Exception as e:
                #     print(f"Error saving all test interactions to CSV {filename}: {e}")


    else:
        print(f"WARNING: One or more columns from {id_cols_test} not found in 'test' DataFrame. Cannot perform detailed interaction missing check.")

    # --- If you specifically want to see which *users* from test have *no rows at all* in top_k_filtered ---
    # This is what the previous version of the code did, and it's still useful.
    if 'userID' in test.columns:
        test_user_ids_set = set(test['userID'].unique())
        
        if 'userID' in top_k_filtered.columns and not top_k_filtered.empty:
            pred_user_ids_top_k_set = set(top_k_filtered['userID'].unique())
        else:
            pred_user_ids_top_k_set = set()
            
        users_entirely_missing_from_top_k = test_user_ids_set - pred_user_ids_top_k_set
        if users_entirely_missing_from_top_k:
            print(f"INFO: {len(users_entirely_missing_from_top_k)} userIDs from 'test' have NO predictions AT ALL in 'top_k_filtered'.")
            missing_users_series = pd.Series(list(users_entirely_missing_from_top_k), name="userID")
            # filename_users_missing = f'users_with_no_predictions_in_top_k_filtered_{args.dataset}.csv'
            # try:
            #     missing_users_series.to_csv(filename_users_missing, index=False, header=True)
            #     print(f"Saved list of {len(missing_users_series)} users with no predictions in 'top_k_filtered' to {filename_users_missing}")
            # except Exception as e:
            #     print(f"Error saving users with no predictions to CSV {filename_users_missing}: {e}")
            print(f"Sample of users with NO predictions in 'top_k_filtered' (max 10): {list(users_entirely_missing_from_top_k)[:10]}")
            
    # ---- End of check for users with no rows ----


    # You can do a similar block for `top_filtered` if needed.
    # For example, to see which (userID, itemID, prediction) from top_filtered are NOT in test.
    # Or which (userID, itemID) from test are NOT in top_filtered.

    # ... (rest of the code: rating_pred_df check, metric calculations) ...
    if rating_pred_df.empty or rating_true_df.empty: # Your existing check
        print("Cannot calculate metrics: Prediction or Ground Truth DataFrame is empty.")
        return


    # try:
        
    # Metrics
    # eval_map = map_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction",relevancy_method="top_k", k=TOP_K)
    # eval_ndcg_at_k = ndcg_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction",relevancy_method="top_k", k=TOP_K)
    eval_precision_at_k = precision_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K)
    eval_recall_at_k = recall_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K)
    eval_ndcg = ndcg_at_k(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", relevancy_method="top_k",k=1)
    eval_precision = precision_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    eval_recall = recall_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    eval_mae = mae(test, top_k_filtered)
    eval_rmse = rmse(test, top_k_filtered)

    # eval_novelty = novelty(train, top_filtered)
    # eval_historical_item_novelty = historical_item_novelty(train, top_filtered)
    # eval_user_item_serendipity = user_item_serendipity(train, top_filtered)
    # eval_user_serendipity = user_serendipity(train, top_filtered)
    # eval_serendipity = serendipity(train, top_filtered)
    # eval_catalog_coverage = catalog_coverage(train, top_filtered)
    # eval_distributional_coverage = distributional_coverage(train, top_filtered)

    # eval_f1 = f1(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    eval_mrr = mrr(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    # eval_accuracy = accuracy(test, top_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    eval_user_coverage = user_coverage(test, top_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    eval_item_coverage = item_coverage(test, top_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")


    eval_intra_list_similarity = intra_list_similarity_score(data, top_k_filtered, feature_cols=['genres'])
    eval_intra_list_dissimilarity = intra_list_dissimilarity(data, top_k_filtered, feature_cols=['genres'])
    eval_personalization = personalization_score(train, top_filtered)

    print(
        "Precision:\t%f" % eval_precision,
        "Precision@K:\t%f" % eval_precision_at_k,
        "Recall:\t%f" % eval_recall,
        "Recall@K:\t%f" % eval_recall_at_k,
        # "F1:\t%f" % eval_f1,
        # "Accuracy:\t%f" % eval_accuracy,
        "MAE:\t%f" % eval_mae,
        "RMSE:\t%f" % eval_rmse,
        "NDCG:\t%f" % eval_ndcg,
        "MRR:\t%f" % eval_mrr,
        # "Novelty:\t%f" % eval_novelty,
        # "Serendipity:\t%f" % eval_serendipity,
        "User covarage:\t%f" % eval_user_coverage,
        "Item coverage:\t%f" % eval_item_coverage,
        # "Catalog coverage:\t%f" % eval_catalog_coverage,
        # "Distributional coverage:\t%f" % eval_distributional_coverage,
        "Personalization:\t%f" % eval_personalization,
        "Intra-list similarity:\t%f" % eval_intra_list_similarity,
        "Intra-list dissimilarity:\t%f" % eval_intra_list_dissimilarity,
      sep='\n')
    
    
    # # mlflow
    metrics = {
            "precision": eval_precision,
            "precision_at_k": eval_precision_at_k,
            "recall": eval_recall,
            "recall_at_k": eval_recall_at_k,
            # "f1": eval_f1,
            "mae": eval_mae,                      
            "rmse": eval_rmse,
            "mrr": eval_mrr,                    
            "ndcg_at_k": eval_ndcg,               
            # "novelty": eval_novelty,
            # "serendipity": eval_serendipity,
            "user_coverage": eval_user_coverage,  
            "item_coverage": eval_item_coverage,
            # "catalog_coverage": eval_catalog_coverage,
            # "distributional_coverage": eval_distributional_coverage,
            "personalization": eval_personalization,
            "intra_list_similarity": eval_intra_list_similarity,
            "intra_list_dissimilarity": eval_intra_list_dissimilarity,
        }
    return metrics, rating_pred_df, human_recs


def test(TOP_K, want_col, num_rows, ratio, data_df, train_df, test_df, privacy, personalization, args):
    policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)

    if not os.path.exists(policy_file):
        print(f"ERROR: Policy file not found: {policy_file}")
        print("Please ensure the agent has been trained for the specified number of epochs.")
        return

    if args.run_path or not os.path.exists(path_file):
         print(f"Running path prediction (run_path={args.run_path}, file_exists={os.path.exists(path_file)})...")
         model = predict_paths(policy_file, path_file, args)
    else:
         print(f"Skipping path prediction, using existing file: {path_file}")

    if args.run_eval:
        if not os.path.exists(path_file):
             print(f"ERROR: Cannot run evaluation because path file does not exist: {path_file}")
             return

        print("Loading data labels for evaluation...")
        try:
            train_labels = load_labels(args.dataset, 'train')
            test_labels = load_labels(args.dataset, 'test')
        except FileNotFoundError as e:
             print(f"ERROR: Could not load label file: {e}. Make sure preprocessing was completed.")
             return
        except Exception as e:
             print(f"ERROR loading labels: {e}")
             return

        metrics, rating_pred_df, human_recs = run_evaluation(path_file, train_labels, test_labels, TOP_K, data_df, train_df, test_df, args)

        rl_hyperparams = {
            "dataset": args.dataset,
            "want_col": want_col,
            "num_rows": num_rows,
            "ratio": ratio,
            "seed": args.seed,
            'epochs_loaded': args.epochs,
            'max_acts': args.max_acts,
            'max_path_len': args.max_path_len,
            'gamma': args.gamma,
            'state_history': args.state_history,
            'hidden_sizes': args.hidden,
        }
        
        # data & train in preprocess pahse there is a data and train df
        # print('########################################################################') 
        # print(rating_pred_df) 
        # print(type(rating_pred_df))
        # print('########################################################################')
        # print(human_recs)
        # print('########################################################################')

        log_mlflow.log_mlflow(args.dataset, rating_pred_df, metrics, num_rows, args.seed, model, 'RL', rl_hyperparams, data_df, train_df, privacy=privacy, personalization=personalization) #human_recs_top_k is a dict and dont have atribute head - i have to provide here a single user top_k

    else:
        print("Skipping evaluation.")



def test_agent_rl(dataset, TOP_K, want_col, num_rows, ratio, seed, data_df, train_df, test_df, privacy, personalization):
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset name (set automatically).')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name matching training.')
    parser.add_argument('--seed', type=int, default=seed, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='Epoch number of the model to load.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions (must match training).')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length (must match training).')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor (used in model definition).')
    parser.add_argument('--state_history', type=int, default=1, help='state history length (must match training).')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='hidden layer sizes (must match training).')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 1], help='Beam search topk width per hop.')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    if not os.path.isdir(args.log_dir):
         print(f"Warning: Log directory {args.log_dir} not found. Ensure '--name' matches training.")

    set_random_seed(args.seed)
    test(TOP_K, want_col, num_rows, ratio, data_df, train_df, test_df, privacy, personalization, args)

