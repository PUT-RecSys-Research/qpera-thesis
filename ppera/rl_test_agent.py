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
    # print(all_users_top_k_full_candidates)
    top, top_k = calculate_predictons(all_users_top_k_full_candidates, median_score, scale_factor, k)

    print(f"\n--- Ensuring Consistent DataFrame Dtypes for AmazonSales ---")

    # --- Debug: Print Dtypes Before Standardization ---
    print("--- Dtypes BEFORE standardization ---")
    dfs_to_check_before = {'train': train, 'test': test}
    if 'top' in locals() and top is not None and not top.empty:
        dfs_to_check_before['top'] = top
    if 'top_k' in locals() and top_k is not None and not top_k.empty:
        dfs_to_check_before['top_k'] = top_k

    for name, df_val in dfs_to_check_before.items():
        if df_val is not None and not df_val.empty:
            print(f"Dtypes for '{name}':\n{df_val.dtypes.to_string()}")
            # print(f"Sample head for '{name}':\n{df_val.head(2)}") # Optional for value inspection
        else:
            print(f"DataFrame '{name}' is empty or None.")
    print("------------------------------------")

    # --- Standardize ID and other relevant column dtypes ---
    # For AmazonSales, userID and itemID are expected to be strings.
    # Ratings and predictions are expected to be numeric (float).

    # Create copies to modify, or modify in place if you are sure it's safe.
    # Here, we'll assume modifying the passed DataFrames (train, test) is acceptable,
    # and top/top_k are freshly created.
    
    dfs_to_standardize_list = []
    if train is not None and not train.empty: dfs_to_standardize_list.append(train)
    if test is not None and not test.empty: dfs_to_standardize_list.append(test)
    if 'top' in locals() and top is not None and not top.empty: dfs_to_standardize_list.append(top)
    if 'top_k' in locals() and top_k is not None and not top_k.empty: dfs_to_standardize_list.append(top_k)

    # --- Standardize ID and other relevant column dtypes ---
    id_columns = ['userID', 'itemID']
    numeric_columns = {'rating': float, 'prediction': float}

    # Determine target dtype for ID columns based on dataset
    if args.dataset == AMAZONSALES: # AMAZONSALES is defined in rl_utils.py
        target_id_dtype = str
        print(f"Standardizing ID columns to STRING for dataset: {args.dataset}")
    else: # For Movielens and potentially others, assume integer IDs
        target_id_dtype = int
        print(f"Standardizing ID columns to INTEGER for dataset: {args.dataset}")

    for df_item in dfs_to_standardize_list:
        if df_item is None or df_item.empty: # Skip empty/None DFs
            continue
        for col_name in id_columns:
            if col_name in df_item.columns:
                # Check if current dtype is different from the target AND not already a compatible string type if target is str
                if target_id_dtype == str:
                    if not (df_item[col_name].dtype == 'object' or isinstance(df_item[col_name].dtype, pd.StringDtype)):
                        print(f"Converting column '{col_name}' in DataFrame to string. Original dtype: {df_item[col_name].dtype}")
                        df_item[col_name] = df_item[col_name].astype(str)
                elif target_id_dtype == int:
                    if not pd.api.types.is_integer_dtype(df_item[col_name]):
                        print(f"Converting column '{col_name}' in DataFrame to integer. Original dtype: {df_item[col_name].dtype}")
                        try:
                            df_item[col_name] = pd.to_numeric(df_item[col_name], errors='raise').astype(int)
                        except ValueError as e:
                            print(f"ERROR: Could not convert column '{col_name}' to int for {args.dataset}: {e}. This column might contain non-numeric strings.")
                            # Potentially raise an error here or handle as appropriate for your logic
                            # For now, let's see if this fixes the merge error. If IDs truly can't be int for Movielens,
                            # then the assumption "else integer IDs" is wrong.
                # else: # Add other dtypes if necessary
                #     pass

        for col_name, target_type in numeric_columns.items():
            if col_name in df_item.columns:
                if not pd.api.types.is_numeric_dtype(df_item[col_name]) or df_item[col_name].dtype != target_type:
                    print(f"Converting column '{col_name}' in DataFrame to {target_type}. Original dtype: {df_item[col_name].dtype}")
                    try:
                        df_item[col_name] = df_item[col_name].astype(target_type)
                    except ValueError as e:
                        print(f"Warning: Could not convert column '{col_name}' to {target_type}: {e}. Trying pd.to_numeric with coercion.")
                        df_item[col_name] = pd.to_numeric(df_item[col_name], errors='coerce')


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
    human_recs, rating_pred_df = decoder.decode(pred_labels, k=k)

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

    # 9. Calculate Metrics using metrics.py
    print(f"\n--- Calculating Metrics @{k} using metrics.py ---")
    if rating_pred_df.empty or rating_true_df.empty:
         print("Cannot calculate metrics: Prediction or Ground Truth DataFrame is empty.")
         return
    print('KURWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    print(f"Top K: {top_k}") #TODO: remove
    print(f"Top: {top}") #TODO: remove
    print(f"Train: {train}") #TODO: remove
    print(f"Test: {test}") #TODO: remove
    # try:
        
    # Metrics
    eval_map = map_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction",relevancy_method="top_k", k=TOP_K)
    # eval_ndcg_at_k = ndcg_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction",relevancy_method="top_k", k=TOP_K)
    eval_precision_at_k = precision_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K)
    eval_recall_at_k = recall_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K)
    eval_ndcg = ndcg_at_k(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", relevancy_method="top_k",k=1)
    eval_precision = precision_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    eval_recall = recall_at_k(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    eval_mae = mae(test, top_k_filtered)
    eval_rmse = rmse(test, top_k_filtered)

    eval_novelty = novelty(train, top_filtered)
    # eval_historical_item_novelty = historical_item_novelty(train, top_filtered)
    # eval_user_item_serendipity = user_item_serendipity(train, top_filtered)
    # eval_user_serendipity = user_serendipity(train, top_filtered)
    eval_serendipity = serendipity(train, top_filtered)
    eval_catalog_coverage = catalog_coverage(train, top_filtered)
    eval_distributional_coverage = distributional_coverage(train, top_filtered)

    eval_f1 = f1(test, top_k_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    # eval_mrr = mrr(test, top_filtered, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
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
        "F1:\t%f" % eval_f1,
        # "Accuracy:\t%f" % eval_accuracy,
        "MAE:\t%f" % eval_mae,
        "RMSE:\t%f" % eval_rmse,
        "NDCG:\t%f" % eval_ndcg,
        # "MRR:\t%f" % eval_mrr,
        "Novelty:\t%f" % eval_novelty,
        "Serendipity:\t%f" % eval_serendipity,
        "User covarage:\t%f" % eval_user_coverage,
        "Item coverage:\t%f" % eval_item_coverage,
        "Catalog coverage:\t%f" % eval_catalog_coverage,
        "Distributional coverage:\t%f" % eval_distributional_coverage,
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
            "f1": eval_f1,
            "mae": eval_mae,                      
            "rmse": eval_rmse,                    
            "ndcg_at_k": eval_ndcg,               
            "novelty": eval_novelty,
            "serendipity": eval_serendipity,
            "user_coverage": eval_user_coverage,  
            "item_coverage": eval_item_coverage,
            "catalog_coverage": eval_catalog_coverage,
            "distributional_coverage": eval_distributional_coverage,
            "personalization": eval_personalization,
            "intra_list_similarity": eval_intra_list_similarity,
            "intra_list_dissimilarity": eval_intra_list_dissimilarity,
        }
    return metrics, rating_pred_df, human_recs


def test(TOP_K, want_col, num_rows, ratio, data_df, train_df, test_df, args):
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
        # log_mlflow.log_mlflow(args.dataset, rating_pred_df, metrics, num_rows, args.seed, model, 'RL', rl_hyperparams, data_df, train_df) #human_recs_top_k is a dict and dont have atribute head - i have to provide here a single user top_k

    else:
        print("Skipping evaluation.")



def test_agent_rl(dataset, TOP_K, want_col, num_rows, ratio, seed, data_df, train_df, test_df):
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
    test(TOP_K, want_col, num_rows, ratio, data_df, train_df, test_df, args)

