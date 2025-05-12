from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from math import log
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import threading
from functools import reduce

import metrics
from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from decoder import RLRecommenderDecoder
from train_agent import ActorCritic
from utils import *


def evaluate(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 10:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
        if len(pred_list) == 0:
            continue

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))


def batch_beam_search(env, model, uids, device, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=bool) 
            act_mask[:num_acts] = True
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(3): # Assuming max_path_len is 3
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.BoolTensor(actmask_pool).to(device)
        # Ensure model is in eval mode if dropout/batchnorm used (it is set above)
        with torch.no_grad(): # Ensure no gradients are computed
            probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]

        # Masking probabilities (handling potential -inf from model)
        # Add a small epsilon for numerical stability before taking log if needed,
        # but softmax should handle large negative numbers from masking.
        # probs = probs + actmask_tensor.float() # Original approach might be problematic if probs are already large negatives
        # A safer way to ensure masked actions have zero probability after softmax:
        probs[~actmask_tensor] = -float('inf') # Set masked actions to negative infinity
        probs = F.softmax(probs, dim=1) # Renormalize (masked actions will have prob 0)

        # Check for NaNs after softmax, which might indicate issues
        if torch.isnan(probs).any():
            print("Warning: NaNs detected in probabilities after softmax!")
            # Handle NaN case, e.g., replace NaNs or use uniform distribution over valid actions
            # For now, let's just print a warning and proceed, potentially replacing NaN rows later
            # A simple fix might be to replace NaN rows with uniform distribution over valid actions
            nan_rows = torch.isnan(probs).any(dim=1)
            for r_idx in torch.where(nan_rows)[0]:
                num_valid_actions = actmask_tensor[r_idx].sum().item()
                if num_valid_actions > 0:
                    probs[r_idx, actmask_tensor[r_idx]] = 1.0 / num_valid_actions
                    probs[r_idx, ~actmask_tensor[r_idx]] = 0.0
                else: # No valid actions? Should not happen with self-loop usually
                    probs[r_idx] = 0.0 # Or handle appropriately

        # Use torch.topk on the probabilities
        # Ensure topk[hop] is not larger than the number of valid actions
        # This requires careful handling if some rows have fewer valid actions than k
        # For simplicity, let's assume k is reasonable or handle potential errors
        current_k = min(topk[hop], probs.shape[1]) # Adjust k if needed
        try:
             topk_probs, topk_idxs = torch.topk(probs, current_k, dim=1)  # Use probabilities directly
        except RuntimeError as e:
            print(f"Error during topk: {e}. Check k value ({current_k}) vs dimensions.")
            # Fallback or error handling needed here
            # Example: reduce k if it's too large for some rows (more complex)
            # For now, let's just re-raise or return empty
            raise e


        topk_idxs = topk_idxs.cpu().numpy()
        topk_probs = topk_probs.cpu().numpy() # These are actual probabilities now

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            # Note: probs_pool previously stored log-probs or scores.
            # Now storing actual probabilities. Need to decide if product or sum of logs is desired later.
            # Let's stick to product of probabilities for now.
            current_path_prob = reduce(lambda x, y: x * y, probs_pool[row]) if probs_pool[row] else 1.0

            # Handling cases where the number of valid actions < k
            valid_action_indices_in_row = np.where(actmask_pool[row])[0]

            for idx_in_topk, p in zip(topk_idxs[row], topk_probs[row]):
                # idx_in_topk is the index within the original action space (0 to act_dim-1)
                # Check if the selected index corresponds to a valid action for this row
                 if idx_in_topk >= len(acts_pool[row]): # This index is out of bounds for the specific actions list
                     # This can happen if topk returns indices beyond the actual number of actions available
                     # (e.g., if act_dim is larger than len(acts_pool[row]))
                     # Or if the masking/probability calculation had issues.
                     # print(f"Warning: Skipped topk index {idx_in_topk} >= action list length {len(acts_pool[row])} for row {row}")
                     continue

                 if not actmask_pool[row][idx_in_topk]: # Double check mask just in case
                      # print(f"Warning: Skipped topk index {idx_in_topk} because it's masked out for row {row}")
                      continue

                 # Ensure p is not zero or negative if we intend to multiply probabilities
                 if p <= 0:
                      # This shouldn't happen with softmax unless all inputs were -inf
                      # print(f"Warning: Probability p={p} <= 0 encountered for index {idx_in_topk}, row {row}. Skipping.")
                      continue


                 # Retrieve action details using the valid index
                 relation, next_node_id = acts_pool[row][idx_in_topk]

                 # Determine next node type
                 if relation == SELF_LOOP:
                     next_node_type = path[-1][1]
                 else:
                    # Need safety check for relation key
                    current_node_type = path[-1][1]
                    if current_node_type not in KG_RELATION or relation not in KG_RELATION[current_node_type]:
                         print(f"Warning: Invalid relation '{relation}' for node type '{current_node_type}'. Skipping path extension.")
                         continue
                    next_node_type = KG_RELATION[current_node_type][relation]

                 # Append new path segment and probability
                 new_path = path + [(relation, next_node_type, next_node_id)]
                 new_path_pool.append(new_path)
                 # Store the probability of *this step*
                 new_probs_pool.append(probs_pool[row] + [p]) # Append probability

        path_pool = new_path_pool
        probs_pool = new_probs_pool

        # Check if path_pool is empty, if so, break to avoid errors
        if not path_pool:
             print("Warning: Beam search resulted in an empty path pool. Terminating early.")
             break

        if hop < env.max_num_nodes - 2: # Need state for the next hop unless it's the last one
            try:
                state_pool = env._batch_get_state(path_pool)
            except IndexError as e:
                 print(f"Error getting state at hop {hop+1}: {e}. Path pool might be malformed.")
                 print("Example path causing error:", path_pool[0] if path_pool else "None")
                 break # Stop beam search if state cannot be computed

    # Calculate final path probabilities by multiplying step probabilities
    final_probs = [reduce(lambda x, y: x * y, p_list) if p_list else 0.0 for p_list in probs_pool]

    return path_pool, final_probs # Return paths and final probabilities


def predict_paths(policy_file, path_file, args):
    # ... (load model, env - no changes needed here) ...
    print('Predicting paths using beam search...')
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    # Load the state dict from the checkpoint file
    try:
        pretrain_sd = torch.load(policy_file, map_location=args.device) # Load to target device
        print(f"Successfully loaded policy file: {policy_file}")
    except FileNotFoundError:
        print(f"ERROR: Policy file not found at {policy_file}")
        return # Cannot proceed without the model
    except Exception as e:
        print(f"ERROR: Failed to load policy file {policy_file}: {e}")
        return

    # Initialize the model
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    # Load the state dict into the model
    try:
        model.load_state_dict(pretrain_sd)
        print("Successfully loaded state dict into the model.")
    except RuntimeError as e:
        print(f"ERROR: Failed to load state dict. Mismatched keys or layers?: {e}")
        # Option: Load with strict=False if some keys are expected to be missing/different
        # model.load_state_dict(pretrain_sd, strict=False)
        # print("Attempted loading with strict=False")
        return # Or handle error as appropriate
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading the state dict: {e}")
        return

    model.eval() # Set model to evaluation mode

    test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())

    batch_size = 16 # Reduced batch size for potentially complex beam search
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        try:
             # Use torch.no_grad() during inference
             with torch.no_grad():
                 paths, probs = batch_beam_search(env, model, batch_uids, args.device, topk=args.topk)
             all_paths.extend(paths)
             all_probs.extend(probs) # Store final path probabilities
        except Exception as e:
             print(f"\nERROR during beam search for batch starting at index {start_idx}: {e}")
             # Decide how to handle: skip batch, log error, etc.
             # For now, just print and continue, results might be incomplete.
             pass # Continue to the next batch

        start_idx = end_idx
        pbar.update(len(batch_uids)) # Update progress bar correctly
    pbar.close() # Close progress bar

    # Check if any paths were generated
    if not all_paths:
         print("ERROR: No paths were generated during prediction. Check beam search logic and model loading.")
         return

    predicts = {'paths': all_paths, 'probs': all_probs} # Save final probabilities
    print(f"Saving {len(all_paths)} paths and probabilities to {path_file}")
    try:
        with open(path_file, 'wb') as f:
            pickle.dump(predicts, f)
    except Exception as e:
         print(f"ERROR: Failed to save predictions to {path_file}: {e}")


def run_evaluation(path_file, train_labels, test_labels, args):
    # === NEW EVALUATION LOGIC using Decoder and metrics.py ===

    print("Starting evaluation using Decoder and metrics.py...")
    k = 10 # Define the 'k' for top-k metrics

    # 1. Load path predictions
    print(f"Loading predicted paths from {path_file}")
    try:
        results = pickle.load(open(path_file, 'rb'))
        pred_paths_raw = results['paths']
        pred_probs_raw = results['probs'] # Load probabilities
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
        purchase_embeds = embeds[PURCHASE][0]
        product_embeds = embeds[ITEMID]
        # Calculate base scores (user+purchase affinity for items) - might be optional if using path prob
        base_scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)
    except Exception as e:
         print(f"Warning: Could not load embeddings or compute base scores: {e}. Relying solely on path probability.")
         base_scores = None # Mark as unavailable


    # 3. Process paths and probabilities to get candidate items per user
    print("Processing paths to select candidate items...")
    pred_paths_by_user = {uid: {} for uid in test_labels} # {uid: {pid: [(score, prob, path)]}}

    for path, path_prob in zip(pred_paths_raw, pred_probs_raw):
        if not path or path[-1][1] != ITEMID: # Ensure path is valid and ends with an item
            continue
        uid = path[0][2]
        if uid not in pred_paths_by_user: # Only process users in the test set
            continue
        pid = path[-1][2]

        # Calculate path score (optional, can use if base_scores available)
        path_score = base_scores[uid][pid] if base_scores is not None else 0.0

        if pid not in pred_paths_by_user[uid]:
            pred_paths_by_user[uid][pid] = []
        # Store both score and probability
        pred_paths_by_user[uid][pid].append((path_score, path_prob, path))

    # 4. Select best path per (user, item) and filter train items
    print("Selecting best path per user-item pair and filtering training items...")
    best_path_candidates = {} # {uid: [(score, prob, item_idx)]}

    for uid, item_paths_dict in pred_paths_by_user.items():
        train_pids = set(train_labels.get(uid, [])) # Get train items for the user
        user_candidates = []
        for pid, path_tuples in item_paths_dict.items():
            if pid in train_pids:
                continue # Skip items already seen in training

            # Sort paths for this item based on probability (primary) and score (secondary)
            sorted_paths_for_item = sorted(path_tuples, key=lambda x: (x[1], x[0]), reverse=True)
            best_score, best_prob, _ = sorted_paths_for_item[0] # Get score and prob of best path
            user_candidates.append((best_score, best_prob, pid)) # Store (score, prob, item_idx)

        best_path_candidates[uid] = user_candidates

    # 5. Generate final ranked list (pred_labels format: lowest score first)
    print(f"Generating final top-{k} ranked list for {len(best_path_candidates)} users...")
    pred_labels = {}
    sort_by = 'prob' # Or 'score' if base_scores are reliable

    for uid, candidates in best_path_candidates.items():
        if sort_by == 'prob':
             # Sort by probability (desc), then score (desc)
            sorted_candidates = sorted(candidates, key=lambda x: (x[1], x[0]), reverse=True)
        else: # sort by score
             # Sort by score (desc), then probability (desc)
            sorted_candidates = sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)

        # Extract top K item IDs
        top_k_pids = [pid for _, _, pid in sorted_candidates[:k]]

        # Add filler products if enabled and needed (Using base scores for filler)
        if args.add_products and len(top_k_pids) < k and base_scores is not None:
             train_pids = set(train_labels.get(uid, []))
             cand_pids_from_scores = np.argsort(base_scores[uid]) # Indices sorted ascending score
             num_needed = k - len(top_k_pids)
             filled_count = 0
             for cand_pid in cand_pids_from_scores[::-1]: # Iterate descending score
                 if cand_pid not in train_pids and cand_pid not in top_k_pids:
                     top_k_pids.append(cand_pid)
                     filled_count += 1
                     if filled_count >= num_needed:
                         break

        # Store in pred_labels format (lowest score first for decoder)
        pred_labels[uid] = top_k_pids[::-1] # Reverse to lowest score first

    # --- End of Recommendation Generation ---

    # 6. Instantiate Decoder
    print("Instantiating decoder...")
    decoder = RLRecommenderDecoder(args.dataset)

    # 7. Decode Recommendations
    print("Decoding recommendations...")
    # The decoder expects pred_labels with lowest score first
    human_recs, rating_pred_df = decoder.decode(pred_labels, k=k)

    # Optionally print/save human-readable recommendations
    # Example: Print for first 5 users
    print("\n--- Sample Human-Readable Recommendations ---")
    count = 0
    for user_idx, recs in human_recs.items():
        print(f"User: {user_idx}")
        for r in recs:
             print(f"  Rank {r['rank']}: ID={r['original_id']} (Idx={r['item_idx']}), Title='{r['title']}', Genres={r['genres']}")
        count += 1
        if count >= 5:
            break
    print("--------------------------------------------")


    # 8. Prepare Ground Truth DataFrame for metrics.py
    print("Preparing ground truth DataFrame...")
    true_data = []
    for user_idx, item_idxs in test_labels.items():
        for item_idx in item_idxs:
            true_data.append({
                USERID: user_idx,
                ITEMID: item_idx,
                 # Assuming relevance = 1 for items in test set
                RATING: 1.0
            })
    rating_true_df = pd.DataFrame(true_data)
     # Ensure correct types
    if not rating_true_df.empty:
        rating_true_df[USERID] = rating_true_df[USERID].astype(int)
        rating_true_df[ITEMID] = rating_true_df[ITEMID].astype(int)
        rating_true_df[RATING] = rating_true_df[RATING].astype(float)


    # 9. Calculate Metrics using metrics.py
    print(f"\n--- Calculating Metrics @{k} using metrics.py ---")
    if rating_pred_df.empty or rating_true_df.empty:
         print("Cannot calculate metrics: Prediction or Ground Truth DataFrame is empty.")
         return

    # Ensure the metrics module/functions are correctly imported
    try:
        precision = metrics.precision_at_k(rating_true_df, rating_pred_df, k=k, col_user=USERID, col_item=ITEMID, col_rating=RATING, col_prediction=PREDICTION)
        recall = metrics.recall_at_k(rating_true_df, rating_pred_df, k=k, col_user=USERID, col_item=ITEMID, col_rating=RATING, col_prediction=PREDICTION)
        # Add other metrics as needed, checking their required inputs
        # ndcg = metrics.ndcg_at_k(...) # Requires relevance column or different calculation
        # mrr_score = metrics.mrr(...)

        print(f"Precision@{k} = {precision:.4f}")
        print(f"Recall@{k}    = {recall:.4f}")
        # print(f"NDCG@{k}      = {ndcg:.4f}") # Uncomment if calculated
        # print(f"MRR           = {mrr_score:.4f}") # Uncomment if calculated

        # Example: Calculate Diversity/Personalization if item features are available
        # You would need to load item features similar to how embeddings are loaded
        # item_features_df = load_item_features(args.dataset) # Hypothetical function
        # if item_features_df is not None:
        #     feature_cols = [...] # List of feature columns (e.g., from genres)
        #     ils = metrics.intra_list_similarity_score(item_features_df, rating_pred_df, feature_cols=feature_cols, col_item=ITEMID)
        #     pers = metrics.personalization_score(rating_true_df, rating_pred_df) # Note: personalization_score might only need rating_pred
        #     print(f"Intra-List Similarity = {ils:.4f}")
        #     print(f"Personalization       = {pers:.4f}")

    except AttributeError as e:
         print(f"ERROR calling metrics function: {e}. Check function names and imports in metrics.py.")
    except Exception as e:
         print(f"ERROR during metric calculation: {e}")

    print("--------------------------------------------\n")


def test(args):
    policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)

    # Check if policy file exists before proceeding
    if not os.path.exists(policy_file):
        print(f"ERROR: Policy file not found: {policy_file}")
        print("Please ensure the agent has been trained for the specified number of epochs.")
        return

    # Generate paths if requested or if path file doesn't exist
    if args.run_path or not os.path.exists(path_file):
         print(f"Running path prediction (run_path={args.run_path}, file_exists={os.path.exists(path_file)})...")
         predict_paths(policy_file, path_file, args)
    else:
         print(f"Skipping path prediction, using existing file: {path_file}")


    # Run evaluation if requested and path file exists
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

        # ** Call the NEW evaluation function **
        run_evaluation(path_file, train_labels, test_labels, args)

    else:
        print("Skipping evaluation.")



def test_agent_rl(dataset):
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset name (set automatically).')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name matching training.') # Match training name
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='Epoch number of the model to load.') # Which model epoch to test
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

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name # Path to the training log directory
    # Make sure log_dir exists if needed (e.g., for loading) but testing shouldn't create it.
    if not os.path.isdir(args.log_dir):
         print(f"Warning: Log directory {args.log_dir} not found. Ensure '--name' matches training.")
         # Decide if this is critical. If loading model fails later, that's the real issue.

    set_random_seed(args.seed) # Set seed for reproducibility of beam search randomness? (if any)
    test(args) # Call the main test function

