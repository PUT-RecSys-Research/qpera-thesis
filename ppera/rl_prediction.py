import numpy as np
import pandas as pd


def calculate_predictons(all_users_top_k_full_candidates, median_kge_score, scale_factor_kge, top_k):
    """
    Applies sigmoid function to calculate [0,5] predictions.
    Generates Top-1 predictions for all users and detailed Top-K for the first user encountered.

    Args:
        all_users_top_k_full_candidates (dict):
            {uid: [(KGE_score1, path_prob1, pid1), (KGE_score2, path_prob2, pid2), ...]}
            The inner list IS ALREADY sorted by KGE score (desc), then path_prob (desc).
        center_kge_score (float): The globally determined center for KGE scores.
        scale_factor_kge (float): The globally determined scale factor for KGE scores.
        top_k (int): Number of top items for the detailed report of the first user.

    Returns:
        tuple: (pd.DataFrame for all users' top-1, pd.DataFrame for the first user's top-k)
    """

    all_users_top_1_data = []
    all_users_top_k_data = []

    if median_kge_score is None or scale_factor_kge is None or scale_factor_kge == 0:
        print("Error: center_kge_score or scale_factor_kge is invalid. Cannot calculate sigmoid predictions.")
        # Return empty DataFrames with expected columns
        cols_top1 = cols_topk = ["user_id", "item_id", "prediction"]
        return pd.DataFrame(columns=cols_top1), pd.DataFrame(columns=cols_topk)

    first_user_id_for_detailed_report = None
    # Determine the ID of the first user to process for the detailed report
    if all_users_top_k_full_candidates:
        try:
            first_user_id_for_detailed_report = next(iter(all_users_top_k_full_candidates))
            print(f"Will generate detailed Top-{top_k} report for the first user encountered: User ID {first_user_id_for_detailed_report}")
        except StopIteration:
            # This should not happen if the dictionary is not empty
            print("Warning: Could not get the first user ID from non-empty candidate dictionary.")
    else:
        print("Warning: all_users_top_k_full_candidates is empty. No reports will be generated.")

    for user_id, candidates_for_user in all_users_top_k_full_candidates.items():
        if not candidates_for_user:  # No candidates for this user
            continue

        # --- Process Top-1 for all_users_top_1_data ---
        top_1_kge_score, top_1_path_prob, top_1_item_id = candidates_for_user[0]

        z_top1 = (top_1_kge_score - median_kge_score) / scale_factor_kge
        try:
            prob_like_top1 = 1.0 / (1.0 + np.exp(-z_top1))
        except OverflowError:
            prob_like_top1 = 1.0 if z_top1 > 0 else 0.0
        scaled_pred_top1 = prob_like_top1 * 5.0

        all_users_top_1_data.append({"userID": user_id, "itemID": top_1_item_id, "prediction": scaled_pred_top1})

        # --- Process Top-K for specific_user_top_k_data (if applicable) ---
        rank = 0
        # Iterate through the candidates for this user, up to k_for_top_k_output items
        for kge_score, path_prob, item_id in candidates_for_user[:top_k]:
            rank += 1
            current_kge_score = kge_score

            z = (current_kge_score - median_kge_score) / scale_factor_kge
            try:
                probability_like_score = 1.0 / (1.0 + np.exp(-z))
            except OverflowError:
                probability_like_score = 1.0 if z > 0 else 0.0
            scaled_prediction = probability_like_score * 5.0

            all_users_top_k_data.append(
                {
                    "userID": user_id,
                    "itemID": item_id,
                    "prediction": scaled_prediction,
                }
            )

    df_all_users_top_1 = pd.DataFrame(all_users_top_1_data)
    df_specific_user_top_k = pd.DataFrame(all_users_top_k_data)

    return df_all_users_top_1, df_specific_user_top_k
