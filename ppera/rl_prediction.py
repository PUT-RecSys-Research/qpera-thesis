from __future__ import absolute_import, division, print_function

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def calculate_predictions(
    all_users_top_k_full_candidates: Dict[int, List[Tuple[float, float, int]]], median_kge_score: Optional[float], scale_factor_kge: Optional[float], top_k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate sigmoid-based predictions for user-item recommendations.

    Applies sigmoid function to calculate [0,5] predictions based on KGE scores.
    Generates Top-1 predictions for all users and detailed Top-K for all users.

    Args:
        all_users_top_k_full_candidates: Dictionary mapping user IDs to lists of
            (KGE_score, path_prob, item_id) tuples. Inner lists should be sorted
            by KGE score (descending), then path_prob (descending).
        median_kge_score: Global center value for KGE score normalization
        scale_factor_kge: Global scale factor for KGE score normalization
        top_k: Number of top items to include in detailed report

    Returns:
        Tuple containing:
        - DataFrame with top-1 predictions for all users
        - DataFrame with top-k predictions for all users

    Raises:
        ValueError: If scale_factor_kge is zero
    """
    # Validate input parameters
    if not _validate_input_parameters(median_kge_score, scale_factor_kge, top_k):
        return _create_empty_dataframes()

    if not all_users_top_k_full_candidates:
        print("Warning: No candidates provided. Returning empty DataFrames.")
        return _create_empty_dataframes()

    print(f"Processing predictions for {len(all_users_top_k_full_candidates)} users...")

    # Generate predictions
    top_1_data = []
    top_k_data = []

    for user_id, candidates in all_users_top_k_full_candidates.items():
        if not candidates:
            print(f"Warning: No candidates for user {user_id}. Skipping.")
            continue

        # Process top-1 prediction
        top_1_prediction = _process_top_1_prediction(user_id, candidates[0], median_kge_score, scale_factor_kge)
        if top_1_prediction:
            top_1_data.append(top_1_prediction)

        # Process top-k predictions
        user_top_k_predictions = _process_top_k_predictions(user_id, candidates[:top_k], median_kge_score, scale_factor_kge)
        top_k_data.extend(user_top_k_predictions)

    # Create DataFrames
    df_top_1 = pd.DataFrame(top_1_data)
    df_top_k = pd.DataFrame(top_k_data)

    print(f"Generated {len(df_top_1)} top-1 predictions and {len(df_top_k)} top-k predictions.")

    return df_top_1, df_top_k


def _validate_input_parameters(median_kge_score: Optional[float], scale_factor_kge: Optional[float], top_k: int) -> bool:
    """Validate input parameters for prediction calculation."""
    if median_kge_score is None:
        print("Error: median_kge_score is None. Cannot calculate predictions.")
        return False

    if scale_factor_kge is None:
        print("Error: scale_factor_kge is None. Cannot calculate predictions.")
        return False

    if scale_factor_kge == 0:
        print("Error: scale_factor_kge is zero. Cannot calculate predictions.")
        return False

    if top_k <= 0:
        print("Error: top_k must be positive.")
        return False

    return True


def _create_empty_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create empty DataFrames with expected column structure."""
    columns = ["userID", "itemID", "prediction"]
    return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)


def _process_top_1_prediction(
    user_id: int, top_candidate: Tuple[float, float, int], median_kge_score: float, scale_factor_kge: float
) -> Optional[Dict[str, float]]:
    """Process top-1 prediction for a single user."""
    try:
        kge_score, path_prob, item_id = top_candidate
        prediction = _calculate_sigmoid_prediction(kge_score, median_kge_score, scale_factor_kge)

        return {"userID": user_id, "itemID": item_id, "prediction": prediction}
    except Exception as e:
        print(f"Error processing top-1 prediction for user {user_id}: {e}")
        return None


def _process_top_k_predictions(
    user_id: int, candidates: List[Tuple[float, float, int]], median_kge_score: float, scale_factor_kge: float
) -> List[Dict[str, float]]:
    """Process top-k predictions for a single user."""
    predictions = []

    for kge_score, path_prob, item_id in candidates:
        try:
            prediction = _calculate_sigmoid_prediction(kge_score, median_kge_score, scale_factor_kge)

            predictions.append({"userID": user_id, "itemID": item_id, "prediction": prediction})
        except Exception as e:
            print(f"Error processing prediction for user {user_id}, item {item_id}: {e}")
            continue

    return predictions


def _calculate_sigmoid_prediction(kge_score: float, median_kge_score: float, scale_factor_kge: float) -> float:
    """
    Calculate sigmoid-based prediction score.

    Args:
        kge_score: Raw KGE score
        median_kge_score: Median for normalization
        scale_factor_kge: Scale factor for normalization

    Returns:
        Prediction score in range [0, 5]
    """
    # Normalize the score
    z = (kge_score - median_kge_score) / scale_factor_kge

    # Apply sigmoid with overflow protection
    try:
        probability = 1.0 / (1.0 + np.exp(-z))
    except OverflowError:
        probability = 1.0 if z > 0 else 0.0

    # Scale to [0, 5] range
    return probability * 5.0


def calculate_prediction_statistics(all_users_top_k_full_candidates: Dict[int, List[Tuple[float, float, int]]]) -> Dict[str, float]:
    """
    Calculate statistics for KGE scores to determine normalization parameters.

    Args:
        all_users_top_k_full_candidates: Dictionary of user candidates

    Returns:
        Dictionary containing statistics (median, std, min, max, etc.)
    """
    if not all_users_top_k_full_candidates:
        return {}

    all_kge_scores = []

    for candidates in all_users_top_k_full_candidates.values():
        for kge_score, _, _ in candidates:
            all_kge_scores.append(kge_score)

    if not all_kge_scores:
        return {}

    kge_scores_array = np.array(all_kge_scores)

    return {
        "median": float(np.median(kge_scores_array)),
        "mean": float(np.mean(kge_scores_array)),
        "std": float(np.std(kge_scores_array)),
        "min": float(np.min(kge_scores_array)),
        "max": float(np.max(kge_scores_array)),
        "q25": float(np.percentile(kge_scores_array, 25)),
        "q75": float(np.percentile(kge_scores_array, 75)),
        "count": len(all_kge_scores),
    }


def get_recommended_normalization_params(all_users_top_k_full_candidates: Dict[int, List[Tuple[float, float, int]]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Get recommended normalization parameters based on KGE score distribution.

    Args:
        all_users_top_k_full_candidates: Dictionary of user candidates

    Returns:
        Tuple of (median_kge_score, scale_factor_kge)
    """
    stats = calculate_prediction_statistics(all_users_top_k_full_candidates)

    if not stats:
        print("Warning: No KGE scores found for normalization parameter calculation.")
        return None, None

    median_kge = stats["median"]

    # Use IQR as scale factor for robust normalization
    iqr = stats["q75"] - stats["q25"]
    scale_factor = max(iqr, 0.1)  # Minimum scale to avoid division by zero

    print("Recommended normalization parameters:")
    print(f"  Median KGE score: {median_kge:.4f}")
    print(f"  Scale factor (IQR): {scale_factor:.4f}")
    print(f"  KGE score range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    return median_kge, scale_factor


def validate_predictions(df_predictions: pd.DataFrame) -> bool:
    """
    Validate that predictions are in the expected range and format.

    Args:
        df_predictions: DataFrame with predictions

    Returns:
        True if predictions are valid, False otherwise
    """
    if df_predictions.empty:
        print("Warning: Predictions DataFrame is empty.")
        return True  # Empty is technically valid

    required_columns = {"userID", "itemID", "prediction"}
    if not required_columns.issubset(df_predictions.columns):
        print(f"Error: Missing required columns. Expected: {required_columns}")
        return False

    # Check prediction range
    predictions = df_predictions["prediction"]
    if predictions.min() < 0 or predictions.max() > 5:
        print(f"Warning: Predictions outside expected range [0, 5]. Actual range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        return False

    # Check for NaN values
    if predictions.isna().any():
        print("Error: NaN values found in predictions.")
        return False

    print(f"Predictions validation passed. Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    return True


# Legacy function name for backward compatibility
def calculate_predictons(*args, **kwargs):
    """
    Legacy function name for backward compatibility.

    Note: This is a typo in the original function name.
    Use calculate_predictions() for new code.
    """
    print("Warning: Using legacy function name 'calculate_predictons'. Consider using 'calculate_predictions' instead.")
    return calculate_predictions(*args, **kwargs)
