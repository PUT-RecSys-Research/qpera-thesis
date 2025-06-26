from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from recommenders.evaluation.python_evaluation import (
    merge_rating_true_pred,
)
from recommenders.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RATING_COL,
    DEFAULT_THRESHOLD,
    DEFAULT_USER_COL,
)
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


def precision_at_k(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
    k: int = 10,
) -> float:
    """
    Calculate precision at k for recommendations.

    Args:
        rating_true: DataFrame with actual ratings containing [user, item, rating] columns
        rating_pred: DataFrame with predicted ratings containing [user, item, prediction] columns
        col_user: Name of user identifier column
        col_item: Name of item identifier column
        col_rating: Name of actual ratings column
        col_prediction: Name of predicted ratings column
        k: Number of recommendations to consider in evaluation

    Returns:
        Precision at k value
    """
    users = rating_true[col_user].unique()
    precisions = []

    for user in users:
        # Get actual positively rated items
        true_items = set(rating_true[(rating_true[col_user] == user) & (rating_true[col_rating] > 0)][col_item])

        # Get top k recommendations
        user_pred = rating_pred[rating_pred[col_user] == user].nlargest(k, col_prediction)
        top_k_items = set(user_pred[col_item])

        # Skip users with no recommendations
        if not top_k_items:
            continue

        # Calculate precision (binary: 1 if any relevant item in top-k, 0 otherwise)
        precision = 1 if len(true_items & top_k_items) > 0 else 0
        precisions.append(precision)

    return sum(precisions) / len(precisions) if precisions else 0.0


def recall_at_k(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
    k: int = 10,
) -> float:
    """
    Calculate recall at k for recommendations.

    Args:
        rating_true: DataFrame with actual ratings containing [user, item, rating] columns
        rating_pred: DataFrame with predicted ratings containing [user, item, prediction] columns
        col_user: Name of user identifier column
        col_item: Name of item identifier column
        col_rating: Name of actual ratings column
        col_prediction: Name of predicted ratings column
        k: Number of recommendations to consider in evaluation

    Returns:
        Recall at k value
    """
    users = rating_true[col_user].unique()
    recalls = []

    for user in users:
        # Get actual positively rated items
        true_items = set(rating_true[(rating_true[col_user] == user) & (rating_true[col_rating] > 0)][col_item])

        # Get top k recommendations
        user_pred = rating_pred[rating_pred[col_user] == user].nlargest(k, col_prediction)
        top_k_items = set(user_pred[col_item])

        # Skip users with no true positive items
        if not true_items:
            continue

        # Calculate recall
        recall = len(true_items & top_k_items) / len(true_items)
        recalls.append(recall)

    return sum(recalls) / len(recalls) if recalls else 0.0


def f1_score(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
    k: int = 1,
) -> float:
    """Calculate F1 score from precision and recall at k."""
    precision = precision_at_k(rating_true, rating_pred, col_user, col_item, col_rating, col_prediction, k)
    recall = recall_at_k(rating_true, rating_pred, col_user, col_item, col_rating, col_prediction, k)

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def mrr(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
    k: int = 1,
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        rating_true: DataFrame with actual ratings
        rating_pred: DataFrame with predicted ratings
        col_user: Name of user identifier column
        col_item: Name of item identifier column
        col_rating: Name of actual ratings column
        col_prediction: Name of predicted ratings column
        k: Number of top recommendations to consider

    Returns:
        Mean Reciprocal Rank value
    """
    # Get top k predictions sorted by user and prediction score
    top_k_pred = rating_pred.sort_values(by=[col_user, col_prediction], ascending=[True, False]).groupby(col_user).head(k)

    # Create mapping of users to their relevant items
    true_items = rating_true.groupby(col_user)[col_item].apply(set).to_dict()

    mrr_total = 0.0
    user_count = 0

    for user, user_df in top_k_pred.groupby(col_user):
        predicted_items = user_df[col_item].tolist()
        relevant_items = true_items.get(user, set())

        # Find first relevant item and calculate reciprocal rank
        for rank, item in enumerate(predicted_items, start=1):
            if item in relevant_items:
                mrr_total += 1.0 / rank
                break

        user_count += 1

    return mrr_total / user_count if user_count > 0 else 0.0


def accuracy(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
) -> float:
    """Calculate accuracy by comparing rounded predictions with actual ratings."""
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    y_pred = np.round(y_pred)
    return accuracy_score(y_true, y_pred)


def user_coverage(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """
    Calculate user coverage metric.

    Represents the percentage of users for whom meaningful recommendations can be generated.
    A recommendation is considered meaningful if the predicted rating deviates from the
    actual rating by no more than the given threshold.

    Args:
        rating_true: DataFrame containing actual user ratings
        rating_pred: DataFrame containing predicted user ratings
        col_user: Name of user identifier column
        col_item: Name of item identifier column
        col_rating: Name of actual ratings column
        col_prediction: Name of predicted ratings column
        threshold: Maximum allowed deviation between actual and predicted ratings

    Returns:
        Percentage of users for whom meaningful recommendations can be generated
    """
    # Merge actual and predicted ratings
    user_errors = rating_true.merge(rating_pred, on=[col_user, col_item])
    user_errors["error"] = abs(user_errors[col_rating] - user_errors[col_prediction])

    # Find meaningful recommendations within threshold
    meaningful_recommendations = user_errors[user_errors["error"] <= threshold]
    meaningful_users = len(meaningful_recommendations[col_user].unique())
    total_users = len(rating_true[col_user].unique())

    return meaningful_users / total_users if total_users > 0 else 0.0


def item_coverage(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """
    Calculate item coverage metric.

    Represents the percentage of items for which meaningful recommendations can be generated.
    A recommendation is considered meaningful if the predicted rating deviates from the
    actual rating by no more than the given threshold.

    Args:
        rating_true: DataFrame containing actual user ratings
        rating_pred: DataFrame containing predicted user ratings
        col_user: Name of user identifier column
        col_item: Name of item identifier column
        col_rating: Name of actual ratings column
        col_prediction: Name of predicted ratings column
        threshold: Maximum allowed deviation between actual and predicted ratings

    Returns:
        Percentage of items for which meaningful recommendations can be generated
    """
    # Merge actual and predicted ratings
    item_errors = rating_true.merge(rating_pred, on=[col_user, col_item])
    item_errors["error"] = abs(item_errors[col_rating] - item_errors[col_prediction])

    # Find meaningful recommendations within threshold
    meaningful_recommendations = item_errors[item_errors["error"] <= threshold]
    meaningful_items = len(meaningful_recommendations[col_item].unique())
    total_items = len(rating_true[col_item].unique())

    return meaningful_items / total_items if total_items > 0 else 0.0


def _single_list_similarity(predicted: list, feature_df: pd.DataFrame, u: int) -> float:
    """
    Compute intra-list similarity for a single list of recommendations.

    Args:
        predicted: Ordered list of predictions (e.g., ['X', 'Y', 'Z'])
        feature_df: DataFrame with one-hot encoded or latent features, indexed by item ID
        u: User index for error reporting

    Returns:
        Intra-list similarity for a single recommendation list
    """
    if not predicted:
        raise Exception(f"Predicted list is empty, index: {u}")

    # Get features for all recommended items
    recs_content = feature_df.loc[predicted]
    recs_content = recs_content.dropna()
    recs_content = sp.csr_matrix(recs_content.values)

    # Calculate similarity scores for all items in list
    similarity = cosine_similarity(X=recs_content, dense_output=False)

    # Get indices for upper right triangle without diagonal
    upper_right = np.triu_indices(similarity.shape[0], k=1)

    # Calculate average similarity score of all recommended items in list
    return np.mean(similarity[upper_right])


def intra_list_similarity(predicted: List[list], feature_df: pd.DataFrame) -> float:
    """
    Compute average intra-list similarity of all recommendations.

    This metric measures diversity of recommended item lists. Lower values indicate
    more diverse recommendations.

    Args:
        predicted: List of lists with ordered predictions (e.g., [['X', 'Y', 'Z'], ['A', 'B', 'C']])
        feature_df: DataFrame with one-hot encoded or latent features, indexed by item ID

    Returns:
        Average intra-list similarity for all recommendations
    """
    feature_df = feature_df.fillna(0)
    users = range(len(predicted))
    ils_scores = [_single_list_similarity(predicted[u], feature_df, u) for u in users]
    return np.mean(ils_scores)


def personalization(predicted: List[list]) -> float:
    """
    Measure recommendation similarity across users.

    A high score indicates good personalization (users' recommendation lists are different).
    A low score indicates poor personalization (users' recommendation lists are very similar).

    Args:
        predicted: List of lists with ordered predictions (e.g., [['X', 'Y', 'Z'], ['A', 'B', 'C']])

    Returns:
        Personalization score for all recommendations
    """

    def make_rec_matrix(predicted: List[list]) -> sp.csr_matrix:
        """Convert recommendation lists to sparse binary matrix."""
        df = pd.DataFrame(data=predicted).reset_index().melt(id_vars="index", value_name="item")
        df = df[["index", "item"]].pivot(index="index", columns="item", values="item")
        df = pd.notna(df) * 1
        return sp.csr_matrix(df.values)

    # Create matrix for recommendations
    predicted = np.array(predicted)
    rec_matrix_sparse = make_rec_matrix(predicted)

    # Calculate similarity for every user's recommendation list
    similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)

    # Calculate average similarity
    dim = similarity.shape[0]
    avg_similarity = (similarity.sum() - dim) / (dim * (dim - 1))

    # Return personalization (1 - similarity)
    return 1 - avg_similarity


def intra_list_similarity_score(
    item_features: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = "userID",
    col_item: str = "itemID",
    feature_cols: List[str] = None,
) -> float:
    """
    Calculate average similarity between items in predicted recommendation lists for each user.

    Args:
        item_features: DataFrame with columns [itemID, <feature_cols>]
        rating_pred: DataFrame with columns [userID, itemID, prediction]
        col_user: Name of user identifier column
        col_item: Name of item identifier column
        feature_cols: List of columns containing item features

    Returns:
        Average intra-list similarity score
    """
    if feature_cols is None:
        raise ValueError("Must provide feature_cols - list of item feature columns")

    # Merge prediction data with item features
    rating_pred_with_features = pd.merge(rating_pred, item_features[[col_item] + feature_cols], on=col_item, how="left")

    # Convert features (e.g., text) to vectors (one-hot/dummy encoding)
    feature_df = rating_pred_with_features[[col_item] + feature_cols].copy()

    for col in feature_cols:
        if feature_df[col].dtype == "object":
            # Handle pipe-separated values or regular categorical data
            if feature_df[col].str.contains("|", regex=False).any():
                dummies = feature_df[col].str.get_dummies(sep="|")
            else:
                dummies = pd.get_dummies(feature_df[col], prefix=col)
            feature_df = pd.concat([feature_df.drop(columns=[col]), dummies], axis=1)

    # Remove NaN and duplicates
    feature_df = feature_df.dropna().drop_duplicates(subset=col_item)

    # Create item ID mapping for consistency
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(feature_df[col_item].unique(), start=1)}
    feature_df[col_item] = feature_df[col_item].map(item_id_map)
    rating_pred_with_features[col_item] = rating_pred_with_features[col_item].map(item_id_map)

    # Set index and fill missing values with zeros
    feature_df = feature_df.set_index(col_item).fillna(0)

    # Group recommendations by user
    predicted_lists = rating_pred_with_features.groupby(col_user)[col_item].apply(list)

    # Calculate intra-list similarity manually
    ils_values = []

    for user_list in predicted_lists:
        if len(user_list) < 2:
            continue  # Skip if list has fewer than 2 items

        try:
            item_vectors = feature_df.loc[user_list].values
        except KeyError:
            continue  # Skip if items not found

        # Calculate similarity matrix
        sim_matrix = cosine_similarity(item_vectors)

        # Get upper triangle without diagonal (pairwise similarities)
        n = len(user_list)
        upper_triangle_indices = np.triu_indices(n, k=1)
        similarities = sim_matrix[upper_triangle_indices]

        if len(similarities) > 0:
            ils_values.append(np.mean(similarities))

    # Return average similarity across all users
    return np.mean(ils_values) if ils_values else 0.0


def intra_list_dissimilarity(
    item_features: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = "userID",
    col_item: str = "itemID",
    feature_cols: List[str] = None,
) -> float:
    """
    Calculate intra-list dissimilarity (1 - similarity).

    Higher values indicate more diverse recommendations.
    """
    return 1 - intra_list_similarity_score(
        item_features=item_features,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        feature_cols=feature_cols,
    )


def personalization_score(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
) -> float:
    """
    Interface to personalization function.

    Measures the degree of personalization in recommendations across users.

    Args:
        rating_true: Actual user ratings (unused in this function)
        rating_pred: Predicted user ratings
        col_user: Column identifying the user
        col_item: Column identifying the item
        col_rating: Column with actual rating (unused in this function)
        col_prediction: Column with predicted rating (unused in this function)

    Returns:
        Personalization value (higher is better personalization)
    """
    predicted_lists = rating_pred.groupby(col_user)[col_item].apply(list).tolist()
    return personalization(predicted=predicted_lists)
