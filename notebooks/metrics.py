import numpy as np
import pandas as pd
from recommenders.evaluation.python_evaluation import merge_rating_true_pred, merge_ranking_true_pred, precision_at_k, recall_at_k
from sklearn.metrics import (
    f1_score,
    label_ranking_average_precision_score,
    accuracy_score
)

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RELEVANCE_COL,
    DEFAULT_SIMILARITY_COL,
    DEFAULT_ITEM_FEATURES_COL,
    DEFAULT_ITEM_SIM_MEASURE,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
)

def f1(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=1,
    threshold=DEFAULT_THRESHOLD,
):
    precision = precision_at_k(rating_true, rating_pred, col_user, col_item, col_prediction, relevancy_method, k, threshold)
    recall = recall_at_k(rating_true, rating_pred, col_user, col_item, col_prediction, relevancy_method, k, threshold)
    return (2*precision*recall)/(precision+recall)

def mrr(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return label_ranking_average_precision_score(y_true, y_pred)

def accuracy(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
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
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Calculates the user coverage metric, which represents the percentage of users for whom meaningful recommendations can be generated.
    A recommendation is considered meaningful if the predicted rating deviates from the actual rating by no more than the given threshold.

    :param rating_true: DataFrame containing the actual user ratings.
    :param rating_pred: DataFrame containing the predicted user ratings.
    :param col_user: Name of the column containing user identifiers.
    :param col_item: Name of the column containing item identifiers.
    :param col_rating: Name of the column containing actual ratings.
    :param col_prediction: Name of the column containing predicted ratings.
    :param threshold: Maximum allowed deviation between actual and predicted ratings for a recommendation to be considered meaningful.
    :return: Percentage of users for whom meaningful recommendations can be generated.
    """
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )

    user_errors = rating_true.merge(rating_pred, on=[col_user, col_item])
    user_errors["error"] = abs(user_errors[col_rating] - user_errors[col_prediction])
    meaningful_recommendations = user_errors[user_errors["error"] <= threshold]
    meaningful_users_table = meaningful_recommendations[col_user].unique()
    meaningful_users = len(meaningful_users_table)
    total_users_table = rating_true[col_user].unique()
    total_users = len(total_users_table)
    
    coverage = meaningful_users / total_users if total_users > 0 else 0
    return coverage

def item_coverage(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    threshold=DEFAULT_THRESHOLD
):
    """
    Calculates the item coverage metric, which represents the percentage of items for which meaningful recommendations can be generated.
    A recommendation is considered meaningful if the predicted rating deviates from the actual rating by no more than the given threshold.

    :param rating_true: DataFrame containing the actual user ratings.
    :param rating_pred: DataFrame containing the predicted user ratings.
    :param col_user: Name of the column containing user identifiers.
    :param col_item: Name of the column containing item identifiers.
    :param col_rating: Name of the column containing actual ratings.
    :param col_prediction: Name of the column containing predicted ratings.
    :param threshold: Maximum allowed deviation between actual and predicted ratings for a recommendation to be considered meaningful.
    :return: Percentage of items for which meaningful recommendations can be generated.
    """
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )

    item_errors = rating_true.merge(rating_pred, on=[col_user, col_item])
    item_errors["error"] = abs(item_errors[col_rating] - item_errors[col_prediction])
    meaningful_recommendations = item_errors[item_errors["error"] <= threshold]
    meaningful_items_table = meaningful_recommendations[col_item].unique()
    meaningful_items = len(meaningful_items_table)
    total_items_table = rating_true[col_item].unique()
    total_items = len(total_items_table)
    
    coverage = meaningful_items / total_items if total_items > 0 else 0
    return coverage
