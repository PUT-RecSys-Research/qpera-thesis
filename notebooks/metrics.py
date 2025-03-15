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

# ------------------------------------------------------------------------------------------------------------------------------------------------
# The code below was copied from recmetrics
# https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py

import random
from itertools import product
from math import sqrt
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import warnings

def _single_list_similarity(predicted: list, feature_df: pd.DataFrame, u: int) -> float:
    """
    Computes the intra-list similarity for a single list of recommendations.
    Parameters
    ----------
    predicted : a list
        Ordered predictions
        Example: ['X', 'Y', 'Z']
    feature_df: dataframe
        A dataframe with one hot encoded or latent features.
        The dataframe should be indexed by the id used in the recommendations.
    Returns:
    -------
    ils_single_user: float
        The intra-list similarity for a single list of recommendations.
    """
    # exception predicted list empty
    if not(predicted):
        raise Exception('Predicted list is empty, index: {0}'.format(u))

    #get features for all recommended items
    recs_content = feature_df.loc[predicted]
    recs_content = recs_content.dropna()
    recs_content = sp.csr_matrix(recs_content.values)

    #calculate similarity scores for all items in list
    similarity = cosine_similarity(X=recs_content, dense_output=False)

    #get indicies for upper right triangle w/o diagonal
    upper_right = np.triu_indices(similarity.shape[0], k=1)

    #calculate average similarity score of all recommended items in list
    ils_single_user = np.mean(similarity[upper_right])
    return ils_single_user

def intra_list_similarity(predicted: List[list], feature_df: pd.DataFrame) -> float:
    """
    Computes the average intra-list similarity of all recommendations.
    This metric can be used to measure diversity of the list of recommended items.
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        Example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    feature_df: dataframe
        A dataframe with one hot encoded or latent features.
        The dataframe should be indexed by the id used in the recommendations.
    Returns:
    -------
        The average intra-list similarity for recommendations.
    """
    feature_df = feature_df.fillna(0)
    Users = range(len(predicted))
    ils = [_single_list_similarity(predicted[u], feature_df, u) for u in Users]
    return np.mean(ils)

def personalization(predicted: List[list]) -> float:
    """
    Personalization measures recommendation similarity across users.
    A high score indicates good personalization (user's lists of recommendations are different).
    A low score indicates poor personalization (user's lists of recommendations are very similar).
    A model is "personalizing" well if the set of recommendations for each user is different.
    Parameters:
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        The personalization score for all recommendations.
    """

    def make_rec_matrix(predicted: List[list]) -> sp.csr_matrix:
        df = pd.DataFrame(data=predicted).reset_index().melt(
            id_vars='index', value_name='item',
        )
        df = df[['index', 'item']].pivot(index='index', columns='item', values='item')
        df = pd.notna(df)*1
        rec_matrix = sp.csr_matrix(df.values)
        return rec_matrix

    #create matrix for recommendations
    predicted = np.array(predicted)
    rec_matrix_sparse = make_rec_matrix(predicted)

    #calculate similarity for every user's recommendation list
    similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)

    #calculate average similarity
    dim = similarity.shape[0]
    personalization = (similarity.sum() - dim) / (dim * (dim - 1))
    return 1-personalization