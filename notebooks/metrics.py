import numpy as np
import pandas as pd
from recommenders.evaluation.python_evaluation import merge_rating_true_pred, merge_ranking_true_pred
from sklearn.metrics import (
    label_ranking_average_precision_score,
    accuracy_score
)
from sklearn.metrics._classification import _prf_divide

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

def precision_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    k=10
):
    """
    Oblicza precyzję na poziomie k dla rekomendacji.
    
    Args:
        rating_true (pd.DataFrame): Zbiór rzeczywistych ocen zawierający kolumny [user, item, rating].
        rating_pred (pd.DataFrame): Zbiór przewidywanych ocen zawierający kolumny [user, item, prediction].
        col_user (str): Nazwa kolumny z identyfikatorem użytkownika.
        col_item (str): Nazwa kolumny z identyfikatorem elementu.
        col_rating (str): Nazwa kolumny z rzeczywistymi ocenami.
        col_prediction (str): Nazwa kolumny z przewidywanymi ocenami.
        k (int): Liczba rekomendacji do uwzględnienia w ocenie.
    
    Returns:
        float: Wartość precyzji na poziomie k.
    """
    users = rating_true[col_user].unique()
    precisions = []
    
    for user in users:
        # Pobranie rzeczywistych pozytywnie ocenionych elementów
        true_items = set(rating_true[(rating_true[col_user] == user) & (rating_true[col_rating] > 0)][col_item])
        
        # Pobranie k najlepszych rekomendacji
        user_pred = rating_pred[rating_pred[col_user] == user].nlargest(k, col_prediction)
        top_k_items = set(user_pred[col_item])
        
        # Jeśli użytkownik nie ma rekomendacji, pomijamy go
        if not top_k_items:
            continue
        
        # Obliczenie precyzji
        precision = len(true_items & top_k_items)
        precision = 1 if precision > 0 else 0
        precisions.append(precision)
    
    return sum(precisions) / len(precisions) if precisions else 0.0

def recall_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    k=10
):
    """
    Oblicza recall na poziomie k dla rekomendacji.
    
    Args:
        rating_true (pd.DataFrame): Zbiór rzeczywistych ocen zawierający kolumny [user, item, rating].
        rating_pred (pd.DataFrame): Zbiór przewidywanych ocen zawierający kolumny [user, item, prediction].
        col_user (str): Nazwa kolumny z identyfikatorem użytkownika.
        col_item (str): Nazwa kolumny z identyfikatorem elementu.
        col_rating (str): Nazwa kolumny z rzeczywistymi ocenami.
        col_prediction (str): Nazwa kolumny z przewidywanymi ocenami.
        k (int): Liczba rekomendacji do uwzględnienia w ocenie.
    
    Returns:
        float: Wartość recall na poziomie k.
    """
    users = rating_true[col_user].unique()
    recalls = []
    
    for user in users:
        # Pobranie rzeczywistych pozytywnie ocenionych elementów
        true_items = set(rating_true[(rating_true[col_user] == user) & (rating_true[col_rating] > 0)][col_item])
        
        # Pobranie k najlepszych rekomendacji
        user_pred = rating_pred[rating_pred[col_user] == user].nlargest(k, col_prediction)
        top_k_items = set(user_pred[col_item])
        
        # Jeśli użytkownik nie ma prawdziwych pozytywnych ocen, pomijamy go
        if not true_items:
            continue
        
        # Obliczenie recall
        recall = len(true_items & top_k_items) / len(true_items)
        recalls.append(recall)
    
    return sum(recalls) / len(recalls) if recalls else 0.0


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

# ----------------------------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------------------------

# def intra_list_similarity_score(
#     item_features: pd.DataFrame,
#     rating_pred: pd.DataFrame,
#     col_user: str = 'userID',
#     col_item: str = 'itemID',
#     feature_cols: list = None
# ) -> float:
#     """
#     Oblicza średnie podobieństwo między itemami w przewidywanej liście rekomendacji dla każdego użytkownika.

#     :param item_features: DataFrame z kolumnami [itemID, <feature_cols>]
#     :param rating_pred: DataFrame z kolumnami [userID, itemID, prediction]
#     :param col_user: Nazwa kolumny z identyfikatorem użytkownika
#     :param col_item: Nazwa kolumny z identyfikatorem itemu
#     :param feature_cols: Lista kolumn zawierających cechy itemów
#     :return: Średnia intra-list similarity
#     """
    
#     if feature_cols is None:
#         raise ValueError("Musisz podać feature_cols — listę kolumn cech itemów")

#     # Debugging: sprawdzenie struktury danych
#     print(f"rating_pred columns: {rating_pred.columns}")
#     print(f"item_features columns: {item_features.columns}")

#     # Scalanie rating_pred z cechami z item_features
#     rating_pred_with_features = pd.merge(rating_pred, item_features[[col_item] + feature_cols], on=col_item, how='left')
    
#     # Debugging: Sprawdzenie, czy po scaleniu mamy odpowiednie dane
#     print(f"Zmergowane dane:\n{rating_pred_with_features.head()}")

#     # Sprawdzenie brakujących wartości
#     print(f"Brakujące wartości w rating_pred_with_features:\n{rating_pred_with_features.isna().sum()}")

#     # Obsługa kolumn tekstowych (np. 'genres')
#     feature_df = rating_pred_with_features[[col_item] + feature_cols].copy()
#     for col in feature_cols:
#         if feature_df[col].dtype == 'object':
#             if feature_df[col].str.contains('|', regex=False).any():
#                 dummies = feature_df[col].str.get_dummies(sep='|')
#             else:
#                 dummies = pd.get_dummies(feature_df[col], prefix=col)
#             feature_df = pd.concat([feature_df.drop(columns=[col]), dummies], axis=1)

#     # Usunięcie NaN, duplikatów i ustawienie indeksu na itemID
#     feature_df = feature_df.dropna().drop_duplicates(subset=col_item)
#     feature_df = feature_df.reset_index(drop=True)
#     print(f"Feature DataFrame po przetworzeniu:\n{feature_df.head()}")

#     # Grupowanie rekomendacji po użytkowniku
#     predicted_lists = rating_pred_with_features.groupby(col_user)[col_item].apply(list)
#     predicted_lists = predicted_lists.reset_index(drop=True)
#     print(f"Predicted lists:\n{predicted_lists.head()}")

#     # Obliczenie intra-list similarity
#     return intra_list_similarity(predicted=rating_pred_with_features, feature_df=feature_df)

def intra_list_similarity_score(
    item_features: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str = 'userID',
    col_item: str = 'itemID',
    feature_cols: list = None
) -> float:
    """
    Oblicza średnie podobieństwo między itemami w przewidywanej liście rekomendacji dla każdego użytkownika.

    :param item_features: DataFrame z kolumnami [itemID, <feature_cols>]
    :param rating_pred: DataFrame z kolumnami [userID, itemID, prediction]
    :param col_user: Nazwa kolumny z identyfikatorem użytkownika
    :param col_item: Nazwa kolumny z identyfikatorem itemu
    :param feature_cols: Lista kolumn zawierających cechy itemów
    :return: Średnia intra-list similarity
    """

    if feature_cols is None:
        raise ValueError("Musisz podać feature_cols — listę kolumn cech itemów")

    # Połącz dane predykcji z cechami itemów
    rating_pred_with_features = pd.merge(
        rating_pred,
        item_features[[col_item] + feature_cols],
        on=col_item,
        how='left'
    )

    # Przekształć cechy (np. tekstowe) na wektory (one-hot/dummy)
    feature_df = rating_pred_with_features[[col_item] + feature_cols].copy()
    for col in feature_cols:
        if feature_df[col].dtype == 'object':
            if feature_df[col].str.contains('|', regex=False).any():
                dummies = feature_df[col].str.get_dummies(sep='|')
            else:
                dummies = pd.get_dummies(feature_df[col], prefix=col)
            feature_df = pd.concat([feature_df.drop(columns=[col]), dummies], axis=1)

    # Usuń NaN i duplikaty
    feature_df = feature_df.dropna().drop_duplicates(subset=col_item)

    # Przypisz nowe ID (zaczynając od 1) – synchronizacja
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(feature_df[col_item].unique(), start=1)}
    feature_df[col_item] = feature_df[col_item].map(item_id_map)
    rating_pred_with_features[col_item] = rating_pred_with_features[col_item].map(item_id_map)

    # Ustaw index i uzupełnij brakujące wartości zerami
    feature_df = feature_df.set_index(col_item)
    feature_df = feature_df.fillna(0)

    # Grupuj rekomendacje według użytkownika
    predicted_lists = rating_pred_with_features.groupby(col_user)[col_item].apply(list)

    # Oblicz intra-list similarity ręcznie
    ils_values = []

    for user_list in predicted_lists:
        if len(user_list) < 2:
            continue  # pomiń, jeśli lista ma mniej niż 2 elementy

        try:
            item_vectors = feature_df.loc[user_list].values
        except KeyError as e:
            continue  # pomiń, jeśli nie można znaleźć itemów

        # Oblicz macierz podobieństw
        sim_matrix = cosine_similarity(item_vectors)

        # Bierzemy tylko górny trójkąt bez diagonali (podobieństwa par)
        n = len(user_list)
        upper_triangle_indices = np.triu_indices(n, k=1)
        sims = sim_matrix[upper_triangle_indices]

        if len(sims) > 0:
            ils_values.append(np.mean(sims))

    # Zwróć średnie podobieństwo z wszystkich użytkowników
    return np.mean(ils_values) if ils_values else 0.0



def intra_list_dissimilarity(    
    item_features,
    rating_pred,
    col_user = 'userID',
    col_item = 'itemID',
    feature_cols = None
) -> float:
    return 1 - intra_list_similarity_score(item_features=item_features, rating_pred=rating_pred, col_user=col_user, col_item=col_item, feature_cols=feature_cols)

def personalization_score(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
) -> float:
    """
    Interfejs do funkcji personalization.
    Mierzy stopień personalizacji rekomendacji między użytkownikami.
    
    :param rating_true: Prawdziwe oceny użytkowników (nieużywane w tej funkcji)
    :param rating_pred: Przewidywane oceny użytkowników
    :param col_user: Kolumna identyfikująca użytkownika
    :param col_item: Kolumna identyfikująca przedmiot
    :param col_rating: Kolumna z rzeczywistą oceną (nieużywana w tej funkcji)
    :param col_prediction: Kolumna z przewidywaną oceną (nieużywana w tej funkcji)
    :return: Wartość personalizacji (im wyższa, tym lepsza personalizacja)
    """
    predicted_lists = rating_pred.groupby(col_user)[col_item].apply(list).tolist()
    return personalization(predicted=predicted_lists)