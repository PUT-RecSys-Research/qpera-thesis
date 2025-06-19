from . import data_manipulation as dm
from . import datasets_loader
from . import log_mlflow
import numpy as np
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
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import (
    mae,
    ndcg_at_k,
    rmse,
)
from recommenders.models.tfidf.tfidf_utils import TfidfRecommender


def cbf_experiment_loop(
    TOP_K,
    dataset,
    want_col,
    num_rows,
    ratio,
    seed,
    personalization=False,
    fraction_to_change=0,
    change_rating=False,
    privacy=False,
    hide_type="values_in_column",
    columns_to_hide=None,
    fraction_to_hide=0,
    records_to_hide=None,
):
    TOP_K = TOP_K
    dataset = dataset
    want_col = want_col
    num_rows = num_rows
    ratio = ratio
    seed = seed
    personalization = personalization
    fraction_to_change = fraction_to_change
    change_rating = change_rating
    privacy = privacy
    hide_type = hide_type
    columns_to_hide = columns_to_hide
    fraction_to_hide = fraction_to_hide
    records_to_hide = records_to_hide

    params = {
        "dataset": dataset,
        "want_col": want_col,
        "num_rows": num_rows,
        "ratio": ratio,
        "seed": seed,
    }
    header = {
        "col_user": "userID",
        "col_item": "itemID",
        "col_rating": "rating",
        "col_timestamp": "timestamp",
        "col_title": "title",
        "col_genres": "genres",
        "col_year": "year",
        "col_prediction": "prediction",
    }

    # Load the MovieLens dataset
    data = datasets_loader.loader(dataset, want_col, num_rows, seed)
    data["rating"] = data["rating"].astype(np.float32)

    # Create a TF-IDF model
    recommender = TfidfRecommender(id_col="itemID", tokenization_method="bert")
    # data['genres'] = data['genres'].str.replace('|', ' ', regex=False)

    train, test = python_stratified_split(data, ratio=ratio, col_user=header["col_user"], col_item=header["col_item"], seed=seed)

    # df_clean = train.drop(columns=['userID', 'rating', 'timestamp'])
    # df_clean = df_clean.drop_duplicates(subset=['itemID'])
    # cols_to_clean = ['title','genres']
    # clean_col = 'cleaned_text'
    # df_clean = recommender.clean_dataframe(df_clean, cols_to_clean, clean_col)

    # df_clean = df_clean.reset_index(drop=True)

    # Split the dataset to training and testing dataset
    # train, test = python_stratified_split(
    #     data, ratio=ratio, col_user=header["col_user"], col_item=header["col_item"], seed=seed
    # )

    if privacy:
        train = dm.hide_information_in_dataframe(
            data=train,
            hide_type=hide_type,
            columns_to_hide=columns_to_hide,
            fraction_to_hide=fraction_to_hide,
            records_to_hide=records_to_hide,
            seed=seed,
        )

    if personalization:
        train = dm.change_items_in_dataframe(
            all=data,
            data=train,
            fraction_to_change=fraction_to_change,
            change_rating=change_rating,
            seed=seed,
        )

    df_clean = train.drop(columns=["userID", "rating", "timestamp"])
    df_clean = df_clean.drop_duplicates(subset=["itemID"])
    cols_to_clean = ["title", "genres"]
    clean_col = "cleaned_text"
    df_clean = recommender.clean_dataframe(df_clean, cols_to_clean, clean_col)

    df_clean = df_clean.reset_index(drop=True)
    # train = recommender.clean_dataframe(train, cols_to_clean, clean_col)

    # Tokenize the text
    tf, vectors_tokenized = recommender.tokenize_text(df_clean, text_col="cleaned_text")

    # Fit the model
    recommender.fit(tf, vectors_tokenized)
    tokens = recommender.get_tokens()
    print(list(tokens.keys())[:10])

    top_k_items = recommender.recommend_top_k_items(df_clean, k=5)
    merged_df = data.merge(top_k_items, on="itemID", how="inner")
    merged_df["prediction"] = merged_df["rating"] * merged_df["rec_score"]
    top_k = merged_df[["userID", "rec_itemID", "prediction"]].copy()
    top_k.rename(columns={"rec_itemID": "itemID"}, inplace=True)

    filtered_top_k = top_k.merge(train, on=["userID", "itemID"], how="left", indicator=True)
    filtered_top_k = filtered_top_k[filtered_top_k["_merge"] == "left_only"].drop(columns=["_merge"])
    filtered_top_k = filtered_top_k[["userID", "itemID", "prediction"]]

    idx = filtered_top_k.groupby("userID")["prediction"].idxmax()
    top = filtered_top_k.loc[idx]

    # Metrics
    eval_precision_at_k = precision_at_k(
        test,
        top_k,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K,
    )
    eval_recall_at_k = recall_at_k(
        test,
        top_k,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K,
    )
    eval_ndcg = ndcg_at_k(
        test,
        top,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        relevancy_method="top_k",
        k=1,
    )
    eval_precision = precision_at_k(
        test,
        top_k,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        k=1,
    )
    eval_recall = recall_at_k(
        test,
        top_k,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        k=1,
    )
    eval_mae = mae(test, top_k)
    eval_rmse = rmse(test, top_k)

    # eval_novelty = novelty(train, top)
    # eval_historical_item_novelty = historical_item_novelty(train, top)
    # eval_user_item_serendipity = user_item_serendipity(train, top)
    # eval_user_serendipity = user_serendipity(train, top)
    # eval_serendipity = serendipity(train, top)
    # eval_catalog_coverage = catalog_coverage(train, top)
    # eval_distributional_coverage = distributional_coverage(train, top)

    # eval_f1 = f1(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    eval_mrr = mrr(
        test,
        top_k,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
    )
    # eval_accuracy = accuracy(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    eval_user_coverage = user_coverage(
        test,
        top,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
    )
    eval_item_coverage = item_coverage(
        test,
        top,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
    )

    eval_intra_list_similarity = intra_list_similarity_score(data, top_k, feature_cols=["genres"])
    eval_intra_list_dissimilarity = intra_list_dissimilarity(data, top_k, feature_cols=["genres"])
    eval_personalization = personalization_score(train, top)

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
        "User coverage:\t%f" % eval_user_coverage,
        "Item coverage:\t%f" % eval_item_coverage,
        # "Catalog coverage:\t%f" % eval_catalog_coverage,
        # "Distributional coverage:\t%f" % eval_distributional_coverage,
        "Personalization:\t%f" % eval_personalization,
        "Intra-list similarity:\t%f" % eval_intra_list_similarity,
        "Intra-list dissimilarity:\t%f" % eval_intra_list_dissimilarity,
        sep="\n",
    )

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

    log_mlflow.log_mlflow(
        dataset,
        top_k,
        metrics,
        num_rows,
        seed,
        recommender,
        "CBF",
        params,
        train,
        data,
        tf,
        vectors_tokenized,
        privacy=privacy,
        fraction_to_hide=fraction_to_hide,
        personalization=personalization,
        fraction_to_change=fraction_to_change,
    )
