import numpy as np
import pandas as pd
import datasets_loader

# from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k, mae, rmse, novelty, historical_item_novelty, user_item_serendipity, user_serendipity, serendipity, catalog_coverage, distributional_coverage
from recommenders.models.sar import SAR

import mlflow
from mlflow.models import infer_signature


def cf_experiment_loop(TOP_K, dataset, want_col, num_rows, ratio, seed):
    TOP_K = TOP_K
    dataset = dataset
    want_col = want_col
    num_rows = num_rows
    ratio = ratio
    seed = seed

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
        "col_prediction": "prediction",
    }

    # Load the MovieLens dataset
    data = datasets_loader.loader(dataset, want_col, num_rows)

    # Split the dataset to training and testing dataset
    train, test = python_stratified_split(
        data, ratio=ratio, col_user=header["col_user"], col_item=header["col_item"], seed=seed
    )
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    train['timestamp'] = train['timestamp'].astype('int64') // 10**9

    # Create an SAR model
    model = SAR(
        similarity_type="jaccard", 
        time_decay_coefficient=30, 
        time_now=None, 
        timedecay_formula=True, 
        **header
    )
    model.fit(train)

    top_k = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)
    top = model.recommend_k_items(test, top_k=1, remove_seen=True)

    top_k_with_titles = top_k.join(
        data[["itemID", "title"]].drop_duplicates().set_index("itemID"),
        on="itemID",
        how="inner",
    ).sort_values(by=["userID", "prediction"], ascending=False)
    top_with_titles = top.join(
        data[["itemID", "title"]].drop_duplicates().set_index("itemID"),
        on="itemID",
        how="inner",
    ).sort_values(by=["userID", "prediction"], ascending=False)

    # Metrics
    args = [test, top_k]
    kwargs = dict(
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        relevancy_method="top_k",
        k=TOP_K,
    )
    eval_map = map_at_k(*args, **kwargs)
    eval_ndcg_at_k = ndcg_at_k(*args, **kwargs)
    eval_precision_at_k = precision_at_k(*args, **kwargs)
    eval_recall_at_k = recall_at_k(*args, **kwargs)

    eval_mae = mae(test, top_k)
    eval_rmse = rmse(test, top_k)

    args1 = [test, top]
    kwargs1 = dict(
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        relevancy_method="top_k",
        k=1,
    )
    eval_ndcg = ndcg_at_k(*args1, **kwargs1)
    eval_precision = precision_at_k(*args1, **kwargs1)
    eval_recall = recall_at_k(*args1, **kwargs1)

    eval_novelty = novelty(train, top)
    eval_historical_item_novelty = historical_item_novelty(train, top)
    eval_user_item_serendipity = user_item_serendipity(train, top)
    eval_user_serendipity = user_serendipity(train, top)
    eval_serendipity = serendipity(train, top)
    eval_catalog_coverage = catalog_coverage(train, top)
    eval_distributional_coverage = distributional_coverage(train, top)

    print("Precision:\t%f" % eval_precision,
      "Precision@K:\t%f" % eval_precision_at_k,
      "Recall:\t%f" % eval_recall,
      "Recall@K:\t%f" % eval_recall_at_k,
      "MAE:\t%f" % eval_mae,
      "RMSE:\t%f" % eval_rmse,
      "NDCG:\t%f" % eval_ndcg,
      "Novelty:\t%f" % eval_novelty,
      "Serendipity:\t%f" % eval_serendipity,
      "Catalog coverage:\t%f" % eval_catalog_coverage,
      "Distributional coverage:\t%f" % eval_distributional_coverage,
      sep='\n')

    # mlflow
    metrics = {
        "precision_at_K": eval_precision,
        "recall_at_K": eval_recall,
        "NDCG_at_K": eval_ndcg,
        "RMSE": eval_rmse,
        "MAE": eval_mae,
        "novelty": eval_novelty,
        "serendipity": eval_serendipity,
        "catalog_coverage": eval_catalog_coverage,
        "distributional_coverage": eval_distributional_coverage
    }
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Colaborative Filtering v2")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metrics(metrics)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Metrics Info", "CF model for movielens dataset")

        # Infer the model signature
        signature = infer_signature(train, model.fit(train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="CF-model",
            signature=signature,
            input_example=train,
            registered_model_name="CF-model test",
        )

# if __name__ == "__main__":
#     cf_experiment_loop(TOP_K=10, dataset='movielens',
#                         want_col=["userID", "itemID", "rating", "timestamp", 'title', 'genres'],
#                         num_rows=10000,
#                         ratio=0.75,
#                         seed=42)