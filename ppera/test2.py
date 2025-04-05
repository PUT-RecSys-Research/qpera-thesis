import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import data_manipulation
import datasets_loader

# from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k, mae, rmse, novelty, historical_item_novelty, user_item_serendipity, user_serendipity, serendipity, catalog_coverage, distributional_coverage
from recommenders.models.sar import SAR

import mlflow
from mlflow.models import infer_signature
import log_mlflow


def test_experiment_loop(TOP_K, dataset, want_col, num_rows, ratio, seed):
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
    data = datasets_loader.loader(dataset, want_col, num_rows, seed)

    # Use the standalone function to hide information from the test data
    hidden_test_data = data_manipulation.hide_information_in_dataframe(
        data=data,
        hide_type="values_in_column",
        columns_to_hide="rating",
        fraction_to_hide=0.5, # Hide 50% of ratings in the test set
        seed=42
    )

    print("Test Data with 50% Ratings Hidden:")
    print(hidden_test_data)
    print("-" * 20)

    # Example: Hide entire columns
    hidden_data_cols = data_manipulation.hide_information_in_dataframe(
        data=data,
        hide_type="columns",
        columns_to_hide=["timestamp", "rating"]
    )
    print("Dataset with 'timestamp' and 'rating' columns hidden:")
    print(hidden_data_cols)
    print("-" * 20)


    # Example: Hide random records
    hidden_data_records = data_manipulation.hide_information_in_dataframe(
        data=data,
        hide_type="records_random",
        fraction_to_hide=0.3, # Hide 30% of records
        seed=99
    )
    print(f"Dataset with 30% random records hidden (original size={len(data)}):")
    print(hidden_data_records)
    print(f"New size: {len(hidden_data_records)}")
    print("-" * 20)

    # --- Specify indices to hide ---
    # Let's say we want to remove the rows corresponding to Bob (index 1)
    # and Eve (index 4)
    indices_to_remove = [1, 4]

    # --- Call the function ---
    df_hidden_selective = data_manipulation.hide_information_in_dataframe(
        data=data,
        hide_type="records_selective",
        records_to_hide=indices_to_remove
    )

    print(f"DataFrame after selectively hiding records with indices {indices_to_remove}:")
    print(df_hidden_selective)
    print("-" * 30)


    print(data)