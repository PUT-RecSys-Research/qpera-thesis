import os
import sys
import cornac
import pandas as pd
import datasets_loader

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, mae, rmse, novelty, historical_item_novelty, user_item_serendipity, user_serendipity, serendipity, catalog_coverage, distributional_coverage
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
# from metrics import precision_at_k, recall_at_k, mrr, accuracy, user_coverage, item_coverage

def cf_bpr_experiment_loop(TOP_K, dataset, want_col, num_rows, ratio, seed):
    TOP_K = TOP_K
    dataset = dataset
    want_col = want_col
    num_rows = num_rows
    ratio = ratio
    seed = seed

    # Model parameters
    NUM_FACTORS = 200
    NUM_EPOCHS = 100

    # Load dataset
    data = datasets_loader.loader(dataset, want_col, num_rows, seed)

    # Algorithm
    train, test = python_random_split(data, 0.75)
    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

    bpr = cornac.models.BPR(
        k=NUM_FACTORS,
        max_iter=NUM_EPOCHS,
        learning_rate=0.01,
        lambda_reg=0.001,
        verbose=True,
        seed=SEED
    )
    with Timer() as t:
        bpr.fit(train_set)
    print("Took {} seconds for training.".format(t))

    with Timer() as t:
        all_predictions = predict_ranking(bpr, train, usercol='userID', itemcol='itemID', remove_seen=True)
    print("Took {} seconds for prediction.".format(t))

    print(all_predictions.head())

    all_predictions[all_predictions["userID"] == 1]

    all_predictions[all_predictions["userID"] == 1].sort_values(by="prediction", ascending=False)


    rows, columns = test.shape
    rows

    merged_df = test.merge(all_predictions, on=["userID", "itemID"], how="inner")[["userID", "itemID", "rating", "prediction"]]
    print(merged_df)

    user_counts = test["userID"].value_counts().reset_index()
    user_counts.columns = ["userID", "count"]
    print(user_counts)


    max_predictions = all_predictions.loc[all_predictions.groupby("userID")["prediction"].idxmax(), ["userID", "itemID", "prediction"]]
    print(max_predictions)

    k = 1

    # eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=k)
    # eval_precision_k = precision_at_k(test, all_predictions, col_prediction='prediction', k=10)
    # eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=k)
    # eval_recall_k = recall_at_k(test, all_predictions, col_prediction='prediction', k=10)
    # eval_f1
    eval_mae = mae(test, all_predictions, col_prediction='prediction')
    eval_rmse = rmse(test, all_predictions, col_prediction='prediction')
    # eval_mrr
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=k)
    # eval_intra_list_diversity
    # eval_user_coverage
    # eval_item_coverage
    # eval_personalization

    # eval_novelty = novelty(train, max_predictions)
    # eval_historical_item_novelty = historical_item_novelty(train, max_predictions)
    # eval_user_item_serendipity = user_item_serendipity(train, max_predictions)
    # eval_user_serendipity = user_serendipity(train, max_predictions)
    # eval_serendipity = serendipity(train, max_predictions)
    # eval_catalog_coverage = catalog_coverage(train, max_predictions)
    # eval_distributional_coverage = distributional_coverage(train, max_predictions)

    print(#"Precision:\t%f" % eval_precision,
        # "Precision@K:\t%f" % eval_precision_k,
        # "Recall:\t%f" % eval_recall,
        # "Recall@K:\t%f" % eval_recall_k,
        "MAE:\t%f" % eval_mae,
        "RMSE:\t%f" % eval_rmse,
        "NDCG:\t%f" % eval_ndcg,
        # "Novelty:\t%f" % eval_novelty,
        # "Serendipity:\t%f" % eval_serendipity,
        # "Catalog coverage:\t%f" % eval_catalog_coverage,
        # "Distributional coverage:\t%f" % eval_distributional_coverage,
        sep='\n')