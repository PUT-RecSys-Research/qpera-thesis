import cornac
import data_manipulation as dm
import datasets_loader
import log_mlflow
from metrics import (
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
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer


def cf_experiment_loop(
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

    # Model parameters
    NUM_FACTORS = 200
    NUM_EPOCHS = 100

    # Load dataset
    data = datasets_loader.loader(dataset, want_col, num_rows, seed)

    # Algorithm
    train, test = python_stratified_split(data, ratio=ratio, col_user=header["col_user"], col_item=header["col_item"], seed=seed)

    if privacy:
        train = dm.hide_information_in_dataframe(
            data=train, hide_type=hide_type, columns_to_hide=columns_to_hide, fraction_to_hide=fraction_to_hide, records_to_hide=records_to_hide, seed=seed
        )

    if personalization:
        train = dm.change_items_in_dataframe(all=data, data=train, fraction_to_change=fraction_to_change, change_rating=change_rating, seed=seed)

    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=seed)

    bpr = cornac.models.BPR(k=NUM_FACTORS, max_iter=NUM_EPOCHS, learning_rate=0.01, lambda_reg=0.001, verbose=True, seed=seed)
    with Timer() as t:
        bpr.fit(train_set)
    print("Took {} seconds for training.".format(t))

    with Timer() as t:
        all_predictions = predict_ranking(bpr, train, usercol="userID", itemcol="itemID", remove_seen=True)
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

    top = all_predictions.loc[all_predictions.groupby("userID")["prediction"].idxmax(), ["userID", "itemID", "prediction"]]
    # print(top)
    top_k = all_predictions

    # Metrics
    try:
        eval_precision_at_k = precision_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K)
    except Exception as e:
        eval_precision_at_k = None
        print(f"Error calculating precision at k: {e}")
    try:
        eval_recall_at_k = recall_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K)
    except Exception as e:
        eval_recall_at_k = None
        print(f"Error calculating recall at k: {e}")
    try:
        eval_ndcg = ndcg_at_k(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", relevancy_method="top_k", k=1)
    except Exception as e:
        eval_ndcg = None
        print(f"Error calculating NDCG: {e}")
    try:
        eval_precision = precision_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    except Exception as e:
        eval_precision = None
        print(f"Error calculating precision: {e}")
    try:
        eval_recall = recall_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    except Exception as e:
        eval_recall = None
        print(f"Error calculating recall: {e}")
    try:
        eval_mae = mae(test, top_k)
    except Exception as e:
        eval_mae = None
        print(f"Error calculating MAE: {e}")
    try:
        eval_rmse = rmse(test, top_k)
    except Exception as e:
        eval_rmse = None
        print(f"Error calculating RMSE: {e}")

    # eval_novelty = novelty(train, top)
    # eval_historical_item_novelty = historical_item_novelty(train, top)
    # eval_user_item_serendipity = user_item_serendipity(train, top)
    # eval_user_serendipity = user_serendipity(train, top)
    # eval_serendipity = serendipity(train, top)
    # eval_catalog_coverage = catalog_coverage(train, top)
    # eval_distributional_coverage = distributional_coverage(train, top)

    # eval_f1 = f1(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    try:
        eval_mrr = mrr(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    except Exception as e:
        eval_mrr = None
        print(f"Error calculating MRR: {e}")
    # eval_accuracy = accuracy(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    try:
        eval_user_coverage = user_coverage(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    except Exception as e:
        eval_user_coverage = None
        print(f"Error calculating user coverage: {e}")
    try:
        eval_item_coverage = item_coverage(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    except Exception as e:
        eval_item_coverage = None
        print(f"Error calculating item coverage: {e}")

    try:
        eval_intra_list_similarity = intra_list_similarity_score(data, top_k, feature_cols=["genres"])
    except Exception as e:
        eval_intra_list_similarity = None
        print(f"Error calculating intra-list similarity: {e}")
    try:
        eval_intra_list_dissimilarity = intra_list_dissimilarity(data, top_k, feature_cols=["genres"])
    except Exception as e:
        eval_intra_list_dissimilarity = None
        print(f"Error calculating intra-list dissimilarity: {e}")
    try:
        eval_personalization = personalization_score(train, top)
    except Exception as e:
        eval_personalization = None
        print(f"Error calculating personalization: {e}")

    def format_metric(metric):
        return f"{metric:.4f}" if isinstance(metric, (float, int)) else "N/A"

    print(
        "Precision:\t" + format_metric(eval_precision),
        "Precision@K:\t" + format_metric(eval_precision_at_k),
        "Recall:\t" + format_metric(eval_recall),
        "Recall@K:\t" + format_metric(eval_recall_at_k),
        # "F1:\t" + format_metric(eval_f1),
        # "Accuracy:\t" + format_metric(eval_accuracy),
        "MAE:\t" + format_metric(eval_mae),
        "RMSE:\t" + format_metric(eval_rmse),
        "NDCG:\t" + format_metric(eval_ndcg),
        "MRR:\t" + format_metric(eval_mrr),
        # "Novelty:\t" + format_metric(eval_novelty),
        # "Serendipity:\t" + format_metric(eval_serendipity),
        "User coverage:\t" + format_metric(eval_user_coverage),
        "Item coverage:\t" + format_metric(eval_item_coverage),
        # "Catalog coverage:\t" + format_metric(eval_catalog_coverage),
        # "Distributional coverage:\t" + format_metric(eval_distributional_coverage),
        "Personalization:\t" + format_metric(eval_personalization),
        "Intra-list similarity:\t" + format_metric(eval_intra_list_similarity),
        "Intra-list dissimilarity:\t" + format_metric(eval_intra_list_dissimilarity),
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
        bpr,
        "CF",
        params,
        data,
        train,
        privacy=privacy,
        fraction_to_hide=fraction_to_hide,
        personalization=personalization,
        fraction_to_change=fraction_to_change,
    )
