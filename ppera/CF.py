import cornac

import data_manipulation as dm
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, mae, rmse, novelty, historical_item_novelty, user_item_serendipity, user_serendipity, serendipity, catalog_coverage, distributional_coverage
from metrics import precision_at_k, recall_at_k, f1, mrr, accuracy, user_coverage, item_coverage, intra_list_similarity_score, intra_list_dissimilarity, personalization_score
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer

import datasets_loader
import log_mlflow

def cf_experiment_loop(
        TOP_K, 
        dataset, 
        want_col, 
        num_rows, 
        ratio, 
        seed,
        personalization=False,
        fraction_to_change = 0,
        change_rating = False,
        privacy=False,
        hide_type="values_in_column",
        columns_to_hide=None,
        fraction_to_hide = 0,
        records_to_hide=None
        ):
    TOP_K = TOP_K
    dataset = dataset
    want_col = want_col
    num_rows = num_rows
    ratio = ratio
    seed = seed
    personalization=personalization
    fraction_to_change=fraction_to_change
    change_rating=change_rating
    privacy = privacy
    hide_type=hide_type
    columns_to_hide=columns_to_hide
    fraction_to_hide=fraction_to_hide
    records_to_hide=records_to_hide

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
    train, test = python_stratified_split(
        data, ratio=ratio, col_user=header["col_user"], col_item=header["col_item"], seed=seed
    )

    if privacy:
        train = dm.hide_information_in_dataframe(
            data=train, 
            hide_type=hide_type, 
            columns_to_hide=columns_to_hide, 
            fraction_to_hide=fraction_to_hide,
            records_to_hide=records_to_hide,
            seed=seed)
        
    if personalization:
        train = dm.change_items_in_dataframe(
            all=data,
            data=train,
            fraction_to_change=fraction_to_change,
            change_rating=change_rating,
            seed=seed
        )


    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=seed)

    bpr = cornac.models.BPR(
        k=NUM_FACTORS,
        max_iter=NUM_EPOCHS,
        learning_rate=0.01,
        lambda_reg=0.001,
        verbose=True,
        seed=seed
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

    top = all_predictions.loc[all_predictions.groupby("userID")["prediction"].idxmax(), ["userID", "itemID", "prediction"]]
    # print(top)
    top_k = all_predictions

    #metrics

    eval_map = map_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction",relevancy_method="top_k", k=TOP_K)
    # eval_ndcg_at_k = ndcg_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction",relevancy_method="top_k", k=TOP_K)
    eval_precision_at_k = precision_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K)
    eval_recall_at_k = recall_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=TOP_K)
    eval_ndcg = ndcg_at_k(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", relevancy_method="top_k",k=1)
    eval_precision = precision_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    eval_recall = recall_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    eval_mae = mae(test, top_k)
    eval_rmse = rmse(test, top_k)

    eval_novelty = novelty(train, top)
    # eval_historical_item_novelty = historical_item_novelty(train, top)
    # eval_user_item_serendipity = user_item_serendipity(train, top)
    # eval_user_serendipity = user_serendipity(train, top)
    eval_serendipity = serendipity(train, top)
    eval_catalog_coverage = catalog_coverage(train, top)
    eval_distributional_coverage = distributional_coverage(train, top)

    eval_f1 = f1(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=1)
    # eval_mrr = mrr(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    # eval_accuracy = accuracy(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    eval_user_coverage = user_coverage(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")
    eval_item_coverage = item_coverage(test, top, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction")


    eval_intra_list_similarity = intra_list_similarity_score(data, top_k, feature_cols=['genres'])
    eval_intra_list_dissimilarity = intra_list_dissimilarity(data, top_k, feature_cols=['genres'])
    eval_personalization = personalization_score(train, top)

    print(
        "Precision:\t%f" % eval_precision,
        "Precision@K:\t%f" % eval_precision_at_k,
        "Recall:\t%f" % eval_recall,
        "Recall@K:\t%f" % eval_recall_at_k,
        "F1:\t%f" % eval_f1,
        # "Accuracy:\t%f" % eval_accuracy,
        "MAE:\t%f" % eval_mae,
        "RMSE:\t%f" % eval_rmse,
        "NDCG:\t%f" % eval_ndcg,
        # "MRR:\t%f" % eval_mrr,
        "Novelty:\t%f" % eval_novelty,
        "Serendipity:\t%f" % eval_serendipity,
        "User coverage:\t%f" % eval_user_coverage,
        "Item coverage:\t%f" % eval_item_coverage,
        "Catalog coverage:\t%f" % eval_catalog_coverage,
        "Distributional coverage:\t%f" % eval_distributional_coverage,
        "Personalization:\t%f" % eval_personalization,
        "Intra-list similarity:\t%f" % eval_intra_list_similarity,
        "Intra-list dissimilarity:\t%f" % eval_intra_list_dissimilarity,
      sep='\n')

    
    metrics = {
        "precision": eval_precision,
        "precision_at_k": eval_precision_at_k,
        "recall": eval_recall,
        "recall_at_k": eval_recall_at_k,
        "f1": eval_f1,
        "mae": eval_mae,                      
        "rmse": eval_rmse,                    
        "ndcg_at_k": eval_ndcg,               
        "novelty": eval_novelty,
        "serendipity": eval_serendipity,
        "user_coverage": eval_user_coverage,  
        "item_coverage": eval_item_coverage,
        "catalog_coverage": eval_catalog_coverage,
        "distributional_coverage": eval_distributional_coverage,
        "personalization": eval_personalization,
        "intra_list_similarity": eval_intra_list_similarity,
        "intra_list_dissimilarity": eval_intra_list_dissimilarity,
    }

    log_mlflow.log_mlflow(dataset, top_k, metrics, num_rows, seed, bpr, 'CF', params, data, train)