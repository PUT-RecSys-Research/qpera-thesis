import cornac
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import (
    mae,
    ndcg_at_k,
    rmse,
)
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer

from . import data_manipulation as dm
from . import datasets_loader, log_mlflow
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
    """
    Execute a complete Collaborative Filtering experiment using BPR with optional personalization and privacy modifications.
    
    Args:
        TOP_K (int): Number of top recommendations to generate
        dataset (str): Dataset identifier to load
        want_col (list): Columns to include in the dataset
        num_rows (int): Number of rows to load from dataset
        ratio (float): Train/test split ratio
        seed (int): Random seed for reproducibility
        personalization (bool): Whether to apply personalization modifications
        fraction_to_change (float): Fraction of items to modify for personalization
        change_rating (bool): Whether to change ratings during personalization
        privacy (bool): Whether to apply privacy modifications
        hide_type (str): Type of hiding strategy for privacy
        columns_to_hide (list): Columns to hide for privacy
        fraction_to_hide (float): Fraction of data to hide for privacy
        records_to_hide (list): Specific records to hide for privacy
    
    Returns:
        None: Logs results to MLflow and prints metrics
    """
    
    # Configuration parameters for experiment tracking
    params = {
        "dataset": dataset,
        "want_col": want_col,
        "num_rows": num_rows,
        "ratio": ratio,
        "seed": seed,
    }

    # Column mapping for consistent data handling
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

    # BPR model hyperparameters
    NUM_FACTORS = 200
    NUM_EPOCHS = 100

    # Load and prepare dataset
    data = datasets_loader.loader(dataset, want_col, num_rows, seed)

    # Split data into training and testing sets
    train, test = python_stratified_split(
        data, 
        ratio=ratio, 
        col_user=header["col_user"], 
        col_item=header["col_item"], 
        seed=seed
    )

    # Apply privacy modifications if requested
    if privacy:
        train = dm.hide_information_in_dataframe(
            data=train, 
            hide_type=hide_type, 
            columns_to_hide=columns_to_hide, 
            fraction_to_hide=fraction_to_hide, 
            records_to_hide=records_to_hide, 
            seed=seed
        )

    # Apply personalization modifications if requested
    if personalization:
        train = dm.change_items_in_dataframe(
            all=data, 
            data=train, 
            fraction_to_change=fraction_to_change, 
            change_rating=change_rating, 
            seed=seed
        )

    # Convert training data to Cornac dataset format
    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=seed)

    # Initialize and train BPR model
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
    print(f"Training completed in {t} seconds")

    # Generate predictions for all user-item pairs
    with Timer() as t:
        all_predictions = predict_ranking(
            bpr, 
            train, 
            usercol="userID", 
            itemcol="itemID", 
            remove_seen=True
        )
    print(f"Prediction completed in {t} seconds")

    # Display sample predictions
    print("Sample predictions:")
    print(all_predictions.head())

    # Merge test data with predictions for evaluation
    merged_df = test.merge(
        all_predictions, 
        on=["userID", "itemID"], 
        how="inner"
    )[["userID", "itemID", "rating", "prediction"]]
    print(f"Merged evaluation data: {merged_df.shape[0]} interactions")

    # Get top recommendation per user and all predictions for different metrics
    top = all_predictions.loc[
        all_predictions.groupby("userID")["prediction"].idxmax(), 
        ["userID", "itemID", "prediction"]
    ]
    top_k = all_predictions

    # Calculate evaluation metrics with error handling
    def safe_metric_calculation(metric_func, *args, **kwargs):
        """Safely calculate metrics with error handling."""
        try:
            return metric_func(*args, **kwargs)
        except Exception as e:
            print(f"Error calculating {metric_func.__name__}: {e}")
            return None

    # Accuracy metrics
    eval_precision_at_k = safe_metric_calculation(
        precision_at_k, test, top_k, 
        col_user="userID", col_item="itemID", col_rating="rating", 
        col_prediction="prediction", k=TOP_K
    )
    eval_recall_at_k = safe_metric_calculation(
        recall_at_k, test, top_k, 
        col_user="userID", col_item="itemID", col_rating="rating", 
        col_prediction="prediction", k=TOP_K
    )
    eval_ndcg = safe_metric_calculation(
        ndcg_at_k, test, top, 
        col_user="userID", col_item="itemID", col_rating="rating", 
        col_prediction="prediction", relevancy_method="top_k", k=1
    )
    eval_precision = safe_metric_calculation(
        precision_at_k, test, top_k, 
        col_user="userID", col_item="itemID", col_rating="rating", 
        col_prediction="prediction", k=1
    )
    eval_recall = safe_metric_calculation(
        recall_at_k, test, top_k, 
        col_user="userID", col_item="itemID", col_rating="rating", 
        col_prediction="prediction", k=1
    )
    
    # Error metrics
    eval_mae = safe_metric_calculation(mae, test, top_k)
    eval_rmse = safe_metric_calculation(rmse, test, top_k)
    
    # Ranking metrics
    eval_mrr = safe_metric_calculation(
        mrr, test, top_k, 
        col_user="userID", col_item="itemID", col_rating="rating", 
        col_prediction="prediction"
    )
    
    # Coverage metrics
    eval_user_coverage = safe_metric_calculation(
        user_coverage, test, top, 
        col_user="userID", col_item="itemID", col_rating="rating", 
        col_prediction="prediction"
    )
    eval_item_coverage = safe_metric_calculation(
        item_coverage, test, top, 
        col_user="userID", col_item="itemID", col_rating="rating", 
        col_prediction="prediction"
    )
    
    # Diversity and personalization metrics
    eval_intra_list_similarity = safe_metric_calculation(
        intra_list_similarity_score, data, top_k, feature_cols=["genres"]
    )
    eval_intra_list_dissimilarity = safe_metric_calculation(
        intra_list_dissimilarity, data, top_k, feature_cols=["genres"]
    )
    eval_personalization = safe_metric_calculation(personalization_score, train, top)

    def format_metric(metric):
        """Format metric values for display."""
        return f"{metric:.4f}" if isinstance(metric, (float, int)) else "N/A"

    # Print evaluation results
    print(
        "Precision:\t" + format_metric(eval_precision),
        "Precision@K:\t" + format_metric(eval_precision_at_k),
        "Recall:\t" + format_metric(eval_recall),
        "Recall@K:\t" + format_metric(eval_recall_at_k),
        "MAE:\t" + format_metric(eval_mae),
        "RMSE:\t" + format_metric(eval_rmse),
        "NDCG:\t" + format_metric(eval_ndcg),
        "MRR:\t" + format_metric(eval_mrr),
        "User coverage:\t" + format_metric(eval_user_coverage),
        "Item coverage:\t" + format_metric(eval_item_coverage),
        "Personalization:\t" + format_metric(eval_personalization),
        "Intra-list similarity:\t" + format_metric(eval_intra_list_similarity),
        "Intra-list dissimilarity:\t" + format_metric(eval_intra_list_dissimilarity),
        sep="\n",
    )

    # Compile metrics for logging
    metrics = {
        "precision": eval_precision,
        "precision_at_k": eval_precision_at_k,
        "recall": eval_recall,
        "recall_at_k": eval_recall_at_k,
        "mae": eval_mae,
        "rmse": eval_rmse,
        "mrr": eval_mrr,
        "ndcg_at_k": eval_ndcg,
        "user_coverage": eval_user_coverage,
        "item_coverage": eval_item_coverage,
        "personalization": eval_personalization,
        "intra_list_similarity": eval_intra_list_similarity,
        "intra_list_dissimilarity": eval_intra_list_dissimilarity,
    }

    # Log experiment results to MLflow
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
