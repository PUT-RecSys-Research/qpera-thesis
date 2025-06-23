import os
from datetime import datetime
from typing import Dict, Any, Optional

import cornac
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from mlflow.models.signature import infer_signature
from recommenders.models.cornac.cornac_utils import predict_ranking

from .rl_train_agent import ActorCritic


# Dataset constants
DATASET_NAMES = {
    "MOVIELENS": "movielens",
    "AMAZONSALES": "amazonsales", 
    "POSTRECOMMENDATIONS": "postrecommendations"
}

# Model type constants
MODEL_TYPES = {
    "CF": "Collaborative Filtering",
    "CBF": "Content Based Filtering", 
    "RL": "Reinforcement Learning"
}


def log_mlflow(
    dataset: str,
    top_k: pd.DataFrame,
    metrics: Dict[str, Any],
    num_rows: Optional[int],
    seed: int,
    model: Any,
    model_type: str,
    params: Dict[str, Any],
    data: pd.DataFrame,
    train: pd.DataFrame,
    tf: Any = None,
    vectors_tokenized: Any = None,
    privacy: Optional[bool] = None,
    fraction_to_hide: Optional[float] = None,
    personalization: Optional[bool] = None,
    fraction_to_change: Optional[float] = None,
) -> None:
    """
    Log experiment results, metrics, and models to MLflow with artifacts.
    
    Args:
        dataset: Dataset name identifier
        top_k: DataFrame with top-k predictions
        metrics: Dictionary of evaluation metrics
        num_rows: Number of rows used in experiment
        seed: Random seed used
        model: Trained model object
        model_type: Type of model (CF, CBF, or RL)
        params: Model parameters and configuration
        data: Complete dataset
        train: Training dataset
        tf: TF-IDF transformer (for CBF models)
        vectors_tokenized: Tokenized vectors (for CBF models)
        privacy: Whether privacy modifications were applied
        fraction_to_hide: Fraction of data hidden for privacy
        personalization: Whether personalization modifications were applied
        fraction_to_change: Fraction of data changed for personalization
    """
    
    # Generate timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create and save prediction visualization
    _create_prediction_plot(top_k, model_type, dataset, timestamp)
    
    # Setup directory structure for metrics
    metrics_dir = _setup_metrics_directory(privacy, personalization, fraction_to_hide, fraction_to_change)
    
    # Save metrics to CSV
    metrics_filename = _save_metrics_csv(metrics, metrics_dir, model_type, dataset)
    
    # Setup MLflow experiment
    _setup_mlflow_experiment(model_type)
    
    # Get dataset file path
    dataset_file_path = _get_dataset_file_path(dataset, num_rows, seed)
    
    # Start MLflow run and log everything
    _log_to_mlflow(
        dataset, data, dataset_file_path, privacy, fraction_to_hide,
        personalization, fraction_to_change, timestamp, metrics_filename,
        params, metrics, model_type, model, train, tf, vectors_tokenized
    )


def _create_prediction_plot(top_k: pd.DataFrame, model_type: str, dataset: str, timestamp: str) -> str:
    """Create and save visualization of top-k predictions."""
    top_k_sample = top_k.head(10)
    print("Sample of top-k predictions:")
    print(top_k_sample)
    
    plot_filename = f"ppera/plots/top_k_predictions_{model_type}_{dataset}_{timestamp}.png"
    os.makedirs("ppera/plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(top_k_sample["itemID"], top_k_sample["prediction"], marker="o", linestyle="-")
    plt.xlabel("ItemID")
    plt.ylabel("Prediction")
    plt.title(f"Top K Predictions for User {top_k_sample['userID'].iloc[0]}")
    plt.grid(True)
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename


def _setup_metrics_directory(
    privacy: Optional[bool], 
    personalization: Optional[bool], 
    fraction_to_hide: Optional[float], 
    fraction_to_change: Optional[float]
) -> str:
    """Setup directory structure for saving metrics based on experiment type."""
    metrics_base_dir = "ppera/metrics"
    
    if privacy is True:
        subdir = "privacy"
        specific_dir = f"no_mod_{fraction_to_hide:.2f}"
    elif personalization is True:
        subdir = "personalization"
        specific_dir = f"no_mod_{fraction_to_change:.2f}"
    elif privacy is False and personalization is False:
        subdir = "clean_loop"
        specific_dir = "no_mod"
    else:
        subdir = "other"
        specific_dir = "unknown"
    
    full_metrics_dir = os.path.join(metrics_base_dir, subdir, specific_dir)
    os.makedirs(full_metrics_dir, exist_ok=True)
    return full_metrics_dir


def _save_metrics_csv(metrics: Dict[str, Any], metrics_dir: str, model_type: str, dataset: str) -> str:
    """Save metrics to CSV file."""
    metrics_filename = os.path.join(metrics_dir, f"{model_type}_{dataset}_metrics.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_filename, index=False)
    return metrics_filename


def _setup_mlflow_experiment(model_type: str) -> None:
    """Setup MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    
    experiment_name = f"MLflow {MODEL_TYPES.get(model_type, 'Unknown')}"
    mlflow.set_experiment(experiment_name)


def _get_dataset_file_path(dataset: str, num_rows: Optional[int], seed: int) -> str:
    """Generate file path for dataset based on parameters."""
    base_paths = {
        DATASET_NAMES["MOVIELENS"]: "ppera/datasets/MovieLens/merge_file.csv",
        DATASET_NAMES["AMAZONSALES"]: "ppera/datasets/AmazonSales/merge_file.csv",
        DATASET_NAMES["POSTRECOMMENDATIONS"]: "ppera/datasets/PostRecommendations/merge_file.csv"
    }
    
    base_path = base_paths.get(dataset)
    if not base_path:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if num_rows is not None:
        return base_path.replace(".csv", f"_r{num_rows}_s{seed}.csv")
    return base_path


def _log_to_mlflow(
    dataset: str, data: pd.DataFrame, dataset_file_path: str,
    privacy: Optional[bool], fraction_to_hide: Optional[float],
    personalization: Optional[bool], fraction_to_change: Optional[float],
    timestamp: str, metrics_filename: str, params: Dict[str, Any],
    metrics: Dict[str, Any], model_type: str, model: Any,
    train: pd.DataFrame, tf: Any, vectors_tokenized: Any
) -> None:
    """Log all experiment data to MLflow."""
    
    with mlflow.start_run():
        # Log dataset
        _log_dataset_to_mlflow(dataset, data, dataset_file_path, privacy, fraction_to_hide, 
                              personalization, fraction_to_change)
        
        # Log artifacts
        plot_filename = f"ppera/plots/top_k_predictions_{model_type}_{dataset}_{timestamp}.png"
        mlflow.log_artifact(plot_filename, artifact_path="plots")
        mlflow.log_artifact(metrics_filename, artifact_path="metrics")
        
        # Log parameters and metrics
        mlflow.log_params(params)
        _log_metrics_to_mlflow(metrics)
        
        # Set experiment tags
        mlflow.set_tag("Metrics Info", f"{model_type} model for {dataset} dataset")
        
        # Log model with signature inference
        _log_model_to_mlflow(model_type, model, train, params, tf, vectors_tokenized, dataset)


def _log_dataset_to_mlflow(
    dataset: str, data: pd.DataFrame, file_path: str,
    privacy: Optional[bool], fraction_to_hide: Optional[float],
    personalization: Optional[bool], fraction_to_change: Optional[float]
) -> None:
    """Log dataset information to MLflow."""
    
    dataset_names = {
        DATASET_NAMES["MOVIELENS"]: ("MovieLens Dataset", "ppera/datasets/MovieLens"),
        DATASET_NAMES["AMAZONSALES"]: ("AmazonSales Dataset", "ppera/datasets/AmazonSales"),
        DATASET_NAMES["POSTRECOMMENDATIONS"]: ("PostRecommendations Dataset", "ppera/datasets/PostRecommendations")
    }
    
    if dataset in dataset_names:
        name, artifact_path = dataset_names[dataset]
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
        
        dataset_df = mlflow.data.from_pandas(data, name=name, source=file_path)
        context = f"Privacy: {privacy} | {fraction_to_hide}, Personalization: {personalization} | {fraction_to_change}"
        mlflow.log_input(dataset_df, context=context)


def _log_metrics_to_mlflow(metrics: Dict[str, Any]) -> None:
    """Log metrics to MLflow, filtering out None values."""
    # Filter out None values
    clean_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    mlflow.log_metrics(clean_metrics)
    
    # Report skipped metrics
    none_metrics = {k: v for k, v in metrics.items() if v is None}
    if none_metrics:
        print("Skipped logging the following None-valued metrics to MLflow:", list(none_metrics.keys()))


def _log_model_to_mlflow(
    model_type: str, model: Any, train: pd.DataFrame, params: Dict[str, Any],
    tf: Any, vectors_tokenized: Any, dataset: str
) -> None:
    """Log model to MLflow with signature inference."""
    
    if not model:
        print(f"MLflow: Model object for {model_type} not available. Skipping model logging.")
        return
    
    signature, input_example = _infer_model_signature(model_type, model, train, params, tf, vectors_tokenized)
    
    try:
        _log_model_with_signature(model_type, model, signature, input_example, dataset)
        print(f"MLflow: Successfully logged {model_type} model")
    except Exception as e:
        print(f"MLflow: Error logging {model_type} model: {e}")
        _log_model_fallback(model_type, model, dataset)


def _infer_model_signature(
    model_type: str, model: Any, train: pd.DataFrame, params: Dict[str, Any],
    tf: Any, vectors_tokenized: Any
) -> tuple:
    """Infer model signature and prepare input example."""
    
    signature = None
    input_example = None
    
    try:
        if model_type == "CF" and isinstance(model, cornac.models.Recommender):
            signature, input_example = _infer_cf_signature(model, train, params)
        elif model_type == "CBF":
            signature, input_example = _infer_cbf_signature(model, train, tf, vectors_tokenized)
        elif model_type == "RL" and isinstance(model, ActorCritic):
            signature, input_example = _infer_rl_signature(model)
        else:
            print(f"MLflow: Unknown model type or unsupported model class for {model_type}")
            
    except Exception as e:
        print(f"MLflow: Error during {model_type} signature inference: {e}")
    
    return signature, input_example


def _infer_cf_signature(model: Any, train: pd.DataFrame, params: Dict[str, Any]) -> tuple:
    """Infer signature for Collaborative Filtering models."""
    if train.empty:
        return None, None
    
    user_col = params.get("col_user", "userID")
    item_col = params.get("col_item", "itemID")
    
    # Create sample input
    sample_users = train[user_col].unique()[:2]
    sample_items = train[item_col].unique()[:5]
    
    sample_input_data = []
    for user in sample_users:
        for item in sample_items:
            sample_input_data.append({user_col: user, item_col: item})
    
    if not sample_input_data:
        return None, None
    
    sample_input_df = pd.DataFrame(sample_input_data)
    sample_predictions = predict_ranking(model, sample_input_df, usercol=user_col, itemcol=item_col, remove_seen=True)
    
    if sample_predictions.empty:
        return None, None
    
    signature_input = sample_input_df[[user_col]].drop_duplicates().reset_index(drop=True)
    signature_output = sample_predictions[[user_col, item_col, params.get("col_prediction", "prediction")]]
    
    signature = infer_signature(signature_input, signature_output)
    input_example = signature_input.head(5)
    
    return signature, input_example


def _infer_cbf_signature(model: Any, train: pd.DataFrame, tf: Any, vectors_tokenized: Any) -> tuple:
    """Infer signature for Content-Based Filtering models."""
    input_example = train.head(5)
    signature = infer_signature(train, model.fit(tf, vectors_tokenized))
    return signature, input_example


def _infer_rl_signature(model: ActorCritic) -> tuple:
    """Infer signature for Reinforcement Learning models."""
    # Create sample tensors
    sample_state = torch.randn(1, model.state_dim)
    sample_action_mask = torch.ones(1, model.act_dim, dtype=torch.bool)
    
    # Move to model device
    device = next(model.parameters()).device
    sample_input = (sample_state.to(device), sample_action_mask.to(device))
    
    # Get sample prediction
    with torch.no_grad():
        model.eval()
        probs, values = model(sample_input)
    
    # Prepare for signature inference
    model_input = sample_state.cpu().numpy()
    model_output = {
        "action_probabilities": probs.cpu().numpy(),
        "state_values": values.cpu().numpy(),
    }
    
    signature = infer_signature(model_input, model_output)
    input_example = pd.DataFrame(sample_input[0].cpu().numpy())
    
    return signature, input_example


def _log_model_with_signature(
    model_type: str, model: Any, signature: Any, input_example: Any, dataset: str
) -> None:
    """Log model with signature to MLflow."""
    
    model_name = f"{model_type}-model-{dataset}"
    artifact_path = f"{model_type}-model"
    
    if model_type == "RL":
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )
    elif model_type == "CF":
        mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )
    elif model_type == "CBF":
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )


def _log_model_fallback(model_type: str, model: Any, dataset: str) -> None:
    """Fallback model logging without signature."""
    
    model_name = f"{model_type}-model-{dataset}"
    artifact_path = f"{model_type}-model"
    
    try:
        if model_type == "RL":
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=model_name
            )
        elif model_type in ["CF", "CBF"]:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=model_name
            )
        
        print(f"MLflow: Logged {model_type} model without signature (fallback)")
    except Exception as e:
        print(f"MLflow: Fallback logging also failed for {model_type}: {e}")
