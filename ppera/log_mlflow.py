import os
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
from mlflow.models.signature import infer_signature

def log_mlflow(dataset, top_k, metrics, num_rows, seed, model, model_type, params, data, train, tf=None, vectors_tokenized=None):

    top_k_prediction = top_k.head(10)
    print(top_k_prediction)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'plots/top_k_predictions_{model_type}_{dataset}_{timestamp}.png'
    os.makedirs('plots', exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(top_k_prediction['itemID'], top_k_prediction['prediction'], marker='o', linestyle='-')
    plt.xlabel('ItemID')
    plt.ylabel('Prediction')
    plt.title(f"Top K Predictions for User {top_k_prediction['userID'].iloc[0]}")
    plt.grid(True)
    plt.savefig(plot_filename)
    plt.close()

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    if model_type == 'CF':
        mlflow.set_experiment("MLflow Collaborative Filtering")
    elif model_type == 'CBF':
        mlflow.set_experiment("MLflow Content Based Filtering")
    elif model_type == 'RL':
        mlflow.set_experiment("MLflow Reinforcement Learning")

    if dataset == 'movielens':
        file_path = f"datasets/MovieLens/merge_file_r{num_rows}_s{seed}.csv"
    elif dataset == 'amazonsales':
        file_path = f"datasets/AmazonSales/merge_file_r{num_rows}_s{seed}.csv"
    elif dataset == 'postrecommendations':
        file_path = f"datasets/PostRecommendations/merge_file_r{num_rows}_s{seed}.csv"

    with mlflow.start_run():
        if dataset == 'movielens':
            mlflow.log_artifact(file_path, artifact_path="datasets/MovieLens")
            dataset = mlflow.data.from_pandas(data, name="MovieLens Dataset", source=file_path)
            mlflow.log_input(dataset, context="test")
        elif dataset == 'amazonsales':
            mlflow.log_artifact(file_path, artifact_path="datasets/AmazonSales")
            dataset = mlflow.data.from_pandas(data, name="AmazonSales Dataset", source=file_path)
            mlflow.log_input(dataset, context="test")
        elif dataset == 'postrecommendations':
            mlflow.log_artifact(file_path, artifact_path="datasets/PostRecommendations")
            dataset = mlflow.data.from_pandas(data, name="PostRecommendations Dataset", source=file_path)
            mlflow.log_input(dataset, context="test")

        mlflow.log_artifact(plot_filename, artifact_path='plots')
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("Metrics Info", f"{model_type}model for {dataset} dataset")

        if model_type == 'CF':
            signature = infer_signature(train, model.fit(train))
        elif model_type == 'CBF':
            signature = infer_signature(train, model.fit(tf, vectors_tokenized))
        elif model_type == 'RL':
            signature = infer_signature(train, model.fit(train)) # addapt this line to your RL model

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_type}-model",
            signature=signature,
            input_example=train,
            registered_model_name=f"{model_type}-model-test",
        )