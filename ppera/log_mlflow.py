import os
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
from mlflow.models.signature import infer_signature

def log_mlflow(dataset, top_k, metrics, num_rows, seed, model, model_type, params, data, train, tf=None, vectors_tokenized=None):
    # Assuming 'top_k' and other necessary variables are already defined

    # Generate top_k_prediction DataFrame
    top_k_prediction = top_k.head(10)
    print(top_k_prediction)

    # Create a unique filename for the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'plots/top_k_predictions_{dataset}_{timestamp}.png'
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

    # Define metrics

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    if model_type == 'CF':
        mlflow.set_experiment("MLflow Collaborative Filtering")
    elif model_type == 'CBF':
        mlflow.set_experiment("MLflow Content Based Filtering")

    if dataset == 'movielens':
        file_path = f"datasets/MovieLens/merge_file_r{num_rows}_s{seed}.csv"
    elif dataset == 'amazonsales':
        file_path = f"datasets/AmazonSales/merge_file_r{num_rows}_s{seed}.csv"
    elif dataset == 'postrecommendations':
        file_path = f"datasets/PostRecommendations/merge_file_r{num_rows}_s{seed}.csv"

    # Start an MLflow run
    with mlflow.start_run():
        # Log dataset
        if dataset == 'movielens':
            mlflow.log_artifact(file_path, artifact_path="datasets/MovieLens")
            # Log the dataset to make it appear in the "Dataset" column
            dataset = mlflow.data.from_pandas(data, name="MovieLens Dataset", source=file_path)
            mlflow.log_input(dataset, context="test")  # 'training' or 'evaluation'
        elif dataset == 'amazonsales':
            mlflow.log_artifact(file_path, artifact_path="datasets/AmazonSales")
            # Log the dataset to make it appear in the "Dataset" column
            dataset = mlflow.data.from_pandas(data, name="AmazonSales Dataset", source=file_path)
            mlflow.log_input(dataset, context="test")  # 'training' or 'evaluation'
        elif dataset == 'postrecommendations':
            mlflow.log_artifact(file_path, artifact_path="datasets/PostRecommendations")
            # Log the dataset to make it appear in the "Dataset" column
            dataset = mlflow.data.from_pandas(data, name="PostRecommendations Dataset", source=file_path)
            mlflow.log_input(dataset, context="test")

        mlflow.log_artifact(plot_filename, artifact_path='plots')
        # Optionally, log the data used for plotting
        # top_k_prediction.to_csv('top_k_prediction.csv', index=False)
        # mlflow.log_artifact('top_k_prediction.csv')

        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metrics(metrics)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Metrics Info", f"{model_type}model for {dataset} dataset")

        # Infer the model signature
        if model_type == 'CF':
            signature = infer_signature(train, model.fit(train))
        elif model_type == 'CBF':
            signature = infer_signature(train, model.fit(tf, vectors_tokenized))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_type}-model",
            signature=signature,
            input_example=train,
            registered_model_name=f"{model_type}-model-test",
        )