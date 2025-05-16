import os
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
from mlflow.models.signature import infer_signature

import pandas as pd
import torch
from rl_train_agent import ActorCritic

def log_mlflow(dataset, top_k, metrics, num_rows, seed, model, model_type, params, data, train, tf=None, vectors_tokenized=None):

    MOVIELENS = 'movielens'
    AMAZONSALES = 'amazonsales'
    POSTRECOMMENDATIONS = 'postrecommendations'

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

    if dataset == MOVIELENS:
        file_path = f"datasets/MovieLens/merge_file_r{num_rows}_s{seed}.csv"
    elif dataset == AMAZONSALES:
        file_path = f"datasets/AmazonSales/merge_file_r{num_rows}_s{seed}.csv"
    elif dataset == POSTRECOMMENDATIONS:
        file_path = f"datasets/PostRecommendations/merge_file_r{num_rows}_s{seed}.csv"

    with mlflow.start_run():
        if dataset == MOVIELENS:
            mlflow.log_artifact(file_path, artifact_path="datasets/MovieLens")
            dataset_df = mlflow.data.from_pandas(data, name="MovieLens Dataset", source=file_path)
            mlflow.log_input(dataset_df, context="test")
        elif dataset == AMAZONSALES:
            mlflow.log_artifact(file_path, artifact_path="datasets/AmazonSales")
            dataset_df = mlflow.data.from_pandas(data, name="AmazonSales Dataset", source=file_path)
            mlflow.log_input(dataset_df, context="test")
        elif dataset == POSTRECOMMENDATIONS:
            mlflow.log_artifact(file_path, artifact_path="datasets/PostRecommendations")
            dataset_df = mlflow.data.from_pandas(data, name="PostRecommendations Dataset", source=file_path)
            mlflow.log_input(dataset_df, context="test")

        mlflow.log_artifact(plot_filename, artifact_path='plots')
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("Metrics Info", f"{model_type} model for {dataset} dataset")

        signature = None
        input_example_for_log = None

        if model_type == 'CF':
            signature = infer_signature(train, model.fit(train))
        elif model_type == 'CBF':
            signature = infer_signature(train, model.fit(tf, vectors_tokenized))
        elif model_type == 'RL':
            if isinstance(model, ActorCritic):
                try:
                    # 1. Create sample input tensors
                    # sample_state_for_sig = torch.randn(1, model.state_dim)
                    # sample_act_mask_for_sig = torch.ones(1, model.act_dim, dtype=torch.bool)
                    sample_state_tensor = torch.randn(1, model.state_dim)
                    sample_act_mask_tensor = torch.ones(1, model.act_dim, dtype=torch.bool)
                    
                    # 2. Move sample input to the model's device
                    model_device = next(model.parameters()).device
                    # sample_input_tuple_for_rl = (
                    #     sample_state_for_sig.to(model_device),
                    #     sample_act_mask_for_sig.to(model_device)
                    # )
                    sample_input_tuple_pytorch = (
                        sample_state_tensor.to(model_device),
                        sample_act_mask_tensor.to(model_device)
                    )
                    
                    # 3. Get sample prediction from the model
                    with torch.no_grad():
                        model.eval()
                        # sample_prediction_tuple_rl = model(sample_input_tuple_for_rl)
                        sample_pred_probs_tensor, sample_pred_value_tensor = model(sample_input_tuple_pytorch)
                    
                    model_input_for_sig = sample_state_tensor.cpu().numpy()
                    
                    # 4. Format output for infer_signature (if it's a tuple)
                    # sample_output_for_sig_rl = {
                    #     "action_probabilities": sample_prediction_tuple_rl[0].cpu().numpy(),
                    #     "state_values": sample_prediction_tuple_rl[1].cpu().numpy()
                    # }
                    model_output_for_sig = {
                        "action_probabilities": sample_pred_probs_tensor.cpu().numpy(),
                        "state_values": sample_pred_value_tensor.cpu().numpy()
                    }

                    # 5. Infer signature
                    # signature = infer_signature(
                    #     model_input=sample_input_tuple_for_rl[0],
                    #     model_output=sample_output_for_sig_rl
                    # )
                    signature = infer_signature(
                        model_input=model_input_for_sig, # Use NumPy array or dict of NumPy arrays
                        model_output=model_output_for_sig
                    )
                    
                    # 6. Prepare input_example for mlflow.pytorch.log_model
                    # input_example_for_log = sample_input_tuple_for_rl
                    # input_example_for_log = sample_input_tuple_for_rl[0]
                    # input_example_for_log = {
                    #     "state": sample_input_tuple_for_rl[0],
                    #     "act_mask": sample_input_tuple_for_rl[1]
                    # }

                    # input_example_for_log = sample_input_tuple_pytorch[0]
                    input_example_for_log = pd.DataFrame(sample_input_tuple_pytorch[0].cpu().numpy())

                    print("MLflow: Inferred signature for RL model.")
                except AttributeError as e:
                    print(f"MLflow: Error during RL signature inference (likely missing state_dim/act_dim on model or model not ActorCritic): {e}")
                except Exception as e:
                    print(f"MLflow: General error during RL signature inference: {e}")
            else:
                print(f"MLflow: model is not an instance of ActorCritic for RL type. Got {type(model)}. Skipping signature.")

        if model and signature:
            try:
                if model_type == 'RL':
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path=f"{model_type}-model",
                        signature=signature,
                        input_example=input_example_for_log,
                        registered_model_name=f"{model_type}-model-{dataset}",
                    )
                elif model_type in ['CF', 'CBF']:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=f"{model_type}-model",
                        signature=signature,
                        input_example=train,
                        registered_model_name=f"{model_type}-model-test",
                    )
                print(f"MLflow: Logged {model_type} model with signature.")

            except Exception as e:
                print(f"MLflow: Error logging model WITH signature for {model_type}: {e}.")
                print("MLflow: Attempting to log model WITHOUT signature as a fallback.")

                if model_type == 'RL':
                    mlflow.pytorch.log_model(pytorch_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-{dataset}")
                elif model_type in ['CF', 'CBF']:
                    mlflow.sklearn.log_model(sk_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-test")
                else:
                    print(f"MLflow: Model type {model_type} not recognized for fallback logging.")

                print(f"MLflow: Logged {model_type} model WITHOUT signature (fallback).")

        elif model:
            print(f"MLflow: Signature not inferred for {model_type}. Logging model without signature.")
            if model_type == 'RL':
                mlflow.pytorch.log_model(pytorch_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-{dataset}")
            elif model_type in ['CF', 'CBF']:
                mlflow.sklearn.log_model(sk_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-test")
            print(f"MLflow: Logged {model_type} model without signature.")
        else:
            print(f"MLflow: Model object for {model_type} not available. Skipping model logging entirely.")