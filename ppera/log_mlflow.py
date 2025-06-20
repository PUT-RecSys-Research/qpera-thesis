import os
from datetime import datetime

import cornac
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from mlflow.models.signature import infer_signature
from recommenders.models.cornac.cornac_utils import predict_ranking

from .rl_train_agent import ActorCritic


def log_mlflow(
    dataset,
    top_k,
    metrics,
    num_rows,
    seed,
    model,
    model_type,
    params,
    data,
    train,
    tf=None,
    vectors_tokenized=None,
    privacy=None,
    fraction_to_hide=None,
    personalization=None,
    fraction_to_change=None,
):
    MOVIELENS = "movielens"
    AMAZONSALES = "amazonsales"
    POSTRECOMMENDATIONS = "postrecommendations"

    top_k_prediction = top_k.head(10)
    print(top_k_prediction)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_filename = f"ppera/plots/top_k_predictions_{model_type}_{dataset}_{timestamp}.png"
    os.makedirs("plots", exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(top_k_prediction["itemID"], top_k_prediction["prediction"], marker="o", linestyle="-")
    plt.xlabel("ItemID")
    plt.ylabel("Prediction")
    plt.title(f"Top K Predictions for User {top_k_prediction['userID'].iloc[0]}")
    plt.grid(True)
    plt.savefig(plot_filename)
    plt.close()

    metrics_base_dir = "ppera/metrics"
    if privacy is True:
        subdir = "privacy"
        specific_metrics_dir = f"no_mod_{fraction_to_hide:.2f}"
    elif personalization is True:
        subdir = "personalization"
        specific_metrics_dir = f"no_mod_{fraction_to_change:.2f}"
    elif privacy is False and personalization is False:
        subdir = ""
        specific_metrics_dir = "clean_loop/no_mod"
    else:
        subdir = ""
        specific_metrics_dir = "None"

    full_metrics_dir = os.path.join(metrics_base_dir, subdir, specific_metrics_dir)
    os.makedirs(full_metrics_dir, exist_ok=True)
    metrics_filename = os.path.join(full_metrics_dir, f"{model_type}_{dataset}_metrics.csv")

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    if model_type == "CF":
        mlflow.set_experiment("MLflow Collaborative Filtering")
    elif model_type == "CBF":
        mlflow.set_experiment("MLflow Content Based Filtering")
    elif model_type == "RL":
        mlflow.set_experiment("MLflow Reinforcement Learning")

    if dataset == MOVIELENS:
        if num_rows is None:
            file_path = "ppera/datasets/MovieLens/merge_file.csv"
        else:
            file_path = f"ppera/datasets/MovieLens/merge_file_r{num_rows}_s{seed}.csv"
    elif dataset == AMAZONSALES:
        if num_rows is None:
            file_path = "ppera/datasets/AmazonSales/merge_file.csv"
        else:
            file_path = f"ppera/datasets/AmazonSales/merge_file_r{num_rows}_s{seed}.csv"
    elif dataset == POSTRECOMMENDATIONS:
        if num_rows is None:
            file_path = "ppera/datasets/PostRecommendations/merge_file.csv"
        else:
            file_path = f"ppera/datasets/PostRecommendations/merge_file_r{num_rows}_s{seed}.csv"

    with mlflow.start_run():
        if dataset == MOVIELENS:
            mlflow.log_artifact(file_path, artifact_path="ppera/datasets/MovieLens")
            dataset_df = mlflow.data.from_pandas(data, name="MovieLens Dataset", source=file_path)
            mlflow.log_input(dataset_df, context=f"Privacy: {privacy} | {fraction_to_hide}, Personalization: {personalization} | {fraction_to_change}")
        elif dataset == AMAZONSALES:
            mlflow.log_artifact(file_path, artifact_path="ppera/datasets/AmazonSales")
            dataset_df = mlflow.data.from_pandas(data, name="AmazonSales Dataset", source=file_path)
            mlflow.log_input(dataset_df, context=f"Privacy: {privacy} | {fraction_to_hide}, Personalization: {personalization} | {fraction_to_change}")
        elif dataset == POSTRECOMMENDATIONS:
            mlflow.log_artifact(file_path, artifact_path="ppera/datasets/PostRecommendations")
            dataset_df = mlflow.data.from_pandas(data, name="PostRecommendations Dataset", source=file_path)
            mlflow.log_input(dataset_df, context=f"Privacy: {privacy} | {fraction_to_hide}, Personalization: {personalization} | {fraction_to_change}")

        mlflow.log_artifact(plot_filename, artifact_path="plots")
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(metrics_filename, index=False)
        mlflow.log_artifact(metrics_filename, artifact_path="metrics")
        mlflow.log_params(params)

        # Filter out None values before logging to MLflow
        clean_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(clean_metrics)
        none_metrics = {k: v for k, v in metrics.items() if v is None}
        if none_metrics:
            print("Skipped logging the following None-valued metrics to MLflow:", none_metrics)
            
        mlflow.set_tag("Metrics Info", f"{model_type} model for {dataset} dataset")

        signature = None
        input_example_for_log = None

        if model_type == "CF":
            if isinstance(model, cornac.models.Recommender):
                try:
                    # For Cornac models, 'fit' is already done. We infer signature from prediction.
                    # 1. Prepare sample input for prediction:
                    if not train.empty:
                        user_col = params.get("col_user", "userID")
                        item_col = params.get("col_item", "itemID")

                        sample_users = train[user_col].unique()[:2]  # Take 2 sample users
                        sample_items = train[item_col].unique()[:5]  # Take 5 sample items

                        # Create all user-item pairs for the sample
                        sample_input_data_list = []
                        for u in sample_users:
                            for i in sample_items:
                                sample_input_data_list.append({user_col: u, item_col: i})

                        if not sample_input_data_list:
                            print("MLflow: Could not create sample input for CF signature from train_data_df.")
                        else:
                            sample_input_df_for_cf = pd.DataFrame(sample_input_data_list)

                            # 2. Get sample predictions
                            sample_predictions_df_cf = predict_ranking(model, sample_input_df_for_cf, usercol=user_col, itemcol=item_col, remove_seen=True)

                            if not sample_predictions_df_cf.empty:
                                signature_input_cf = sample_input_df_for_cf[[user_col]].drop_duplicates().reset_index(drop=True)
                                signature_output_cf = sample_predictions_df_cf[[user_col, item_col, params.get("col_prediction", "prediction")]]

                                signature = infer_signature(signature_input_cf, signature_output_cf)
                                input_example_for_log = signature_input_cf.head(5)  # A small sample of users
                                print("MLflow: Inferred signature for CF (Cornac) model using predict_ranking.")
                            else:
                                print("MLflow: Sample predictions for CF were empty. Skipping signature.")
                    else:
                        print("MLflow: train_data_df is empty. Skipping CF signature inference.")
                except Exception as e:
                    print(f"MLflow: Error during CF (Cornac) signature inference: {e}")
                    signature = None  # Ensure signature is None on error
        elif model_type == "CBF":
            print(f"typ TRAIN {type(train)}")
            input_example_for_log = train.head(5)
            signature = infer_signature(train, model.fit(tf, vectors_tokenized))
        elif model_type == "RL":
            if isinstance(model, ActorCritic):
                try:
                    # 1. Create sample input tensors
                    sample_state_tensor = torch.randn(1, model.state_dim)
                    sample_act_mask_tensor = torch.ones(1, model.act_dim, dtype=torch.bool)

                    # 2. Move sample input to the model's device
                    model_device = next(model.parameters()).device
                    sample_input_tuple_pytorch = (sample_state_tensor.to(model_device), sample_act_mask_tensor.to(model_device))

                    # 3. Get sample prediction from the model
                    with torch.no_grad():
                        model.eval()
                        # sample_prediction_tuple_rl = model(sample_input_tuple_for_rl)
                        sample_pred_probs_tensor, sample_pred_value_tensor = model(sample_input_tuple_pytorch)

                    model_input_for_sig = sample_state_tensor.cpu().numpy()

                    # 4. Format output for infer_signature (if it's a tuple)
                    model_output_for_sig = {
                        "action_probabilities": sample_pred_probs_tensor.cpu().numpy(),
                        "state_values": sample_pred_value_tensor.cpu().numpy(),
                    }

                    # 5. Infer signature
                    signature = infer_signature(
                        model_input=model_input_for_sig,  # Use NumPy array or dict of NumPy arrays
                        model_output=model_output_for_sig,
                    )

                    # 6. Prepare input_example for mlflow.pytorch.log_model
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
                if model_type == "RL":
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path=f"{model_type}-model",
                        signature=signature,
                        input_example=input_example_for_log,
                        registered_model_name=f"{model_type}-model-{dataset}",
                    )
                elif model_type in "CF":
                    mlflow.pyfunc.log_model(
                        python_model=model,  # This requires a PyFunc wrapper for Cornac
                        artifact_path=f"{model_type}-model",
                        signature=signature,
                        input_example=input_example_for_log,
                        # code_path=[os.path.dirname(cornac.__file__)], # May need cornac code path
                        registered_model_name=f"{model_type}-model-{dataset}",
                    )
                elif model_type in "CBF":
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=f"{model_type}-model",
                        signature=signature,
                        input_example=input_example_for_log,
                        registered_model_name=f"{model_type}-model-{dataset}",
                    )
                print(f"MLflow: Logged {model_type} model with signature.")

            except Exception as e:
                print(f"MLflow: Error logging model WITH signature for {model_type}: {e}.")
                print("MLflow: Attempting to log model WITHOUT signature as a fallback.")

                if model_type == "RL":
                    mlflow.pytorch.log_model(pytorch_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-{dataset}")
                elif model_type in ["CF", "CBF"]:
                    mlflow.sklearn.log_model(sk_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-{dataset}")
                else:
                    print(f"MLflow: Model type {model_type} not recognized for fallback logging.")

                print(f"MLflow: Logged {model_type} model WITHOUT signature (fallback).")

        elif model:
            print(f"MLflow: Signature not inferred for {model_type}. Logging model without signature.")
            if model_type == "RL":
                mlflow.pytorch.log_model(pytorch_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-{dataset}")
            elif model_type in ["CF", "CBF"]:
                mlflow.sklearn.log_model(sk_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-test")
            print(f"MLflow: Logged {model_type} model without signature.")
        else:
            print(f"MLflow: Model object for {model_type} not available. Skipping model logging entirely.")
