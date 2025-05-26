# import os
# import cornac
# import matplotlib.pyplot as plt
# import mlflow
# from datetime import datetime
# from mlflow.models.signature import infer_signature
# import numpy as np
# from recommenders.models.cornac.cornac_utils import predict_ranking

# import pandas as pd
# import torch
# from rl_train_agent import ActorCritic, ActorCriticPyFuncWrapper

# def log_mlflow(dataset, top_k, metrics, num_rows, seed, model, model_type, params, data, train, tf=None, vectors_tokenized=None):

#     MOVIELENS = 'movielens'
#     AMAZONSALES = 'amazonsales'
#     POSTRECOMMENDATIONS = 'postrecommendations'

#     top_k_prediction = top_k.head(10)
#     print(top_k_prediction)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plot_filename = f'plots/top_k_predictions_{model_type}_{dataset}_{timestamp}.png'
#     os.makedirs('plots', exist_ok=True)

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(top_k_prediction['itemID'], top_k_prediction['prediction'], marker='o', linestyle='-')
#     plt.xlabel('ItemID')
#     plt.ylabel('Prediction')
#     plt.title(f"Top K Predictions for User {top_k_prediction['userID'].iloc[0]}")
#     plt.grid(True)
#     plt.savefig(plot_filename)
#     plt.close()

#     mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

#     if model_type == 'CF':
#         mlflow.set_experiment("MLflow Collaborative Filtering")
#     elif model_type == 'CBF':
#         mlflow.set_experiment("MLflow Content Based Filtering")
#     elif model_type == 'RL':
#         mlflow.set_experiment("MLflow Reinforcement Learning")

#     if dataset == MOVIELENS:
#         file_path = f"datasets/MovieLens/merge_file_r{num_rows}_s{seed}.csv"
#     elif dataset == AMAZONSALES:
#         file_path = f"datasets/AmazonSales/merge_file_r{num_rows}_s{seed}.csv"
#     elif dataset == POSTRECOMMENDATIONS:
#         file_path = f"datasets/PostRecommendations/merge_file_r{num_rows}_s{seed}.csv"

#     with mlflow.start_run():
#         if dataset == MOVIELENS:
#             mlflow.log_artifact(file_path, artifact_path="datasets/MovieLens")
#             dataset_df = mlflow.data.from_pandas(data, name="MovieLens Dataset", source=file_path)
#             mlflow.log_input(dataset_df, context="test")
#         elif dataset == AMAZONSALES:
#             mlflow.log_artifact(file_path, artifact_path="datasets/AmazonSales")
#             dataset_df = mlflow.data.from_pandas(data, name="AmazonSales Dataset", source=file_path)
#             mlflow.log_input(dataset_df, context="test")
#         elif dataset == POSTRECOMMENDATIONS:
#             mlflow.log_artifact(file_path, artifact_path="datasets/PostRecommendations")
#             dataset_df = mlflow.data.from_pandas(data, name="PostRecommendations Dataset", source=file_path)
#             mlflow.log_input(dataset_df, context="test")

#         mlflow.log_artifact(plot_filename, artifact_path='plots')
#         mlflow.log_params(params)
#         mlflow.log_metrics(metrics)
#         mlflow.set_tag("Metrics Info", f"{model_type} model for {dataset} dataset")

#         signature = None
#         input_example_for_log = None

#         if model_type == 'CF':
#             if isinstance(model, cornac.models.Recommender):
#                 try:
#                     # For Cornac models, 'fit' is already done. We infer signature from prediction.
#                     # 1. Prepare sample input for prediction:
#                     if not train.empty:

#                         user_col = params.get("col_user", "userID")
#                         item_col = params.get("col_item", "itemID")
                    
                        
#                         sample_users = train[user_col].unique()[:2] # Take 2 sample users
#                         sample_items = train[item_col].unique()[:5] # Take 5 sample items
                        
#                         # Create all user-item pairs for the sample
#                         sample_input_data_list = []
#                         for u in sample_users:
#                             for i in sample_items:
#                                 sample_input_data_list.append({user_col: u, item_col: i})
                        
#                         if not sample_input_data_list:
#                             print("MLflow: Could not create sample input for CF signature from train_data_df.")
#                         else:
#                             sample_input_df_for_cf = pd.DataFrame(sample_input_data_list)

#                             # 2. Get sample predictions
#                             sample_predictions_df_cf = predict_ranking(
#                                 model, 
#                                 sample_input_df_for_cf,
#                                 usercol=user_col, 
#                                 itemcol=item_col, 
#                                 remove_seen=True
#                             )

#                             if not sample_predictions_df_cf.empty:
#                                 signature_input_cf = sample_input_df_for_cf[[user_col]].drop_duplicates().reset_index(drop=True)
#                                 signature_output_cf = sample_predictions_df_cf[[user_col, item_col, params.get("col_prediction", "prediction")]]


#                                 signature = infer_signature(signature_input_cf, signature_output_cf)
#                                 input_example_for_log = signature_input_cf.head(5) # A small sample of users
#                                 print("MLflow: Inferred signature for CF (Cornac) model using predict_ranking.")
#                             else:
#                                 print("MLflow: Sample predictions for CF were empty. Skipping signature.")
#                     else:
#                         print("MLflow: train_data_df is empty. Skipping CF signature inference.")
#                 except Exception as e:
#                     print(f"MLflow: Error during CF (Cornac) signature inference: {e}")
#                     signature = None # Ensure signature is None on error
#         elif model_type == 'CBF':
#             signature = infer_signature(train, model.fit(tf, vectors_tokenized))
#         # elif model_type == 'RL':
#         #     if isinstance(model, ActorCritic):
#         #         try:
#         #             # 1. Create sample input tensors
#         #             # sample_state_for_sig = torch.randn(1, model.state_dim)
#         #             # sample_act_mask_for_sig = torch.ones(1, model.act_dim, dtype=torch.bool)
#         #             sample_state_tensor = torch.randn(1, model.state_dim)
#         #             sample_act_mask_tensor = torch.ones(1, model.act_dim, dtype=torch.bool)
                    
#         #             # 2. Move sample input to the model's device
#         #             model_device = next(model.parameters()).device
#         #             # sample_input_tuple_for_rl = (
#         #             #     sample_state_for_sig.to(model_device),
#         #             #     sample_act_mask_for_sig.to(model_device)
#         #             # )
#         #             sample_input_tuple_pytorch = (
#         #                 sample_state_tensor.to(model_device),
#         #                 sample_act_mask_tensor.to(model_device)
#         #             )
                    
#         #             # 3. Get sample prediction from the model
#         #             with torch.no_grad():
#         #                 model.eval()
#         #                 # sample_prediction_tuple_rl = model(sample_input_tuple_for_rl)
#         #                 sample_pred_probs_tensor, sample_pred_value_tensor = model(sample_input_tuple_pytorch)
                    
#         #             #model_input_for_sig = sample_state_tensor.cpu().numpy()
                    
#         #             # 4. Format output for infer_signature (if it's a tuple)
#         #             # sample_output_for_sig_rl = {
#         #             #     "action_probabilities": sample_prediction_tuple_rl[0].cpu().numpy(),
#         #             #     "state_values": sample_prediction_tuple_rl[1].cpu().numpy()
#         #             # }
#         #             # model_output_for_sig = {
#         #             #     "action_probabilities": sample_pred_probs_tensor.cpu().numpy(),
#         #             #     "state_values": sample_pred_value_tensor.cpu().numpy()
#         #             # }
#         #             signature_model_input = {
#         #                 "state": sample_input_tuple_pytorch[0].cpu().numpy(),
#         #                 "act_mask": sample_input_tuple_pytorch[1].cpu().numpy()
#         #             }
#         #             signature_model_output = {
#         #                 "action_probabilities": sample_pred_probs_tensor.cpu().numpy(),
#         #                 "state_values": sample_pred_value_tensor.cpu().numpy()
#         #             }
#         #             # 5. Infer signature
#         #             # signature = infer_signature(
#         #             #     model_input=sample_input_tuple_for_rl[0],
#         #             #     model_output=sample_output_for_sig_rl
#         #             # )
#         #             # signature = infer_signature(
#         #             #     model_input=model_input_for_sig, # Use NumPy array or dict of NumPy arrays
#         #             #     model_output=model_output_for_sig
#         #             # )
#         #             signature = infer_signature(
#         #                 model_input=signature_model_input, # Use the dictionary of named inputs
#         #                 model_output=signature_model_output
#         #             )

#         #             # 6. Prepare input_example for mlflow.pytorch.log_model
#         #             # input_example_for_log = sample_input_tuple_for_rl
#         #             # input_example_for_log = sample_input_tuple_for_rl[0]
#         #             # input_example_for_log = {
#         #             #     "state": sample_input_tuple_for_rl[0],
#         #             #     "act_mask": sample_input_tuple_for_rl[1]
#         #             # }

#         #             # input_example_for_log = sample_input_tuple_pytorch[0]
#         #             # input_example_for_log = pd.DataFrame(sample_input_tuple_pytorch[0].cpu().numpy())
#         #             # input_example_for_log = (
#         #             #     sample_input_tuple_pytorch[0].cpu().numpy(), # state as numpy array
#         #             #     sample_input_tuple_pytorch[1].cpu().numpy()  # act_mask as numpy array
#         #             # )
#         #             input_example_for_log = {
#         #                 "state": sample_input_tuple_pytorch[0].cpu().numpy(),
#         #                 "act_mask": sample_input_tuple_pytorch[1].cpu().numpy()
#         #             }

#         #             print("MLflow: Inferred signature for RL model.")
#         #         except AttributeError as e:
#         #             print(f"MLflow: Error during RL signature inference (likely missing state_dim/act_dim on model or model not ActorCritic): {e}")
#         #         except Exception as e:
#         #             print(f"MLflow: General error during RL signature inference: {e}")
#         #     else:
#         #         print(f"MLflow: model is not an instance of ActorCritic for RL type. Got {type(model)}. Skipping signature.")
#         elif model_type == 'RL':
#             if isinstance(model, ActorCritic): # model is your original ActorCritic instance
#                 try:
#                     # 1. Instantiate your wrapper
#                     wrapped_model = ActorCriticPyFuncWrapper(model) # model is your trained ActorCritic

#                     # 2. Define Signature and Input Example for the WRAPPER
#                     # The wrapper expects a DataFrame with columns like 'state_0', 'state_1', ..., 'mask_0', ...
#                     # The exact column names depend on your _preprocess_input logic.
#                     # Let's assume state_dim and act_dim are accessible from your original model.
#                     state_dim = model.state_dim
#                     act_dim = model.act_dim

#                     sample_df_input_data = {}
#                     # For state features (assuming they are individual columns)
#                     for i in range(state_dim):
#                         sample_df_input_data[f'state_{i}'] = [np.random.rand()] # Single row example
#                     # For mask features (assuming they are individual columns)
#                     for i in range(act_dim):
#                         sample_df_input_data[f'mask_{i}'] = [True] # Single row example

#                     signature_model_input_df = pd.DataFrame(sample_df_input_data)

#                     # The output of the wrapper's predict method is a dictionary
#                     signature_model_output_dict = {
#                         "action_probabilities": np.random.rand(1, act_dim).astype(np.float32), # Batch size 1
#                         "state_values": np.random.rand(1, 1).astype(np.float32)            # Batch size 1
#                     }

#                     # signature_for_wrapper = infer_signature(
#                     #     model_input=signature_model_input_df,
#                     #     model_output=pd.DataFrame(signature_model_output_dict) # infer_signature often prefers DataFrame output
#                     # )
#                     signature_for_wrapper = infer_signature(
#                         model_input=signature_model_input_df,
#                         model_output=signature_model_output_dict # <--- USE DICTIONARY HERE
#                     )
#                     # Or, if infer_signature handles dict output well for pyfunc:
#                     # signature_for_wrapper = infer_signature(signature_model_input_df, signature_model_output_dict)


#                     input_example_for_wrapper_log = signature_model_input_df.head(1)

#                     print("MLflow: Prepared signature and input example for RL PyFunc wrapper.")

#                 except Exception as e:
#                     print(f"MLflow: Error during RL PyFunc wrapper logging: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     # Fallback to logging the raw PyTorch model without input_example if needed
#                     # (though this won't use your wrapper for serving)
#                     print("MLflow: Attempting to log raw PyTorch model WITHOUT input_example as a fallback.")
#                     try:
#                         # Create signature for the RAW model (expecting dict input from pyfunc layer)
#                         raw_signature_model_input = {
#                             "state": np.random.randn(1, model.state_dim).astype(np.float32),
#                             "act_mask": np.ones((1, model.act_dim), dtype=bool)
#                         }
#                         raw_signature_model_output = {
#                             "action_probabilities": np.random.rand(1, model.act_dim).astype(np.float32),
#                             "state_values": np.random.rand(1, 1).astype(np.float32)
#                         }
#                         raw_model_signature = infer_signature(raw_signature_model_input, raw_signature_model_output)

#                         mlflow.pytorch.log_model(
#                             pytorch_model=model,
#                             artifact_path=f"{model_type}-model-pytorch-raw",
#                             signature=raw_model_signature, # Signature for pyfunc layer
#                             input_example=None,          # Avoids PyTorch flavor's input_example issue
#                             registered_model_name=f"{model_type}-model-{dataset}",
#                         )
#                         print(f"MLflow: Logged raw RL PyTorch model (no input_example) successfully as fallback.")
#                     except Exception as fallback_e:
#                         print(f"MLflow: Error during fallback logging of raw PyTorch model: {fallback_e}")

#             else:
#                 print(f"MLflow: model is not an instance of ActorCritic for RL type. Got {type(model)}. Skipping signature/PyFunc logging.")

#         if model and signature:
#             try:
#                 if model_type == 'RL':
#                     # mlflow.pytorch.log_model(
#                     #     pytorch_model=model,
#                     #     artifact_path=f"{model_type}-model",
#                     #     signature=signature,
#                     #     input_example=input_example_for_log,
#                     #     registered_model_name=f"{model_type}-model-{dataset}",
#                     # )
#                     mlflow.pytorch.log_model(
#                         pytorch_model=wrapped_model,
#                         artifact_path=f"{model_type}-model",
#                         signature=signature_for_wrapper, # Now signature reflects dict input
#                         input_example=input_example_for_wrapper_log , # Now input_example is a tuple of numpy arrays
#                         registered_model_name=f"{model_type}-model-{dataset}",
#                     )
#                 elif model_type in 'CF':
#                     mlflow.pyfunc.log_model(
#                             python_model=model, # This requires a PyFunc wrapper for Cornac
#                             artifact_path=f"{model_type}-model",
#                             signature=signature,
#                             input_example=input_example_for_log,
#                             # code_path=[os.path.dirname(cornac.__file__)], # May need cornac code path
#                             registered_model_name=f"{model_type}-model-{dataset}",
#                         )
#                 elif model_type in 'CBF':
#                     mlflow.sklearn.log_model(
#                         sk_model=model,
#                         artifact_path=f"{model_type}-model",
#                         signature=signature,
#                         input_example=train,
#                         registered_model_name=f"{model_type}-model-{dataset}",
#                     )
#                 print(f"MLflow: Logged {model_type} model with signature.")

#             except Exception as e:
#                 print(f"MLflow: Error logging model WITH signature for {model_type}: {e}.")
#                 print("MLflow: Attempting to log model WITHOUT signature as a fallback.")

#                 if model_type == 'RL':
#                     mlflow.pytorch.log_model(pytorch_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-{dataset}")
#                 elif model_type in ['CF', 'CBF']:
#                     mlflow.sklearn.log_model(sk_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-{dataset}")
#                 else:
#                     print(f"MLflow: Model type {model_type} not recognized for fallback logging.")

#                 print(f"MLflow: Logged {model_type} model WITHOUT signature (fallback).")

#         elif model:
#             print(f"MLflow: Signature not inferred for {model_type}. Logging model without signature.")
#             if model_type == 'RL':
#                 mlflow.pytorch.log_model(pytorch_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-{dataset}")
#             elif model_type in ['CF', 'CBF']:
#                 mlflow.sklearn.log_model(sk_model=model, artifact_path=f"{model_type}-model", registered_model_name=f"{model_type}-model-test")
#             print(f"MLflow: Logged {model_type} model without signature.")
#         else:
#             print(f"MLflow: Model object for {model_type} not available. Skipping model logging entirely.")

import os
import cornac
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
from mlflow.models.signature import infer_signature
import numpy as np
from recommenders.models.cornac.cornac_utils import predict_ranking

import pandas as pd
import torch
# Ensure these are correctly imported based on your file structure
from rl_train_agent import ActorCritic, ActorCriticPyFuncWrapper # Assuming they are in this file

def log_mlflow(dataset, top_k, metrics, num_rows, seed, model, model_type, params, data, train, tf=None, vectors_tokenized=None):

    # ... (initial setup, plotting, mlflow.set_experiment, file_path logic remains the same) ...
    MOVIELENS = 'movielens'
    AMAZONSALES = 'amazonsales'
    POSTRECOMMENDATIONS = 'postrecommendations'

    top_k_prediction = top_k.head(10)
    # print(top_k_prediction) # You can uncomment this for debugging

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'plots/top_k_predictions_{model_type}_{dataset}_{timestamp}.png'
    os.makedirs('plots', exist_ok=True)

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
    else:
        file_path = None # Handle cases where dataset name might not match


    with mlflow.start_run():
        if file_path and os.path.exists(file_path):
            if dataset == MOVIELENS:
                mlflow.log_artifact(file_path, artifact_path="datasets/MovieLens")
            elif dataset == AMAZONSALES:
                mlflow.log_artifact(file_path, artifact_path="datasets/AmazonSales")
            elif dataset == POSTRECOMMENDATIONS:
                mlflow.log_artifact(file_path, artifact_path="datasets/PostRecommendations")
            
            # Common dataset logging
            dataset_name_mlflow = f"{dataset.capitalize()} Dataset"
            dataset_df_mlflow = mlflow.data.from_pandas(data, name=dataset_name_mlflow, source=file_path)
            mlflow.log_input(dataset_df_mlflow, context="test") # Using 'test' as context, adjust if needed
        elif file_path:
            print(f"Warning: Dataset file not found at {file_path}, not logging as artifact.")


        mlflow.log_artifact(plot_filename, artifact_path='plots')
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("Metrics Info", f"{model_type} model for {dataset} dataset")

        # Initialize signature and input_example at the function level
        # These will be updated if model logging with signature is successful
        model_signature = None # Renamed from 'signature' to avoid confusion with infer_signature
        model_input_example = None # Renamed from 'input_example_for_log'

        # --- Model Specific Logging ---
        if model_type == 'CF':
            # ... (your CF logging logic, ensure it sets model_signature and model_input_example if successful) ...
            if isinstance(model, cornac.models.Recommender):
                try:
                    if not train.empty:
                        user_col = params.get("col_user", "userID")
                        item_col = params.get("col_item", "itemID")
                        sample_users = train[user_col].unique()[:2]
                        sample_items = train[item_col].unique()[:5]
                        sample_input_data_list = [{'user': u, 'item': i} for u in sample_users for i in sample_items] # Simplified
                        if sample_input_data_list:
                            sample_input_df_for_cf = pd.DataFrame(sample_input_data_list)
                            sample_predictions_df_cf = predict_ranking(model, sample_input_df_for_cf, usercol='user', itemcol='item', remove_seen=True)
                            if not sample_predictions_df_cf.empty:
                                signature_input_cf = sample_input_df_for_cf[['user']].drop_duplicates().reset_index(drop=True)
                                signature_output_cf = sample_predictions_df_cf[['user', 'item', params.get("col_prediction", "prediction")]]
                                model_signature = infer_signature(signature_input_cf, signature_output_cf)
                                model_input_example = signature_input_cf.head(5)
                                print("MLflow: Inferred signature for CF (Cornac) model.")
                                mlflow.pyfunc.log_model( # Example: Assuming Cornac needs pyfunc
                                    python_model=model, # May need a Cornac wrapper
                                    artifact_path=f"{model_type}-model",
                                    signature=model_signature,
                                    input_example=model_input_example,
                                    registered_model_name=f"{model_type}-model-{dataset}",
                                )
                                print(f"MLflow: Logged {model_type} model with signature.")
                            else: print("MLflow: Sample predictions for CF were empty.")
                        else: print("MLflow: Could not create sample input for CF.")
                    else: print("MLflow: train_data_df is empty for CF.")
                except Exception as e:
                    print(f"MLflow: Error during CF (Cornac) signature inference/logging: {e}")
                    model_signature = None
        elif model_type == 'CBF':
            # ... (your CBF logging, ensure it sets model_signature and model_input_example) ...
            try:
                # Assuming model.fit() for CBF returns something usable for signature
                # This part needs to be specific to your CBF model
                # For example, if it's an sklearn model:
                # model.fit(X_train_cbf, y_train_cbf) # Assume X_train_cbf, y_train_cbf are prepared
                # model_signature = infer_signature(X_train_cbf, model.predict(X_train_cbf))
                # model_input_example = X_train_cbf[:5]
                # print("MLflow: Inferred signature for CBF model.")
                # mlflow.sklearn.log_model(
                #     sk_model=model,
                #     artifact_path=f"{model_type}-model",
                #     signature=model_signature,
                #     input_example=model_input_example,
                #     registered_model_name=f"{model_type}-model-{dataset}",
                # )
                # print(f"MLflow: Logged {model_type} model with signature.")
                pass # Placeholder - implement CBF specific logging
            except Exception as e:
                print(f"MLflow: Error during CBF signature inference/logging: {e}")
                model_signature = None

        elif model_type == 'RL':
            if isinstance(model, ActorCritic):
                wrapped_model = None
                temp_signature = None
                temp_input_example = None
                logging_successful = False

                # --- Attempt 1: Log with PyFunc Wrapper ---
                try:
                    print("MLflow: Attempting to log RL model with PyFunc wrapper.")
                    wrapped_model = ActorCriticPyFuncWrapper(model)
                    state_dim = model.state_dim
                    act_dim = model.act_dim

                    sample_df_input_data = {}
                    for i in range(state_dim): sample_df_input_data[f'state_{i}'] = [np.random.rand()]
                    for i in range(act_dim): sample_df_input_data[f'mask_{i}'] = [True]
                    signature_model_input_df = pd.DataFrame(sample_df_input_data)

                    signature_model_output_dict = {
                        "action_probabilities": np.random.rand(1, act_dim).astype(np.float32),
                        "state_values": np.random.rand(1, 1).astype(np.float32)
                    }
                    temp_signature = infer_signature(
                        model_input=signature_model_input_df,
                        model_output=signature_model_output_dict
                    )
                    temp_input_example = signature_model_input_df.head(1)
                    print("MLflow: Prepared signature and input example for RL PyFunc wrapper.")

                    mlflow.pyfunc.log_model(
                        python_model=wrapped_model,
                        artifact_path=f"{model_type}-model-pyfunc",
                        signature=temp_signature,
                        input_example=temp_input_example,
                        registered_model_name=f"{model_type}-model-{dataset}",
                        # You might need code_paths:
                        # code_path=[os.path.dirname(os.path.abspath(rl_train_agent.__file__))] # if rl_train_agent is a module
                    )
                    print(f"MLflow: Logged RL model as PyFunc PythonModel successfully.")
                    model_signature = temp_signature # Update main signature
                    model_input_example = temp_input_example # Update main input example
                    logging_successful = True

                except Exception as e_wrapper:
                    print(f"MLflow: Error during RL PyFunc wrapper logging: {e_wrapper}")
                    import traceback
                    traceback.print_exc()
                    logging_successful = False # Explicitly set

                # --- Attempt 2 (Fallback): Log raw PyTorch model if wrapper failed ---
                if not logging_successful:
                    try:
                        print("MLflow: Attempting to log raw PyTorch model (no input_example) as fallback.")
                        raw_signature_model_input = {
                            "state": np.random.randn(1, model.state_dim).astype(np.float32),
                            "act_mask": np.ones((1, model.act_dim), dtype=bool)
                        }
                        raw_signature_model_output = {
                            "action_probabilities": np.random.rand(1, model.act_dim).astype(np.float32),
                            "state_values": np.random.rand(1, 1).astype(np.float32)
                        }
                        temp_signature = infer_signature(raw_signature_model_input, raw_signature_model_output)
                        temp_input_example = None # For raw pytorch log

                        mlflow.pytorch.log_model(
                            pytorch_model=model,
                            artifact_path=f"{model_type}-model-pytorch-raw",
                            signature=temp_signature,
                            input_example=temp_input_example, # This is None
                            registered_model_name=f"{model_type}-model-{dataset}",
                        )
                        print(f"MLflow: Logged raw RL PyTorch model (no input_example) successfully as fallback.")
                        model_signature = temp_signature # Update main signature
                        model_input_example = None # Update main input example
                        logging_successful = True
                    except Exception as e_pytorch_fallback:
                        print(f"MLflow: Error during fallback logging of raw PyTorch model: {e_pytorch_fallback}")
                        logging_successful = False


                # --- Attempt 3 (Final Fallback): Log raw PyTorch model without signature if all else failed ---
                if not logging_successful and model: # Check if model object exists
                    try:
                        print(f"MLflow: All signature methods failed for RL. Logging model without signature as final fallback.")
                        mlflow.pytorch.log_model(
                            pytorch_model=model,
                            artifact_path=f"{model_type}-model-pytorch-no-sig",
                            registered_model_name=f"{model_type}-model-{dataset}"
                        )
                        print(f"MLflow: Logged {model_type} model WITHOUT signature (final fallback).")
                        # model_signature and model_input_example remain None
                    except Exception as e_final_fallback:
                        print(f"MLflow: Error during final fallback (no signature) for RL model: {e_final_fallback}")

            else: # if not isinstance(model, ActorCritic):
                print(f"MLflow: model is not an instance of ActorCritic for RL type. Got {type(model)}. Skipping RL model logging.")
        
        # Generic message if model was logged but without signature
        if model and not model_signature and model_type == 'RL' and logging_successful: # Check if any RL logging happened
             pass # Specific messages already printed
        elif model and not model_signature:
             print(f"MLflow: Model for {model_type} logged, but signature could not be inferred or logging with signature failed.")
        elif not model:
             print(f"MLflow: Model object for {model_type} was not available. Skipping model logging.")