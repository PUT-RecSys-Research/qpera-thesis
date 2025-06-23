from __future__ import absolute_import, division, print_function

from typing import List, Optional, Union

import pandas as pd

from .rl_preprocess import preprocess_rl
from .rl_test_agent import test_agent_rl
from .rl_train_agent import train_agent_rl
from .rl_train_transe_model import train_transe_model_rl


def rl_experiment_loop(
    TOP_K: int,
    dataset: str,
    want_col: List[str],
    num_rows: Optional[int],
    ratio: float,
    seed: int,
    personalization: bool = False,
    fraction_to_change: float = 0.0,
    change_rating: bool = False,
    privacy: bool = False,
    hide_type: str = "values_in_column",
    columns_to_hide: Optional[List[str]] = None,
    fraction_to_hide: float = 0.0,
    records_to_hide: Optional[List[int]] = None,
) -> None:
    """
    Execute complete RL-based recommendation system experiment pipeline.
    
    This function orchestrates the full experimental workflow:
    1. Data preprocessing and knowledge graph construction
    2. TransE model training for entity/relation embeddings
    3. RL agent training for recommendation policy learning
    4. Testing and evaluation with metrics calculation
    
    Args:
        TOP_K: Number of top recommendations to generate
        dataset: Dataset name (e.g., "movielens", "amazonsales")
        want_col: List of required columns in the dataset
        num_rows: Number of rows to use (None for all rows)
        ratio: Train/test split ratio (e.g., 0.8 for 80% train, 20% test)
        seed: Random seed for reproducibility
        personalization: Whether to apply personalization modifications
        fraction_to_change: Fraction of data to modify for personalization
        change_rating: Whether to modify rating values for personalization
        privacy: Whether to apply privacy modifications
        hide_type: Type of privacy hiding ("values_in_column" or "records")
        columns_to_hide: List of column names to hide for privacy
        fraction_to_hide: Fraction of data to hide for privacy
        records_to_hide: Specific record indices to hide for privacy
        
    Returns:
        None
        
    Raises:
        Exception: If any stage of the pipeline fails
    """
    print(f"\n===== Starting RL Experiment Pipeline for {dataset.upper()} =====")
    print(f"Configuration: TOP_K={TOP_K}, seed={seed}, rows={num_rows}, ratio={ratio}")
    
    if privacy:
        print(f"Privacy settings: hide_type={hide_type}, fraction_to_hide={fraction_to_hide}")
    if personalization:
        print(f"Personalization settings: fraction_to_change={fraction_to_change}, change_rating={change_rating}")
    
    # Stage 1: Data Preprocessing and Knowledge Graph Construction
    try:
        print("\n===== Stage 1: Data Preprocessing & Knowledge Graph Construction =====")
        data_df, train_df, test_df = _run_preprocessing_stage(
            dataset=dataset,
            want_col=want_col,
            num_rows=num_rows,
            ratio=ratio,
            seed=seed,
            personalization=personalization,
            fraction_to_change=fraction_to_change,
            change_rating=change_rating,
            privacy=privacy,
            hide_type=hide_type,
            columns_to_hide=columns_to_hide,
            fraction_to_hide=fraction_to_hide,
            records_to_hide=records_to_hide,
        )
        print(f"✓ Preprocessing completed. Train: {len(train_df)}, Test: {len(test_df)}, Total: {len(data_df)}")
        
    except Exception as e:
        print(f"✗ Stage 1 failed: {e}")
        raise Exception(f"Preprocessing stage failed: {e}") from e

    # Stage 2: Knowledge Graph Embedding Training
    try:
        print("\n===== Stage 2: TransE Knowledge Graph Embedding Training =====")
        _run_kge_training_stage(dataset=dataset, seed=seed)
        print("✓ TransE model training completed successfully")
        
    except Exception as e:
        print(f"✗ Stage 2 failed: {e}")
        raise Exception(f"KGE training stage failed: {e}") from e

    # Stage 3: Reinforcement Learning Agent Training
    try:
        print("\n===== Stage 3: RL Agent Policy Training =====")
        _run_rl_training_stage(dataset=dataset, seed=seed)
        print("✓ RL agent training completed successfully")
        
    except Exception as e:
        print(f"✗ Stage 3 failed: {e}")
        raise Exception(f"RL training stage failed: {e}") from e

    # Stage 4: Testing and Evaluation
    try:
        print("\n===== Stage 4: Testing & Evaluation =====")
        _run_testing_stage(
            dataset=dataset,
            TOP_K=TOP_K,
            want_col=want_col,
            num_rows=num_rows,
            ratio=ratio,
            seed=seed,
            data_df=data_df,
            train_df=train_df,
            test_df=test_df,
            privacy=privacy,
            fraction_to_hide=fraction_to_hide,
            personalization=personalization,
            fraction_to_change=fraction_to_change,
        )
        print("✓ Testing and evaluation completed successfully")
        
    except Exception as e:
        print(f"✗ Stage 4 failed: {e}")
        raise Exception(f"Testing stage failed: {e}") from e

    print(f"\n===== RL Experiment Pipeline Completed Successfully for {dataset.upper()} =====")


def _run_preprocessing_stage(
    dataset: str,
    want_col: List[str],
    num_rows: Optional[int],
    ratio: float,
    seed: int,
    personalization: bool,
    fraction_to_change: float,
    change_rating: bool,
    privacy: bool,
    hide_type: str,
    columns_to_hide: Optional[List[str]],
    fraction_to_hide: float,
    records_to_hide: Optional[List[int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the preprocessing stage of the RL pipeline.
    
    Returns:
        Tuple of (full_data, train_data, test_data) DataFrames
    """
    print("Running data preprocessing and knowledge graph construction...")
    
    data_df, train_df, test_df = preprocess_rl(
        dataset=dataset,
        want_col=want_col,
        num_rows=num_rows,
        ratio=ratio,
        seed=seed,
        personalization=personalization,
        fraction_to_change=fraction_to_change,
        change_rating=change_rating,
        privacy=privacy,
        hide_type=hide_type,
        columns_to_hide=columns_to_hide,
        fraction_to_hide=fraction_to_hide,
        records_to_hide=records_to_hide,
    )
    
    # Validate preprocessing results
    if data_df.empty or train_df.empty or test_df.empty:
        raise ValueError("Preprocessing returned empty DataFrames")
    
    return data_df, train_df, test_df


def _run_kge_training_stage(dataset: str, seed: int) -> None:
    """
    Execute the Knowledge Graph Embedding training stage.
    
    Args:
        dataset: Dataset name
        seed: Random seed for reproducibility
    """
    print("Training TransE model for knowledge graph embeddings...")
    print("This stage learns vector representations for entities and relations.")
    
    train_transe_model_rl(dataset=dataset, seed=seed)


def _run_rl_training_stage(dataset: str, seed: int) -> None:
    """
    Execute the Reinforcement Learning agent training stage.
    
    Args:
        dataset: Dataset name
        seed: Random seed for reproducibility
    """
    print("Training RL agent for recommendation policy learning...")
    print("This stage learns to navigate the knowledge graph for recommendations.")
    
    train_agent_rl(dataset=dataset, seed=seed)


def _run_testing_stage(
    dataset: str,
    TOP_K: int,
    want_col: List[str],
    num_rows: Optional[int],
    ratio: float,
    seed: int,
    data_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    privacy: bool,
    fraction_to_hide: float,
    personalization: bool,
    fraction_to_change: float,
) -> None:
    """
    Execute the testing and evaluation stage.
    
    Args:
        dataset: Dataset name
        TOP_K: Number of top recommendations
        want_col: Required columns
        num_rows: Number of rows used
        ratio: Train/test split ratio
        seed: Random seed
        data_df: Full dataset
        train_df: Training data
        test_df: Test data
        privacy: Privacy flag
        fraction_to_hide: Fraction hidden for privacy
        personalization: Personalization flag
        fraction_to_change: Fraction changed for personalization
    """
    print("Running trained RL agent on test data...")
    print("This stage generates recommendations and calculates evaluation metrics.")
    
    test_agent_rl(
        dataset=dataset,
        TOP_K=TOP_K,
        want_col=want_col,
        num_rows=num_rows,
        ratio=ratio,
        seed=seed,
        data_df=data_df,
        train_df=train_df,
        test_df=test_df,
        privacy=privacy,
        fraction_to_hide=fraction_to_hide,
        personalization=personalization,
        fraction_to_change=fraction_to_change,
    )


def run_rl_experiment(
    TOP_K: int = 10,
    dataset: str = "movielens",
    want_col: Optional[List[str]] = None,
    num_rows: Optional[int] = None,
    ratio: float = 0.8,
    seed: int = 42,
    **kwargs
) -> None:
    """
    Convenience function to run RL experiment with default parameters.
    
    Args:
        TOP_K: Number of top recommendations (default: 10)
        dataset: Dataset name (default: "movielens")
        want_col: Required columns (default: None, uses dataset defaults)
        num_rows: Number of rows to use (default: None, uses all)
        ratio: Train/test split ratio (default: 0.8)
        seed: Random seed (default: 42)
        **kwargs: Additional arguments passed to rl_experiment_loop
    """
    # Set default columns if not provided
    if want_col is None:
        default_columns = {
            "movielens": ["userID", "itemID", "rating", "genres"],
            "amazonsales": ["userID", "itemID", "rating"],
            "postrecommendations": ["userID", "itemID", "rating"],
        }
        want_col = default_columns.get(dataset, ["userID", "itemID", "rating"])
    
    print(f"Running RL experiment with default parameters for {dataset}")
    print(f"Using columns: {want_col}")
    
    rl_experiment_loop(
        TOP_K=TOP_K,
        dataset=dataset,
        want_col=want_col,
        num_rows=num_rows,
        ratio=ratio,
        seed=seed,
        **kwargs
    )


def validate_experiment_parameters(
    TOP_K: int,
    dataset: str,
    want_col: List[str],
    ratio: float,
    fraction_to_change: float = 0.0,
    fraction_to_hide: float = 0.0,
) -> None:
    """
    Validate experiment parameters before running the pipeline.
    
    Args:
        TOP_K: Number of top recommendations
        dataset: Dataset name
        want_col: Required columns
        ratio: Train/test split ratio
        fraction_to_change: Fraction to change for personalization
        fraction_to_hide: Fraction to hide for privacy
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if TOP_K <= 0:
        raise ValueError("TOP_K must be positive")
    
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")
    
    if not isinstance(want_col, list) or not want_col:
        raise ValueError("want_col must be a non-empty list")
    
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")
    
    if not 0 <= fraction_to_change <= 1:
        raise ValueError("fraction_to_change must be between 0 and 1")
    
    if not 0 <= fraction_to_hide <= 1:
        raise ValueError("fraction_to_hide must be between 0 and 1")


# Convenience exports for easier imports
__all__ = [
    "rl_experiment_loop",
    "run_rl_experiment", 
    "validate_experiment_parameters",
]
