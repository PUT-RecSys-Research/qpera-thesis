from typing import List, Union

import numpy as np
import pandas as pd


def hide_information_in_dataframe(
    data: pd.DataFrame,
    hide_type: str = "columns",
    columns_to_hide: Union[str, List[str]] = None,
    fraction_to_hide: float = 0.0,
    records_to_hide: List[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Hide information in a DataFrame for testing recommendation algorithm robustness and privacy.

    Args:
        data: The input DataFrame to modify
        hide_type: The type of hiding to perform. Options:
            - "columns": Hide entire columns
            - "records_random": Hide a random fraction of records (rows)
            - "records_selective": Hide specific records based on index
            - "values_in_column": Randomly hide values within specified columns (replace with NaN)
        columns_to_hide: Column names to hide (for "columns" and "values_in_column" hide_types)
        fraction_to_hide: Fraction of records or values to hide (0.0 to 1.0)
        records_to_hide: List of record indices to hide (for "records_selective" hide_type)
        seed: Random seed for reproducibility

    Returns:
        A new DataFrame with the specified information hidden. Original DataFrame is not modified.

    Raises:
        ValueError: If invalid arguments are provided
        KeyError: If specified columns or indices are not found in the DataFrame
        TypeError: If input types are incorrect
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a Pandas DataFrame.")

    df = data.copy()
    np.random.seed(seed)

    if hide_type == "columns":
        _hide_columns(df, columns_to_hide)
    elif hide_type == "records_random":
        df = _hide_random_records(df, fraction_to_hide)
    elif hide_type == "records_selective":
        df = _hide_selective_records(df, records_to_hide)
    elif hide_type == "values_in_column":
        _hide_values_in_columns(df, columns_to_hide, fraction_to_hide)
    else:
        raise ValueError(
            f"Invalid 'hide_type': {hide_type}. "
            f"Valid options are 'columns', 'records_random', 'records_selective', 'values_in_column'."
        )

    return df


def _hide_columns(df: pd.DataFrame, columns_to_hide: Union[str, List[str]]) -> None:
    """Hide entire columns from the DataFrame."""
    if columns_to_hide is None:
        raise ValueError("Must specify 'columns_to_hide' for hide_type='columns'.")
    
    if isinstance(columns_to_hide, str):
        columns_to_hide = [columns_to_hide]
    
    if not isinstance(columns_to_hide, list) or not all(isinstance(c, str) for c in columns_to_hide):
        raise TypeError("'columns_to_hide' must be a string or a list of strings.")

    # Validate all columns exist before dropping
    invalid_columns = [col for col in columns_to_hide if col not in df.columns]
    if invalid_columns:
        raise KeyError(
            f"The following column(s) not found in DataFrame: {invalid_columns}"
        )
    
    df.drop(columns=columns_to_hide, inplace=True)


def _hide_random_records(df: pd.DataFrame, fraction_to_hide: float) -> pd.DataFrame:
    """Hide a random fraction of records from the DataFrame."""
    if not isinstance(fraction_to_hide, (int, float)) or not 0.0 <= fraction_to_hide <= 1.0:
        raise ValueError("'fraction_to_hide' must be a number between 0.0 and 1.0.")
    
    if len(df) == 0:
        return df
    
    num_to_hide = int(len(df) * fraction_to_hide)
    if num_to_hide > 0:
        indices_to_hide = np.random.choice(df.index, size=num_to_hide, replace=False)
        df = df.drop(index=indices_to_hide)
    
    return df


def _hide_selective_records(df: pd.DataFrame, records_to_hide: List[int]) -> pd.DataFrame:
    """Hide specific records based on provided indices."""
    if records_to_hide is None:
        raise ValueError("Must specify 'records_to_hide' for hide_type='records_selective'.")
    
    if not isinstance(records_to_hide, list) or not all(isinstance(i, (int, np.integer)) for i in records_to_hide):
        raise TypeError("'records_to_hide' must be a list of integers (indices).")

    # Validate all indices exist before dropping
    invalid_indices = [idx for idx in records_to_hide if idx not in df.index]
    if invalid_indices:
        raise KeyError(
            f"The following index/indices not found in DataFrame: {invalid_indices}"
        )
    
    df = df.drop(index=records_to_hide)
    return df


def _hide_values_in_columns(df: pd.DataFrame, columns_to_hide: Union[str, List[str]], fraction_to_hide: float) -> None:
    """Hide random values within specified columns by replacing with NaN."""
    if columns_to_hide is None:
        raise ValueError("Must specify 'columns_to_hide' for hide_type='values_in_column'.")
    
    if not isinstance(fraction_to_hide, (int, float)) or not 0.0 <= fraction_to_hide <= 1.0:
        raise ValueError("'fraction_to_hide' must be a number between 0.0 and 1.0.")
    
    if isinstance(columns_to_hide, str):
        columns_to_hide = [columns_to_hide]
    
    if not isinstance(columns_to_hide, list) or not all(isinstance(c, str) for c in columns_to_hide):
        raise TypeError("'columns_to_hide' must be a string or a list of strings.")

    # Validate all columns exist before proceeding
    invalid_columns = [col for col in columns_to_hide if col not in df.columns]
    if invalid_columns:
        raise KeyError(
            f"The following column(s) not found in DataFrame: {invalid_columns}"
        )

    if len(df) == 0:
        return

    # Hide values in each specified column
    for col in columns_to_hide:
        num_to_hide = int(len(df) * fraction_to_hide)
        if num_to_hide > 0:
            indices_to_hide = np.random.choice(df.index, size=num_to_hide, replace=False)
            df.loc[indices_to_hide, col] = np.nan


def change_items_in_dataframe(
    all: pd.DataFrame,
    data: pd.DataFrame,
    fraction_to_change: float = 0.0,
    change_rating: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Modify items in a DataFrame for personalization experiments by replacing them with new items.
    
    Args:
        all: Complete dataset used to derive item distribution and metadata
        data: DataFrame to modify (subset of 'all')
        fraction_to_change: Fraction of items to change per user (0.0 to 1.0)
        change_rating: Whether to update ratings based on new item's average rating
        seed: Random seed for reproducibility
    
    Returns:
        Modified DataFrame with changed items and updated metadata
    """
    np.random.seed(seed)
    df = data.copy()

    # Create probability distribution of items from the complete dataset
    item_distribution = all["itemID"].value_counts(normalize=True)

    # Create mapping of itemID to metadata (title, genres)
    item_details = (
        all.drop_duplicates("itemID")
        .set_index("itemID")[["title", "genres"]]
        .to_dict(orient="index")
    )

    # Prepare item average ratings if rating changes are requested
    item_avg_rating = {}
    if change_rating:
        item_avg_rating = (
            all.groupby("itemID")["rating"]
            .mean()
            .apply(lambda x: round(x * 2) / 2)  # Round to nearest 0.5
            .to_dict()
        )

    # Process each user separately to maintain user-specific changes
    for user_id in df["userID"].unique():
        _change_user_items(
            df, user_id, fraction_to_change, item_distribution, 
            item_details, item_avg_rating, change_rating
        )

    return df


def _change_user_items(
    df: pd.DataFrame,
    user_id: int,
    fraction_to_change: float,
    item_distribution: pd.Series,
    item_details: dict,
    item_avg_rating: dict,
    change_rating: bool
) -> None:
    """Change items for a specific user based on the specified fraction."""
    user_mask = df["userID"] == user_id
    user_data = df[user_mask]
    n_rows_to_change = int(len(user_data) * fraction_to_change)

    if n_rows_to_change == 0:
        return

    # Select random indices to change
    indices_to_change = np.random.choice(
        user_data.index, size=n_rows_to_change, replace=False
    )

    # Sample new items based on the distribution
    new_itemIDs = np.random.choice(
        item_distribution.index, 
        size=n_rows_to_change, 
        p=item_distribution.values
    )

    # Update DataFrame with new items and their metadata
    for idx, new_id in zip(indices_to_change, new_itemIDs):
        df.at[idx, "itemID"] = new_id
        df.at[idx, "title"] = item_details[new_id]["title"]
        df.at[idx, "genres"] = item_details[new_id]["genres"]
        
        # Update rating if requested and available
        if change_rating and new_id in item_avg_rating:
            df.at[idx, "rating"] = item_avg_rating[new_id]
