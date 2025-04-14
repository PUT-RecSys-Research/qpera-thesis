import numpy as np
import pandas as pd
from typing import Optional, List, Union

def hide_information_in_dataframe(data: pd.DataFrame,
                                  hide_type: str = "columns",
                                  columns_to_hide: Union[str, List[str]] = None,
                                  fraction_to_hide: float = 0.0,
                                  records_to_hide: List[int] = None,
                                  seed: int = 42) -> pd.DataFrame:
    """
    Hides information in a Pandas DataFrame for testing recommendation algorithm robustness.

    Args:
        data: The input Pandas DataFrame.
        hide_type: The type of hiding to perform. Options:
            - "columns": Hide entire columns.
            - "records_random": Hide a random fraction of records (rows).
            - "records_selective": Hide specific records based on index.
            - "values_in_column": Randomly hide values within specified columns (replace with NaN).
        columns_to_hide: List of column names to hide (for "columns" and "values_in_column" hide_types).
                        Can be a single column name (string) or a list of column names.
        fraction_to_hide: Fraction of records or values to hide (for "records_random" and "values_in_column" hide_types).
                        Must be between 0.0 and 1.0.
        records_to_hide: List of record indices to hide (for "records_selective" hide_type).
        seed: Random seed for reproducibility.

    Returns:
        A new Pandas DataFrame with the specified information hidden. Original DataFrame is not modified.

    Raises:
        ValueError: If invalid arguments are provided.
        KeyError: If specified columns or indices are not found in the DataFrame.
        TypeError: If input types are incorrect.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a Pandas DataFrame.")

    df = data.copy()
    np.random.seed(seed) # Set seed for reproducibility

    if hide_type == "columns":
        if columns_to_hide is None:
            raise ValueError("Must specify 'columns_to_hide' for hide_type='columns'.")
        if isinstance(columns_to_hide, str):
            columns_to_hide = [columns_to_hide]
        if not isinstance(columns_to_hide, list) or not all(isinstance(c, str) for c in columns_to_hide):
             raise TypeError("'columns_to_hide' must be a string or a list of strings.")

        # Check if all columns exist before attempting to drop
        invalid_columns = [col for col in columns_to_hide if col not in df.columns]
        if invalid_columns:
            raise KeyError(f"The following column(s) specified in 'columns_to_hide' not found in DataFrame: {invalid_columns}")
        df = df.drop(columns=columns_to_hide)

    elif hide_type == "records_random":
        if not isinstance(fraction_to_hide, (int, float)) or not 0.0 <= fraction_to_hide <= 1.0:
            raise ValueError("'fraction_to_hide' must be a number between 0.0 and 1.0.")
        if len(df) == 0: # Handle empty DataFrame case
             return df
        num_to_hide = int(len(df) * fraction_to_hide)
        if num_to_hide > 0:
            indices_to_hide = np.random.choice(df.index, size=num_to_hide, replace=False)
            df = df.drop(index=indices_to_hide)

    elif hide_type == "records_selective":
        if records_to_hide is None:
            raise ValueError("Must specify 'records_to_hide' for hide_type='records_selective'.")
        if not isinstance(records_to_hide, list) or not all(isinstance(i, (int, np.integer)) for i in records_to_hide):
             raise TypeError("'records_to_hide' must be a list of integers (indices).")

        # Check if all indices exist before attempting to drop
        invalid_indices = [idx for idx in records_to_hide if idx not in df.index]
        if invalid_indices:
            # Consider raising KeyError or ValueError here. KeyError might be more specific.
            raise KeyError(f"The following index/indices specified in 'records_to_hide' not found in DataFrame: {invalid_indices}")
        df = df.drop(index=records_to_hide)

    elif hide_type == "values_in_column":
        if columns_to_hide is None:
            raise ValueError("Must specify 'columns_to_hide' for hide_type='values_in_column'.")
        if not isinstance(fraction_to_hide, (int, float)) or not 0.0 <= fraction_to_hide <= 1.0:
            raise ValueError("'fraction_to_hide' must be a number between 0.0 and 1.0.")
        if isinstance(columns_to_hide, str):
            columns_to_hide = [columns_to_hide]
        if not isinstance(columns_to_hide, list) or not all(isinstance(c, str) for c in columns_to_hide):
             raise TypeError("'columns_to_hide' must be a string or a list of strings.")

        # Check if all columns exist before proceeding
        invalid_columns = [col for col in columns_to_hide if col not in df.columns]
        if invalid_columns:
            raise KeyError(f"The following column(s) specified in 'columns_to_hide' not found in DataFrame: {invalid_columns}")

        if len(df) == 0: # Handle empty DataFrame case
            return df

        for col in columns_to_hide:
            num_to_hide = int(len(df) * fraction_to_hide)
            if num_to_hide > 0:
                # Ensure indices are valid before assignment
                valid_indices = df.index.tolist()
                indices_to_hide = np.random.choice(valid_indices, size=num_to_hide, replace=False)
                df.loc[indices_to_hide, col] = np.nan # Use .loc for safe assignment
    else:
        raise ValueError(f"Invalid 'hide_type': {hide_type}. Valid options are 'columns', 'records_random', 'records_selective', 'values_in_column'.")

    return df


def change_items_in_dataframe(all: pd.DataFrame,
                              data: pd.DataFrame,
                              fraction_to_change: float = 0.0,
                              change_rating: bool = False,
                              seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    df = data.copy()

    # Tworzymy rozkład prawdopodobieństwa itemID w całym zbiorze
    item_distribution = all['itemID'].value_counts(normalize=True)

    # Słownik: itemID -> (title, genres)
    item_details = all.drop_duplicates('itemID').set_index('itemID')[['title', 'genres']].to_dict(orient='index')

    # Jeśli zmieniamy rating — przygotuj słownik itemID -> średnia ocena
    if change_rating:
        item_avg_rating = (
            all.groupby('itemID')['rating']
            .mean()
            .apply(lambda x: round(x * 2) / 2)
            .to_dict()
        )

    # Dla każdego użytkownika
    for user_id in df['userID'].unique():
        user_mask = df['userID'] == user_id
        user_data = df[user_mask]
        n_rows_to_change = int(len(user_data) * fraction_to_change)

        if n_rows_to_change == 0:
            continue

        # Wybierz indeksy do podmiany
        indices_to_change = np.random.choice(user_data.index, size=n_rows_to_change, replace=False)

        # Losuj nowe itemID na podstawie rozkładu
        new_itemIDs = np.random.choice(item_distribution.index, size=n_rows_to_change, p=item_distribution.values)

        # Podmień itemID oraz odpowiadające dane
        for idx, new_id in zip(indices_to_change, new_itemIDs):
            df.at[idx, 'itemID'] = new_id
            df.at[idx, 'title'] = item_details[new_id]['title']
            df.at[idx, 'genres'] = item_details[new_id]['genres']
            if change_rating and new_id in item_avg_rating:
                df.at[idx, 'rating'] = item_avg_rating[new_id]

    return df

