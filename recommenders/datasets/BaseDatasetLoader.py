import os
import numpy as np
import pandas as pd
from abc import ABC
from typing import Tuple, Dict, List, Union

class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """
  
    def __init__(self, data_path: str):
        """
        Initializes the DatasetLoader.

        Args:
            data_path: The path to the directory containing the dataset.
        """
        self.data_path = data_path
        self.train_path = None
        self.test_path = None

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads the dataset.

        Returns:
            A Pandas DataFrame with all columns.
        """
        dataset_df = pd.read_csv(self.dataset_file)
        return dataset_df
    
    def load_dataset_useful_columns(self) -> pd.DataFrame:
        """
        Loads the dataset.

        Returns:
            A Pandas DataFrame with specific columns.
        """
        pass

    def get_train_test_split(self, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """"
        Splits the interaction dataset into training and testing sets.

        Args:
            test_size (float): Proportion of data to use for testing.
            seed (int): Random seed for reproducibility. To get the same split each time.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
        """
        dataset_df = self.load_dataset()
        train_df = dataset_df.sample(frac=1 - test_size, random_state=seed)
        test_df = dataset_df.drop(train_df.index)

        train_file = os.path.join(self.train_path, "train.csv")
        test_file = os.path.join(self.test_path, "test.csv")

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        return train_df, test_df

    def hide_information(self, data: pd.DataFrame,
                     hide_type: str = "columns",
                     columns_to_hide: Union[str, List[str]] = None,
                     fraction_to_hide: float = 0.0,
                     records_to_hide: List[int] = None,
                     seed: int = 42) -> pd.DataFrame:
        """
            Hides information in a Pandas DataFrame for testing recommendation algorithm robustness.

            Args:
                data: The input Pandas DataFrame.
                hide_type: The type of hiding to perform.  Options:
                    - "columns": Hide entire columns.
                    - "records_random": Hide a random fraction of records (rows).
                    - "records_selective": Hide specific records based on index.
                    - "values_in_column":  Randomly hide values within specified columns.
                columns_to_hide:  List of column names to hide (for "columns" and "values_in_column" hide_types).
                                Can be a single column name (string) or a list of column names.
                fraction_to_hide: Fraction of records or values to hide (for "records_random" and "values_in_column" hide_types).
                                Must be between 0.0 and 1.0.
                records_to_hide: List of record indices to hide (for "records_selective" hide_type).
                seed: Random seed for reproducibility.

            Returns:
                A new Pandas DataFrame with the specified information hidden.  Original DataFrame is not modified.

            Raises:
                ValueError: If invalid arguments are provided.
            """
        df = data.copy()
        np.random.seed(seed)

        if hide_type == "columns":
            if columns_to_hide is None:
                raise ValueError("Must specify 'columns_to_hide' for hide_type='columns'.")
            if isinstance(columns_to_hide, str):
                columns_to_hide = [columns_to_hide]
            if not all(col in df.columns for col in columns_to_hide):
                raise ValueError("One or more 'columns_to_hide' not found in DataFrame.")
            df = df.drop(columns=columns_to_hide)

        elif hide_type == "records_random":
            if not 0.0 <= fraction_to_hide <= 1.0:
                raise ValueError("'fraction_to_hide' must be between 0.0 and 1.0.")
            num_to_hide = int(len(df) * fraction_to_hide)
            indices_to_hide = np.random.choice(df.index, size=num_to_hide, replace=False)
            df = df.drop(index=indices_to_hide)

        elif hide_type == "records_selective":
            if records_to_hide is None:
                raise ValueError("Must specify 'records_to_hide' for hide_type='records_selective'.")
            if not all(idx in df.index for idx in records_to_hide):
                raise ValueError("One or more 'records_to_hide' indices not found in DataFrame.")
            df = df.drop(index=records_to_hide)

        elif hide_type == "values_in_column":
            if columns_to_hide is None:
                raise ValueError("Must specify 'columns_to_hide' for hide_type='values_in_column'.")
            if not 0.0 <= fraction_to_hide <= 1.0:
                raise ValueError("'fraction_to_hide' must be between 0.0 and 1.0.")
            if isinstance(columns_to_hide, str):
                columns_to_hide = [columns_to_hide]
            if not all(col in df.columns for col in columns_to_hide):
                raise ValueError("One or more 'columns_to_hide' not found in DataFrame.")

            for col in columns_to_hide:
                num_to_hide = int(len(df) * fraction_to_hide)
                indices_to_hide = np.random.choice(df.index, size=num_to_hide, replace=False)
                df.loc[indices_to_hide, col] = np.nan
        else:
            raise ValueError(f"Invalid 'hide_type': {hide_type}")

        return df










    def load_item_features(self) -> pd.DataFrame:
        """
        Loads item features (e.g., movie genres, product descriptions).

        Returns:
            A Pandas DataFrame with at least 'item_id' and feature columns.
        """
        pass

    def get_user_item_interactions(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Gets a dictionary mapping user IDs to a list of (item_id, rating) tuples.
        This is useful for CF algorithms.

        Returns:
            A dictionary where keys are user IDs and values are lists of
            (item_id, rating) tuples.
        """
        pass