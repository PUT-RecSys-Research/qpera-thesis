import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """

    def __init__(self, data_path: str, merge_file_name: str = "merge_file.csv"):
        self.data_path = data_path
        self.merge_file = os.path.join(self.data_path, merge_file_name)

    def load_dataset(self, columns: Optional[List[str]] = None, num_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Loads the dataset, merges it if necessary, selects specified columns, and optionally limits the number of rows.

        Args:
            columns: Optional list of column names to return.
            num_rows: Optional integer specifying the number of rows to load. If None, loads all rows.

        Returns:
            A pandas DataFrame.
        """
        if os.path.exists(self.merge_file):
            # If num_rows is specified, try to read only that many rows from the CSV
            if num_rows is not None:
                try:
                    dataset_df = pd.read_csv(self.merge_file, nrows=num_rows)
                except ValueError: # Fallback to loading all if there are less than `num_rows`
                    dataset_df = pd.read_csv(self.merge_file)
            else:
                dataset_df = pd.read_csv(self.merge_file)
        else:
            dataset_df = self.merge_datasets()
            dataset_df.to_csv(self.merge_file, index=False)  # Save after merging

        if columns is not None:
            self._validate_columns(dataset_df, columns)
            dataset_df = dataset_df[columns]

        # Apply row limit *after* merging and saving, but before column selection if merging happened
        if num_rows is not None and not os.path.exists(self.merge_file.replace('.csv', f'_{num_rows}.csv')):
          if len(dataset_df) > num_rows:
            dataset_df = dataset_df.sample(n=num_rows, random_state=42).reset_index(drop=True)  # Consistent random sampling
            dataset_df.to_csv(self.merge_file.replace('.csv', f'_{num_rows}.csv'), index=False) # Save limited dataset.


        return dataset_df

    @abstractmethod
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merges the necessary data files to create the dataset.  Must be implemented by subclasses.

        Returns:
            A pandas DataFrame containing the merged dataset.
        """
        pass

    @staticmethod
    def normalize_column_names(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        return df.rename(columns=column_mapping)

    @staticmethod
    def _validate_columns(df: pd.DataFrame, columns: List[str]):
        if not all(isinstance(c, str) for c in columns):
            raise TypeError("The 'columns' argument must be a list of strings (column names).")

        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            raise KeyError(f"The following column names were not found in the DataFrame: {invalid_columns}")


class AmazonSalesDataset(BaseDatasetLoader):
    def __init__(self, data_path: str = "datasets/AmazonSales"):
        super().__init__(data_path)
        self.dataset = os.path.join(self.data_path, "amazon.csv")
        self.column_mapping = {"user_id": "userID", "product_id": "itemID","category": "genres", 'product_name': 'title', 'predicted_rating': 'rating'}

    def merge_datasets(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset)
        df = self.normalize_column_names(df, self.column_mapping)
        df = df.drop_duplicates(subset=['userID', 'itemID'], keep='first')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')


        df['genres'] = df['genres'].str.replace('|', ' ', regex=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].astype('int64') // 10**9


        return df


class MovieLensDataset(BaseDatasetLoader):
    def __init__(self, data_path: str = "datasets/MovieLens"):
        super().__init__(data_path)
        self.ratings_file = os.path.join(self.data_path, "rating.csv")
        self.movies_file = os.path.join(self.data_path, "movie.csv")
        self.tag_file = os.path.join(self.data_path, "tag.csv")
        self.column_mapping = {"userId": "userID", "movieId": "itemID"}


    def merge_datasets(self) -> pd.DataFrame:
      ratings_df = pd.read_csv(self.ratings_file)
      movies_df = pd.read_csv(self.movies_file)
      tags_df = pd.read_csv(self.tag_file)
      tags_df = tags_df.rename(columns={'timestamp': 'tag_timestamp'})
      merge_file_df = pd.merge(ratings_df, movies_df, on="movieId", how="left")
      final_merge_file_df = pd.merge(merge_file_df, tags_df, on=["movieId", "userId"], how="left")
      final_merge_file_df = self.normalize_column_names(final_merge_file_df, self.column_mapping)


      final_merge_file_df['genres'] = final_merge_file_df['genres'].str.replace('|', ' ', regex=False)
      final_merge_file_df['timestamp'] = pd.to_datetime(final_merge_file_df['timestamp'])
      final_merge_file_df['timestamp'] = final_merge_file_df['timestamp'].astype('int64') // 10**9


      final_merge_file_df = final_merge_file_df.drop_duplicates(subset=['userID', 'itemID', 'rating'], keep='first')
      return final_merge_file_df



class PostRecommendationsDataset(BaseDatasetLoader):
    def __init__(self, data_path: str = "datasets/PostRecommendations"):
        super().__init__(data_path)
        self.userData_file = os.path.join(self.data_path, "user_data.csv")
        self.viewData_file = os.path.join(self.data_path, "view_data.csv")
        self.postData_file = os.path.join(self.data_path, "post_data.csv")
        self.column_mapping = {"user_id": "userID", "post_id": "itemID"}

    def merge_datasets(self) -> pd.DataFrame:
        user_df = pd.read_csv(self.userData_file)
        view_df = pd.read_csv(self.viewData_file)
        post_df = pd.read_csv(self.postData_file)

        merge_file_df = pd.merge(user_df, view_df, on="user_id", how="left")
        final_merge_file_df = pd.merge(merge_file_df, post_df, on="post_id", how="left")
        final_merge_file_df = self.normalize_column_names(final_merge_file_df, self.column_mapping)


        # TODO: Uncomment this line after adding the genres column to the dataset
        # final_merge_file_df['genres'] = final_merge_file_df['genres'].str.replace('|', ' ', regex=False)
        # final_merge_file_df['timestamp'] = pd.to_datetime(final_merge_file_df['timestamp'])
        # final_merge_file_df['timestamp'] = final_merge_file_df['timestamp'].astype('int64') // 10**9


        final_merge_file_df = final_merge_file_df.drop_duplicates(subset=['userID', 'itemID'], keep='first')
        return final_merge_file_df


def loader(dataset_name: str = "movielens", want_col: Optional[List[str]] = None,
           num_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads a specified dataset.

    Args:
        dataset_name: The name of the dataset.
        want_col: Optional list of column names to return.
        num_rows: Optional integer specifying the number of rows to load (for supported datasets).

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If an invalid dataset name is provided.
    """
    loaders = {
        "amazonsales": AmazonSalesDataset,
        "movielens": MovieLensDataset,
        "postrecommendations": PostRecommendationsDataset,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Invalid dataset name: {dataset_name}. Choose from {list(loaders.keys())}")

    loader_instance = loaders[dataset_name]()
    return loader_instance.load_dataset(want_col, num_rows)

# Example Usage
# # Load the entire MovieLens dataset
# full_movielens = loader("movielens", ["userID", "itemID", "rating"])
# print(f"Full MovieLens dataset shape: {full_movielens.shape}")
# print(full_movielens.head())

# # Load only 100,000 rows of the MovieLens dataset
# limited_movielens = loader("movielens", ["userID", "itemID", "rating"], num_rows=100000)
# print(f"Limited MovieLens dataset shape: {limited_movielens.shape}")
# print(limited_movielens.head())

# # Load full amazon dataset
# amazon_data = loader("amazonsales",  ["userID", "itemID"])
# print(f"Full Amazon dataset shape: {amazon_data.shape}")
# print(amazon_data.head())

# # Load full post recommendations dataset
# post_data = loader("postrecommendations",  ["userID", "itemID"])
# print(f"Full Post Recommendations dataset shape: {post_data.shape}")
# print(post_data.head())
