import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from . import frequency_based_rating_gen, rating_timestamp_gen


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """

    def __init__(self, raw_data_path: str, processed_data_path: str, merge_file_name: str = "merge_file.csv"):
        self.raw_data_path = raw_data_path  # Where raw CSV files are stored
        self.processed_data_path = processed_data_path  # Where processed files go
        self.merge_file = os.path.join(self.processed_data_path, merge_file_name)
        
        # Ensure processed directory exists
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Check if required files exist (no auto-download)
        if not self._check_local_files_exist():
            raise FileNotFoundError(
                f"Required dataset files not found in {self.raw_data_path}. "
                f"Please run 'make check-datasets' to download them automatically."
            )

    def _check_local_files_exist(self) -> bool:
        """Check if required dataset files exist locally."""
        # This method should be implemented by subclasses
        return False

    def load_dataset(
        self,
        columns: Optional[List[str]] = None,
        num_rows: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Loads the dataset, merges it if necessary, selects specified columns, and optionally limits the number of rows.

        Args:
            columns: Optional list of column names to return.
            num_rows: Optional integer specifying the number of rows to load. If None, loads all rows.
            seed: Optional seed for random number generator, used when limiting number of rows.

        Returns:
            A pandas DataFrame.
        """
        # Check if a pre-processed file with specific rows and seed exists
        if num_rows is not None and seed is not None:
            limited_file = self.merge_file.replace(".csv", f"_r{num_rows}_s{seed}.csv")
            if os.path.exists(limited_file):
                print(f"Loading pre-processed file: {limited_file}")
                dataset_df = pd.read_csv(limited_file)
                if columns is not None:
                    self.validate_columns(dataset_df, columns)
                    dataset_df = dataset_df[columns]
                return dataset_df

        # Load or create the merged file with sequential row limiting
        if os.path.exists(self.merge_file):
            print(f"Loading existing merged file: {self.merge_file}")
            # KEY CHANGE: Use nrows for sequential limiting instead of loading all then sampling
            if num_rows is not None:
                try:
                    dataset_df = pd.read_csv(self.merge_file, nrows=num_rows)
                    print(f"Loaded first {num_rows} rows sequentially from existing file")
                except ValueError:  # Fallback if file has fewer rows
                    dataset_df = pd.read_csv(self.merge_file)
                    print(f"File has fewer than {num_rows} rows, loaded all {len(dataset_df)} rows")
            else:
                dataset_df = pd.read_csv(self.merge_file)
        else:
            print(f"Merging datasets from {self.raw_data_path} and creating: {self.merge_file}")
            dataset_df = self.merge_datasets()
            dataset_df.to_csv(self.merge_file, index=False)
            
            # Apply sequential limiting after merging if needed
            if num_rows is not None and len(dataset_df) > num_rows:
                print(f"Limiting merged dataset to first {num_rows} rows")
                dataset_df = dataset_df.head(num_rows)

        # Save the limited dataset for future use (only if using sequential limiting)
        if num_rows is not None and seed is not None and len(dataset_df) <= num_rows:
            limited_file = self.merge_file.replace(".csv", f"_r{num_rows}_s{seed}.csv")
            if not os.path.exists(limited_file):
                dataset_df.to_csv(limited_file, index=False)
                print(f"Saved limited dataset: {limited_file}")

        # Select specific columns if requested
        if columns is not None:
            self.validate_columns(dataset_df, columns)
            dataset_df = dataset_df[columns]

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
    def validate_columns(df: pd.DataFrame, columns: List[str]):
        if not all(isinstance(c, str) for c in columns):
            raise TypeError("The 'columns' argument must be a list of strings (column names).")

        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            raise KeyError(f"The following column names were not found in the DataFrame: {invalid_columns}")


class AmazonSalesDataset(BaseDatasetLoader):
    def __init__(self, raw_data_path: str = "datasets/AmazonSales", processed_data_path: str = "ppera/datasets/AmazonSales"):
        # Set file paths BEFORE calling super().__init__
        self.dataset_file = os.path.join(raw_data_path, "amazon.csv")
        self.column_mapping = {
            "user_id": "userID",
            "product_id": "itemID",
            "category": "genres",
            "product_name": "title",
            "predicted_rating": "rating",
        }
        
        # Now call parent constructor
        super().__init__(raw_data_path, processed_data_path)

    def _check_local_files_exist(self) -> bool:
        """Check if all required Amazon Sales files exist."""
        return os.path.exists(self.dataset_file)

    def merge_datasets(self) -> pd.DataFrame:
        rating_timestamp_gen.rating_timestamp_gen(self.dataset_file, self.dataset_file)
        df = pd.read_csv(self.dataset_file)
        df = self.normalize_column_names(df, self.column_mapping)
        df = df.drop_duplicates(subset=["userID", "itemID"], keep="first")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        df["genres"] = df[["genres", "about_product"]].astype(str).agg(" | ".join, axis=1).str.strip(" |")
        df["genres"] = df["genres"].str.replace("|", " ", regex=False)

        # df = df.drop(columns=['about_product'])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64") // 10**9

        # drop unnecessary columns
        df = df.drop(
            columns=[
                "discounted_price",
                "actual_price",
                "discount_percentage",
                "rating_count",
                "about_product",
                "user_name",
                "review_id",
                "review_title",
                "review_content",
                "img_link",
                "product_link",
            ]
        )
        return df


class MovieLensDataset(BaseDatasetLoader):
    def __init__(self, raw_data_path: str = "datasets/MovieLens", processed_data_path: str = "ppera/datasets/MovieLens"):
        # Set file paths BEFORE calling super().__init__
        self.ratings_file = os.path.join(raw_data_path, "rating.csv")
        self.movies_file = os.path.join(raw_data_path, "movie.csv")
        self.tag_file = os.path.join(raw_data_path, "tag.csv")
        self.column_mapping = {"userId": "userID", "movieId": "itemID"}
        
        # Now call parent constructor
        super().__init__(raw_data_path, processed_data_path)

    def _check_local_files_exist(self) -> bool:
        """Check if all required MovieLens files exist."""
        return (os.path.exists(self.ratings_file) and 
                os.path.exists(self.movies_file) and 
                os.path.exists(self.tag_file))

    def merge_datasets(self) -> pd.DataFrame:
        # Read from raw data path
        if not os.path.exists(self.ratings_file):
            raise FileNotFoundError(f"MovieLens ratings file not found at {self.ratings_file}")
        if not os.path.exists(self.movies_file):
            raise FileNotFoundError(f"MovieLens movies file not found at {self.movies_file}")
        if not os.path.exists(self.tag_file):
            raise FileNotFoundError(f"MovieLens tags file not found at {self.tag_file}")
        
        print(f"Loading MovieLens files:")
        print(f"  - Ratings: {self.ratings_file}")
        print(f"  - Movies: {self.movies_file}")
        print(f"  - Tags: {self.tag_file}")
        
        ratings_df = pd.read_csv(self.ratings_file)
        movies_df = pd.read_csv(self.movies_file)
        tags_df = pd.read_csv(self.tag_file)
        
        print(f"Loaded shapes - Ratings: {ratings_df.shape}, Movies: {movies_df.shape}, Tags: {tags_df.shape}")
        
        tags_df = tags_df.rename(columns={"timestamp": "tag_timestamp"})
        merge_file_df = pd.merge(ratings_df, movies_df, on="movieId", how="left")
        final_merge_file_df = pd.merge(merge_file_df, tags_df, on=["movieId", "userId"], how="left")
        final_merge_file_df = self.normalize_column_names(final_merge_file_df, self.column_mapping)

        final_merge_file_df["genres"] = final_merge_file_df["genres"].str.replace("|", " ", regex=False)

        final_merge_file_df["timestamp"] = pd.to_datetime(final_merge_file_df["timestamp"])
        final_merge_file_df["timestamp"] = final_merge_file_df["timestamp"].astype("int64") // 10**9

        final_merge_file_df = final_merge_file_df.drop_duplicates(subset=["userID", "itemID", "rating"], keep="first")
        # drop unnecessary columns
        final_merge_file_df = final_merge_file_df.drop(columns=["tag_timestamp", "tag"])

        print(f"Final merged dataset shape: {final_merge_file_df.shape}")
        return final_merge_file_df


class PostRecommendationsDataset(BaseDatasetLoader):
    def __init__(self, raw_data_path: str = "datasets/PostRecommendations", processed_data_path: str = "ppera/datasets/PostRecommendations"):
        # Set file paths BEFORE calling super().__init__
        self.userData_file = os.path.join(raw_data_path, "user_data.csv")
        self.viewData_file = os.path.join(raw_data_path, "view_data.csv")
        self.postData_file = os.path.join(raw_data_path, "post_data.csv")
        self.column_mapping = {
            "user_id": "userID",
            "post_id": "itemID",
            "time_stamp": "timestamp",
            "category": "genres",
        }
        
        # Now call parent constructor
        super().__init__(raw_data_path, processed_data_path)

    def _check_local_files_exist(self) -> bool:
        """Check if all required Post Recommendations files exist."""
        return (os.path.exists(self.userData_file) and 
                os.path.exists(self.viewData_file) and 
                os.path.exists(self.postData_file))

    def merge_datasets(self) -> pd.DataFrame:
        user_df = pd.read_csv(self.userData_file)
        view_df = pd.read_csv(self.viewData_file)
        post_df = pd.read_csv(self.postData_file)

        merge_file_df = pd.merge(user_df, view_df, on="user_id", how="left")
        final_merge_file_df = pd.merge(merge_file_df, post_df, on="post_id", how="left")
        final_merge_file_df = self.normalize_column_names(final_merge_file_df, self.column_mapping)

        final_merge_file_df["genres"] = final_merge_file_df["genres"].str.replace("|", " ", regex=False)
        final_merge_file_df["timestamp"] = pd.to_datetime(final_merge_file_df["timestamp"])
        final_merge_file_df["timestamp"] = final_merge_file_df["timestamp"].astype("int64") // 10**9

        final_merge_file_df = final_merge_file_df.drop(columns=["avatar"])
        final_merge_file_df = final_merge_file_df.drop_duplicates(subset=["userID", "itemID"], keep="first")

        # generate ratings
        final_merge_file_df = frequency_based_rating_gen.frequency_based_rating_gen(final_merge_file_df, user_col="userID", category_col="genres")

        return final_merge_file_df


def loader(
    dataset_name: str = "movielens",
    want_col: Optional[List[str]] = None,
    num_rows: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
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
    return loader_instance.load_dataset(want_col, num_rows, seed)
