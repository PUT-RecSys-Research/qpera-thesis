import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from . import frequency_based_rating_gen, rating_timestamp_gen


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders that handles common functionality
    for loading, merging, and processing recommendation datasets.
    """

    def __init__(self, raw_data_path: str, processed_data_path: str, merge_file_name: str = "merge_file.csv"):
        """
        Initialize the dataset loader.

        Args:
            raw_data_path: Directory containing raw CSV files
            processed_data_path: Directory for processed files
            merge_file_name: Name of the merged dataset file
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.merge_file = os.path.join(self.processed_data_path, merge_file_name)

        # Ensure processed directory exists
        os.makedirs(self.processed_data_path, exist_ok=True)

        # Validate that required files exist locally
        if not self._check_local_files_exist():
            raise FileNotFoundError(
                f"Required dataset files not found in {self.raw_data_path}. Please run 'make check-datasets' to download them automatically."
            )

    @abstractmethod
    def _check_local_files_exist(self) -> bool:
        """Check if required dataset files exist locally. Must be implemented by subclasses."""
        return False

    def load_dataset(
        self,
        columns: Optional[List[str]] = None,
        num_rows: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load the dataset with optional column selection and row limiting.

        Args:
            columns: List of column names to return
            num_rows: Number of rows to load (sequential from beginning)
            seed: Random seed for reproducibility (used in filename for caching)

        Returns:
            Pandas DataFrame with the requested data
        """
        # Check for pre-processed cached file
        if num_rows is not None and seed is not None:
            cached_file = self._get_cached_filename(num_rows, seed)
            if os.path.exists(cached_file):
                print(f"Loading cached file: {cached_file}")
                dataset_df = pd.read_csv(cached_file)
                return self._apply_column_selection(dataset_df, columns)

        # Load or create the main merged file
        dataset_df = self._load_or_create_merged_file(num_rows)

        # Cache the limited dataset for future use
        if num_rows is not None and seed is not None and len(dataset_df) <= num_rows:
            self._save_cached_file(dataset_df, num_rows, seed)

        # Apply column selection if requested
        return self._apply_column_selection(dataset_df, columns)

    def _get_cached_filename(self, num_rows: int, seed: int) -> str:
        """Generate filename for cached dataset with specific parameters."""
        return self.merge_file.replace(".csv", f"_r{num_rows}_s{seed}.csv")

    def _load_or_create_merged_file(self, num_rows: Optional[int]) -> pd.DataFrame:
        """Load existing merged file or create new one from raw data."""
        if os.path.exists(self.merge_file):
            print(f"Loading existing merged file: {self.merge_file}")
            return self._load_with_row_limit(self.merge_file, num_rows)
        else:
            print(f"Creating merged file from {self.raw_data_path}")
            dataset_df = self.merge_datasets()
            dataset_df.to_csv(self.merge_file, index=False)

            # Apply row limiting after merging if needed
            if num_rows is not None and len(dataset_df) > num_rows:
                print(f"Limiting merged dataset to first {num_rows} rows")
                dataset_df = dataset_df.head(num_rows)

            return dataset_df

    def _load_with_row_limit(self, file_path: str, num_rows: Optional[int]) -> pd.DataFrame:
        """Load CSV file with optional row limiting."""
        if num_rows is not None:
            try:
                dataset_df = pd.read_csv(file_path, nrows=num_rows)
                print(f"Loaded first {num_rows} rows sequentially")
                return dataset_df
            except ValueError:
                # Fallback if file has fewer rows than requested
                dataset_df = pd.read_csv(file_path)
                print(f"File has fewer than {num_rows} rows, loaded all {len(dataset_df)} rows")
                return dataset_df
        else:
            return pd.read_csv(file_path)

    def _save_cached_file(self, dataset_df: pd.DataFrame, num_rows: int, seed: int) -> None:
        """Save limited dataset to cache for future use."""
        cached_file = self._get_cached_filename(num_rows, seed)
        if not os.path.exists(cached_file):
            dataset_df.to_csv(cached_file, index=False)
            print(f"Saved cached dataset: {cached_file}")

    def _apply_column_selection(self, dataset_df: pd.DataFrame, columns: Optional[List[str]]) -> pd.DataFrame:
        """Apply column selection if specified."""
        if columns is not None:
            self.validate_columns(dataset_df, columns)
            return dataset_df[columns]
        return dataset_df

    @abstractmethod
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge necessary data files to create the final dataset.
        Must be implemented by subclasses.

        Returns:
            Merged pandas DataFrame
        """
        pass

    @staticmethod
    def normalize_column_names(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Rename columns according to the provided mapping."""
        return df.rename(columns=column_mapping)

    @staticmethod
    def validate_columns(df: pd.DataFrame, columns: List[str]) -> None:
        """Validate that requested columns exist in the DataFrame."""
        if not all(isinstance(c, str) for c in columns):
            raise TypeError("The 'columns' argument must be a list of strings.")

        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            raise KeyError(f"Column(s) not found in DataFrame: {invalid_columns}")


class AmazonSalesDataset(BaseDatasetLoader):
    """Dataset loader for Amazon Sales data with rating generation."""

    def __init__(self, raw_data_path: str = "datasets/AmazonSales", processed_data_path: str = "ppera/datasets/AmazonSales"):
        # Initialize file paths before calling parent constructor
        self.dataset_file = os.path.join(raw_data_path, "amazon.csv")
        self.column_mapping = {
            "user_id": "userID",
            "product_id": "itemID",
            "category": "genres",
            "product_name": "title",
            "predicted_rating": "rating",
        }

        super().__init__(raw_data_path, processed_data_path)

    def _check_local_files_exist(self) -> bool:
        """Check if Amazon Sales dataset file exists."""
        return os.path.exists(self.dataset_file)

    def merge_datasets(self) -> pd.DataFrame:
        """Process Amazon Sales dataset with rating and timestamp generation."""
        # Generate ratings and timestamps
        rating_timestamp_gen.rating_timestamp_gen(self.dataset_file, self.dataset_file)

        # Load and process data
        df = pd.read_csv(self.dataset_file)
        df = self.normalize_column_names(df, self.column_mapping)

        # Remove duplicates and clean ratings
        df = df.drop_duplicates(subset=["userID", "itemID"], keep="first")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        # Combine category and product description for genres
        df["genres"] = df[["genres", "about_product"]].astype(str).agg(" | ".join, axis=1).str.strip(" |").str.replace("|", " ", regex=False)

        # Convert timestamp to Unix format
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64") // 10**9

        # Remove unnecessary columns
        columns_to_drop = [
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
        df = df.drop(columns=columns_to_drop)

        return df


class MovieLensDataset(BaseDatasetLoader):
    """Dataset loader for MovieLens data with ratings, movies, and tags."""

    def __init__(self, raw_data_path: str = "datasets/MovieLens", processed_data_path: str = "ppera/datasets/MovieLens"):
        # Initialize file paths before calling parent constructor
        self.ratings_file = os.path.join(raw_data_path, "rating.csv")
        self.movies_file = os.path.join(raw_data_path, "movie.csv")
        self.tag_file = os.path.join(raw_data_path, "tag.csv")
        self.column_mapping = {"userId": "userID", "movieId": "itemID"}

        super().__init__(raw_data_path, processed_data_path)

    def _check_local_files_exist(self) -> bool:
        """Check if all required MovieLens files exist."""
        return os.path.exists(self.ratings_file) and os.path.exists(self.movies_file) and os.path.exists(self.tag_file)

    def merge_datasets(self) -> pd.DataFrame:
        """Merge MovieLens ratings, movies, and tags data."""
        # Verify files exist
        for file_path, name in [(self.ratings_file, "ratings"), (self.movies_file, "movies"), (self.tag_file, "tags")]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"MovieLens {name} file not found at {file_path}")

        print("Loading MovieLens files:")
        print(f"  - Ratings: {self.ratings_file}")
        print(f"  - Movies: {self.movies_file}")
        print(f"  - Tags: {self.tag_file}")

        # Load data files
        ratings_df = pd.read_csv(self.ratings_file)
        movies_df = pd.read_csv(self.movies_file)
        tags_df = pd.read_csv(self.tag_file)

        print(f"Loaded shapes - Ratings: {ratings_df.shape}, Movies: {movies_df.shape}, Tags: {tags_df.shape}")

        # Rename tag timestamp to avoid conflicts
        tags_df = tags_df.rename(columns={"timestamp": "tag_timestamp"})

        # Merge datasets
        merged_df = pd.merge(ratings_df, movies_df, on="movieId", how="left")
        final_df = pd.merge(merged_df, tags_df, on=["movieId", "userId"], how="left")

        # Normalize column names
        final_df = self.normalize_column_names(final_df, self.column_mapping)

        # Clean genres format
        final_df["genres"] = final_df["genres"].str.replace("|", " ", regex=False)

        # Convert timestamp to Unix format
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
        final_df["timestamp"] = final_df["timestamp"].astype("int64") // 10**9

        # Remove duplicates and unnecessary columns
        final_df = final_df.drop_duplicates(subset=["userID", "itemID", "rating"], keep="first")
        final_df = final_df.drop(columns=["tag_timestamp", "tag"])

        print(f"Final merged dataset shape: {final_df.shape}")
        return final_df


class PostRecommendationsDataset(BaseDatasetLoader):
    """Dataset loader for Post Recommendations data with frequency-based rating generation."""

    def __init__(self, raw_data_path: str = "datasets/PostRecommendations", processed_data_path: str = "ppera/datasets/PostRecommendations"):
        # Initialize file paths before calling parent constructor
        self.userData_file = os.path.join(raw_data_path, "user_data.csv")
        self.viewData_file = os.path.join(raw_data_path, "view_data.csv")
        self.postData_file = os.path.join(raw_data_path, "post_data.csv")
        self.column_mapping = {
            "user_id": "userID",
            "post_id": "itemID",
            "time_stamp": "timestamp",
            "category": "genres",
        }

        super().__init__(raw_data_path, processed_data_path)

    def _check_local_files_exist(self) -> bool:
        """Check if all required Post Recommendations files exist."""
        return os.path.exists(self.userData_file) and os.path.exists(self.viewData_file) and os.path.exists(self.postData_file)

    def merge_datasets(self) -> pd.DataFrame:
        """Merge Post Recommendations data and generate ratings."""
        # Load data files
        user_df = pd.read_csv(self.userData_file)
        view_df = pd.read_csv(self.viewData_file)
        post_df = pd.read_csv(self.postData_file)

        # Merge datasets
        merged_df = pd.merge(user_df, view_df, on="user_id", how="left")
        final_df = pd.merge(merged_df, post_df, on="post_id", how="left")

        # Normalize column names
        final_df = self.normalize_column_names(final_df, self.column_mapping)

        # Clean genres format and timestamp
        final_df["genres"] = final_df["genres"].str.replace("|", " ", regex=False)
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
        final_df["timestamp"] = final_df["timestamp"].astype("int64") // 10**9

        # Remove unnecessary columns and duplicates
        final_df = final_df.drop(columns=["avatar"])
        final_df = final_df.drop_duplicates(subset=["userID", "itemID"], keep="first")

        # Generate ratings based on user interaction frequency
        final_df = frequency_based_rating_gen.frequency_based_rating_gen(final_df, user_col="userID", category_col="genres")

        return final_df


def loader(
    dataset_name: str = "movielens",
    want_col: Optional[List[str]] = None,
    num_rows: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a specified dataset with optional parameters.

    Args:
        dataset_name: Name of the dataset to load
        want_col: List of column names to return
        num_rows: Number of rows to load (sequential from beginning)
        seed: Random seed for reproducibility

    Returns:
        Pandas DataFrame with the requested data

    Raises:
        ValueError: If an invalid dataset name is provided
    """
    # Registry of available dataset loaders
    loaders = {
        "amazonsales": AmazonSalesDataset,
        "movielens": MovieLensDataset,
        "postrecommendations": PostRecommendationsDataset,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Invalid dataset name: {dataset_name}. Available: {list(loaders.keys())}")

    # Initialize and load dataset
    loader_instance = loaders[dataset_name]()
    return loader_instance.load_dataset(want_col, num_rows, seed)
