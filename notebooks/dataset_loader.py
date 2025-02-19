import os
import pandas as pd
from abc import ABC
from typing import List, Optional


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """
  
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.train_path = None
        self.test_path = None
    
    def load_dataset(self, columns: Optional[List[int]] = None) -> pd.DataFrame:
        if os.path.exists(self.merge_file):
            # print("Loading cached merged dataset...")
            dataset_df = pd.read_csv(self.merge_file)
        else:
            # print("Merged dataset not found.  Merging and saving...")
            dataset_df = self.merge_datasets()

        if columns is not None:
            if not all(isinstance(c, str) for c in columns):
                raise TypeError("The 'columns' argument must be a list of strings (column names).")
            
            invalid_columns = [col for col in columns if col not in dataset_df.columns]
            if invalid_columns:
                raise KeyError(f"The following column names were not found in the DataFrame: {invalid_columns}")

            dataset_df = dataset_df.loc[:, columns]
        return dataset_df
    
class AmazonSalesDataset(BaseDatasetLoader):
    """
    Dataset loader for the Amazon Sales dataset.
    """

    def __init__(self, data_path: str = "datafiles/AmazonSales"):
        super().__init__(data_path)
        self.dataset = f"{self.data_path}/amazon.csv"
        self.merge_file = f"{self.data_path}/merge_file.csv"
    
    def normalize_column_names(self, df):
        df = df.rename(columns={"user_id": "userID", "product_id": "itemID"})
        return df

    def merge_datasets(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset)
        df = self.normalize_column_names(df)
        df.to_csv(self.merge_file, index=False)
        return df

class MovieLensDataset(BaseDatasetLoader):
    """
    Dataset loader for the MovieLens 20M dataset.
    """

    def __init__(self, data_path: str = "datafiles/MovieLens"):
        super().__init__(data_path)
        self.ratings_file = f"{self.data_path}/rating.csv"
        self.movies_file = f"{self.data_path}/movie.csv"
        self.tag_file = f"{self.data_path}/tag.csv"

        self.merge_file = f"{self.data_path}/merge_file.csv"

    def normalize_column_names(self, df):
        df = df.rename(columns={"userId": "userID", "movieId": "itemID"})
        return df

    def merge_datasets(self) -> pd.DataFrame:
        ratings_df = pd.read_csv(self.ratings_file)
        movies_df = pd.read_csv(self.movies_file)
        tags_df = pd.read_csv(self.tag_file)
        
        tags_df = tags_df.rename(columns={'timestamp': 'tag_timestamp'})

        merge_file = pd.merge(ratings_df, movies_df, on="movieId", how="left")
        final_merge_file = pd.merge(merge_file, tags_df, on=["movieId", "userId"], how="left")
        final_merge_file = self.normalize_column_names(final_merge_file)

        final_merge_file.to_csv(self.merge_file, index=False)
        return final_merge_file
    
class PostRecommendationsDataset(BaseDatasetLoader):
    """
    Dataset loader for the PostRecommendation dataset.
    """

    def __init__(self, data_path: str = "datafiles/PostRecommendations"):
        super().__init__(data_path)
        self.userData_file = f"{self.data_path}/user_data.csv"
        self.viewData_file = f"{self.data_path}/view_data.csv"
        self.postData_file = f"{self.data_path}/post_data.csv"

        self.merge_file = f"{self.data_path}/merge_file.csv"


    def normalize_column_names(self, df):
        df = df.rename(columns={"user_id": "userID", "post_id": "itemID"})
        return df
    
    def merge_datasets(self) -> pd.DataFrame:
        user_df = pd.read_csv(self.userData_file)
        view_df = pd.read_csv(self.viewData_file)
        post_df = pd.read_csv(self.postData_file)

        merge_file = pd.merge(user_df, view_df, on="user_id", how="left")
        final_merge_file = pd.merge(merge_file, post_df, on="post_id", how="left")
        final_merge_file = self.normalize_column_names(final_merge_file)
        final_merge_file.drop_duplicates(inplace=True)

        final_merge_file.to_csv(self.merge_file, index=False)
        return final_merge_file

def prepare_dataset():
    AmazonSalesDataset().load_dataset()
    MovieLensDataset().load_dataset()
    PostRecommendationsDataset().load_dataset()


def loader(dataset_name = "movielens", want_col= ['userID', 'itemID']):
    prepare_dataset()
    if dataset_name == "amazonsales":
        loader = AmazonSalesDataset()
    elif dataset_name == "movielens":
        loader = MovieLensDataset()
    elif dataset_name == "postrecommendations":
        loader = PostRecommendationsDataset()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    dataset = loader.load_dataset(want_col)
    return dataset

# prepare_dataset()
dataset = loader("amazonsales")
print(dataset)