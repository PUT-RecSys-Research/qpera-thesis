import os
import pandas as pd

from BaseDatasetLoader import BaseDatasetLoader





class PostRecommendationsDataset(BaseDatasetLoader):
    """
    Dataset loader for the PostRecommendation dataset.
    """

    def __init__(self, data_path: str = "datasets/PostRecommendations"):
        super().__init__(data_path)
        self.userData_file = f"{self.data_path}/user_data.csv"
        self.viewData_file = f"{self.data_path}/view_data.csv"
        self.postData_file = f"{self.data_path}/post_data.csv"

        self.user_view_post_merge = f"{self.data_path}/user_view_post_merge.csv"

        self.test_path = "datasets/PostRecommendations/testDataset"
        self.train_path = "datasets/PostRecommendations/trainDataset"

    def merge_datasets(self) -> pd.DataFrame:
        user_df = pd.read_csv(self.userData_file)
        view_df = pd.read_csv(self.viewData_file)
        post_df = pd.read_csv(self.postData_file)

        merge_file = pd.merge(user_df, view_df, on="user_id", how="left")
        final_merge_file = pd.merge(merge_file, post_df, on="post_id", how="left")
        final_merge_file.to_csv(self.user_view_post_merge, index=False)
        return final_merge_file

    def load_dataset(self) -> pd.DataFrame:
        # Check if the merged file exists
        if os.path.exists(self.user_view_post_merge):
            print("Loading cached merged dataset...")
            dataset_df = pd.read_csv(self.user_view_post_merge)
        else:
            print("Merged dataset not found.  Merging and saving...")
            dataset_df = self.merge_datasets()  # Merge and save
        return dataset_df

    def load_dataset_useful_columns(self) -> pd.DataFrame:
        userData_df = pd.read_csv(self.userData_file)
        return userData_df[["user_id", "first_name", "city"]]