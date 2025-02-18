import os
import pandas as pd


from BaseDatasetLoader import BaseDatasetLoader
#from typing import Optional, Tuple, Dict, List



class MovieLensDataset(BaseDatasetLoader):
    """
    Dataset loader for the MovieLens 20M dataset.
    """

    def __init__(self, data_path: str = "datasets/MovieLens"):
        super().__init__(data_path)
        self.ratings_file = f"{self.data_path}/rating.csv"
        self.movies_file = f"{self.data_path}/movie.csv"
        self.tag_file = f"{self.data_path}/tag.csv"

        self.rating_movie_tag_merge = f"{self.data_path}/rating_movie_tag_merge.csv"

        self.test_path = "datasets/MovieLens/testDataset"
        self.train_path = "datasets/MovieLens/trainDataset"

    def merge_datasets(self) -> pd.DataFrame:
        ratings_df = pd.read_csv(self.ratings_file)
        movies_df = pd.read_csv(self.movies_file)
        tags_df = pd.read_csv(self.tag_file)
        
        tags_df = tags_df.rename(columns={'timestamp': 'tag_timestamp'})

        merge_file = pd.merge(ratings_df, movies_df, on="movieId", how="left")
        final_merge_file = pd.merge(merge_file, tags_df, on=["movieId", "userId"], how="left")

        final_merge_file.to_csv(self.rating_movie_tag_merge, index=False)
        return final_merge_file

    def load_dataset(self) -> pd.DataFrame:
        # Check if the merged file exists
        if os.path.exists(self.rating_movie_tag_merge):
            print("Loading cached merged dataset...")
            dataset_df = pd.read_csv(self.rating_movie_tag_merge)
        else:
            print("Merged dataset not found.  Merging and saving...")
            dataset_df = self.merge_datasets()  # Merge and save
        return dataset_df

#     def load_item_features(self) -> pd.DataFrame:
#         movies_df = pd.read_csv(self.movies_file)
#         # Convert genres to a list of genres
#         movies_df["genres"] = movies_df["genres"].str.split("|")
#         return movies_df[["movieId", "genres", "title"]] # Include title

# # Jak dany użytkownik ocenił poszczególne filmy
#     def get_user_item_interactions(self) -> Dict[int, List[Tuple[int, float]]]:
#         ratings_df = self.load_dataset()
#         interactions = {}
#         for user_id, group in ratings_df.groupby("userId"):
#             interactions[user_id] = list(zip(group["movieId"], group["rating"]))
#         return interactions
    
#     def get_item_features_for_item(self, item_id: int) -> Optional[Dict]:
#         """
#         (Optional) Get features for a single item.  Useful for L2R or RL
#         when you need to represent the current state.

#         Args:
#             item_id: The ID of the item.

#         Returns:
#             A dictionary of item features, or None if the item is not found.
#         """
#         item_features_df = self.load_item_features()
#         if item_id in item_features_df['movieId'].values:
#             return item_features_df[item_features_df['movieId'] == item_id].iloc[0].to_dict()
#         else:
#             return None

#     def get_user_history(self, user_id: int) -> List[Tuple[int, float]]:
#         """
#         (Optional) Get the interaction history for a single user. Useful for RL.

#         Args:
#             user_id: The ID of the user.

#         Returns:
#             A list of (item_id, rating) tuples.
#         """
#         interactions = self.get_user_item_interactions()
#         return interactions.get(user_id, [])