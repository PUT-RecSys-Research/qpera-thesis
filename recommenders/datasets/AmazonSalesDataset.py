import os
import pandas as pd

from BaseDatasetLoader import BaseDatasetLoader





class AmazonSalesDataset(BaseDatasetLoader):
    """
    Dataset loader for the Amazon Sales dataset.
    """

    def __init__(self, data_path: str = "datasets/AmazonSales"):
        super().__init__(data_path)
        self.dataset_file = f"{self.data_path}/amazon.csv"

        self.train_path = os.path.join(self.data_path, "trainDataset")
        self.test_path = os.path.join(self.data_path, "testDataset")
    
    def load_dataset_useful_columns(self) -> pd.DataFrame:
        dataset_df = pd.read_csv(self.dataset_file)
        return dataset_df[["product_id", "product_name", "category", "rating", "user_id", "review_content"]]

    # To zostaje - split na test data i train data więc uniwersalne do każdego dataeu
    # def get_train_test_split(self, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     dataset_df = self.load_dataset()
    #     train_df = dataset_df.sample(frac=1 - test_size, random_state=seed)
    #     test_df = dataset_df.drop(train_df.index)
        
    #     save_path_train = self.AmazonSalesPathTrain
    #     save_path_test = self.AmazonSalesPathTest

    #     train_file = os.path.join(save_path_train, "train.csv")
    #     test_file = os.path.join(save_path_test, "test.csv")

    #     train_df.to_csv(train_file, index=False)
    #     test_df.to_csv(test_file, index=False)

    #     return train_df, test_df
    