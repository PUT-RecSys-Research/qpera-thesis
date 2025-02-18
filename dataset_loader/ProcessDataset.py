import AmazonSalesDataset
import MovieLensDataset
import PostRecommendationsDataset


def process_dataset(dataset_name):
  """
    Processes a dataset (either AmazonSales or MovieLens) and demonstrates data
    loading, splitting, and hiding information.

    Args:
        dataset_name:  The name of the dataset ("amazonsales" or "movielens").
                        Case-insensitive.
    """

  if dataset_name == "amazonsales":
    loader = AmazonSalesDataset.AmazonSalesDataset()
        

  elif dataset_name == "movielens":
    loader = MovieLensDataset.MovieLensDataset()
        

  elif dataset_name == "postrecommendations":
    loader = PostRecommendationsDataset.PostRecommendationsDataset()
        
  else:
    raise ValueError(f"Invalid dataset name: {dataset_name}")
  
  while True:
    try:
      input1 = int(input("(1) Load dataset, (2) Split dataset, (3) Hide information, (0) Exit: "))
      match input1:
        case 1:
          while True:
            try:
              input2 = int(input("(1) Load dataset, (2) Load useful columns, (0) Exit: "))
              match input2:
                case 1:
                  dataset = loader.load_dataset()
                  print(dataset.head())
                  break
                case 2:
                  dataset = loader.load_dataset_useful_columns()
                  print(dataset.head())
                  break
                case 0:
                  break
                case _:
                  print("Invalid selection. Please choose 1, 2, or 0 to exit.")
            except ValueError:
              print("Invalid input. Please enter a number (1, 2, or 0).")
          break
        case 2:
          train_data, test_data = loader.get_train_test_split()
          print(f"Train data shape: {train_data.shape}")
          print(f"Test data shape: {test_data.shape}")
          break
        case 3:
          while True:
            try:
              input3 = int(input("(1) Hide column, (2) Hide records randomly, (3) Hide selective records, (4) Hide values in column, (5) Hide values in multiple columns, (0) Exit: "))
              dataset_choice = input("Choose the dataset to modify: (1) Whole dataset, (2) Training dataset, (3) Test dataset: ")

              if dataset_choice == "1":
                  dataset = loader.load_dataset()
              else:
                  train_data, test_data = loader.get_train_test_split()
                  if dataset_choice == "2":
                      dataset = train_data
                  elif dataset_choice == "3":
                      dataset = test_data
                  else:
                      print("Invalid choice. Defaulting to whole dataset.")
                      dataset = loader.load_dataset()

              match input3:
                case 1:
                  column_name = input("Enter the name of the column to hide: ")
                  dataset_hidden = loader.hide_information(dataset, hide_type="columns", columns_to_hide=column_name)
                  print(dataset_hidden)
                  break
                case 2:
                  fraction = float(input("Enter the fraction of records to hide (e.g., 0.5 to hide 50%): "))
                  dataset_hidden = loader.hide_information(dataset, hide_type="records_random", fraction_to_hide=fraction)
                  print(dataset_hidden)
                  break
                case 3:
                  records = input("Enter the indices of the records to hide (e.g., 1,3): ")
                  records_to_hide = [int(i) for i in records.split(",")]
                  dataset_hidden = loader.hide_information(dataset, hide_type="records_selective", records_to_hide=records_to_hide)
                  print(dataset_hidden)
                  break
                case 4:
                  column_name = input("Enter the name of the column to hide: ")
                  fraction = float(input("Enter the fraction of values to hide (e.g., 0.3 to hide 30%): "))
                  dataset_hidden = loader.hide_information(dataset, hide_type="values_in_column", columns_to_hide=column_name, fraction_to_hide=fraction)
                  print(dataset_hidden)
                  break
                case 5:
                  columns = input("Enter the names of the columns to hide (e.g., rating,genre): ")
                  columns_to_hide = columns.split(",")
                  fraction = float(input("Enter the fraction of values to hide (e.g., 0.3 to hide 30%): "))
                  dataset_hidden = loader.hide_information(dataset, hide_type="values_in_column", columns_to_hide=columns_to_hide, fraction_to_hide=fraction)
                  print(dataset_hidden)
                  break
                case 0:
                  break
                case _:
                  print("Invalid selection. Please choose 1, 2, 3, 4, 5, or 0 to exit.")
            except ValueError:
              print("Invalid input. Please enter a number (1, 2, 3, 4, 5, or 0).")
          break  
        case 0:
          print("Exit")
          break
        case _:
          print("Invalid selection. Please choose 1, 2, 3, or 0 to exit.")
    except ValueError:
      print("Invalid input. Please enter a number (1, 2, 3, or 0).")