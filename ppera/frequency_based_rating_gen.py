# frequency_based_rating_gen.py
from typing import Optional  # Import Optional for type hinting

import numpy as np
import pandas as pd


# Added seed parameter
def frequency_based_rating_gen(
    df: pd.DataFrame,
    user_col: str = "userID",
    category_col: str = "genres",
    seed: Optional[int] = 42,  # Add seed parameter (optional)
) -> pd.DataFrame:
    """
    Calculates frequency-based ratings for user-category interactions within a DataFrame.

    Ratings are normalized per user based on their interaction frequency across categories
    and scaled to a 1-5 integer range.

    Args:
        df (pd.DataFrame): Input DataFrame containing user-item interactions.
                           Must include columns specified by user_col and category_col.
        user_col (str): The name of the user identifier column. Defaults to 'userID'.
        category_col (str): The name of the category/genre column. Defaults to 'genres'.
        seed (Optional[int]): An optional random seed for numpy to ensure reproducibility
                              if any random operations were involved (currently none, but good practice).
                              Defaults to None (no seed set explicitly here).

    Returns:
        pd.DataFrame: The input DataFrame with an added 'rating' column.
    """
    # Set the numpy random seed if provided
    if seed is not None:
        print(f"Setting numpy random seed to: {seed}")
        np.random.seed(seed)
    # Note: The current rating logic below is deterministic and doesn't use np.random directly.
    # However, setting the seed ensures consistency if future changes introduce randomness
    # or if other parts of a larger pipeline rely on the numpy random state.

    print("Calculating category counts per user...")
    # Calculate interaction counts per user-category using specified column names
    user_category_counts = df.groupby([user_col, category_col]).size().reset_index(name="category_count")

    print("Calculating user min/max category counts for normalization...")
    # Calculate min/max counts per user for normalization
    user_stats = user_category_counts.groupby(user_col)["category_count"].agg(["min", "max"]).reset_index()
    user_stats.rename(columns={"min": "user_min_count", "max": "user_max_count"}, inplace=True)

    # Merge counts and stats back
    user_category_counts = pd.merge(user_category_counts, user_stats, on=user_col)

    print("Calculating normalized ratings (1-5 scale)...")
    # Calculate normalized score (0 to 1) and then scale to 1-5
    epsilon = 1e-6  # Add epsilon to avoid division by zero if min == max for a user
    user_range = user_category_counts["user_max_count"] - user_category_counts["user_min_count"]

    # Calculate normalized score (0 if range is 0, otherwise normalized value)
    user_category_counts["normalized_score"] = np.where(
        user_range < epsilon,
        0,  # Assign 0 if min == max (will result in rating 1 or 3 depending on logic below)
        (user_category_counts["category_count"] - user_category_counts["user_min_count"]) / (user_range + epsilon),
    )

    # Scale to 1-5
    # If min == max (user interacted same amount in all their categories, or only one category), assign a middle rating (e.g., 3)
    user_category_counts["rating"] = np.where(
        user_range < epsilon,
        3,  # Assign a default middle rating if user range is zero
        1 + 4 * user_category_counts["normalized_score"],  # Scale 0-1 -> 1-5
    )

    # Round to nearest integer rating and ensure it's within [1, 5]
    user_category_counts["rating"] = user_category_counts["rating"].round().astype(int)
    user_category_counts["rating"] = user_category_counts["rating"].clip(1, 5)  # Ensure ratings stay within 1-5 bounds

    # Select relevant columns for merging back to original df
    ratings_map = user_category_counts[[user_col, category_col, "rating"]]

    print("Merging ratings back into the dataframe...")
    # Merge ratings back into the original dataframe passed as input
    df_with_ratings = pd.merge(df, ratings_map, on=[user_col, category_col], how="left")

    # --- Verification ---
    print("\nSample of data with generated ratings:")
    display_cols = [user_col, "itemID", category_col, "rating"] if "itemID" in df_with_ratings.columns else [user_col, category_col, "rating"]
    print(df_with_ratings[display_cols].head(10))

    print("\nGenerated Rating Distribution:")
    print(df_with_ratings["rating"].value_counts().sort_index())

    missing_ratings = df_with_ratings["rating"].isnull().sum()
    if missing_ratings > 0:
        print(f"\nWarning: Found {missing_ratings} rows with missing ratings!")
    else:
        print("\nSuccessfully generated ratings for all interaction rows.")
    # --- End Verification ---

    return df_with_ratings
