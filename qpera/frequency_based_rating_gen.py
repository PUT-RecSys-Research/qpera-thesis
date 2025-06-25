# frequency_based_rating_gen.py
from typing import Optional

import numpy as np
import pandas as pd


def frequency_based_rating_gen(
    df: pd.DataFrame,
    user_col: str = "userID",
    category_col: str = "genres",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate frequency-based ratings for user-category interactions.

    Ratings are normalized per user based on their interaction frequency across categories
    and scaled to a 1-5 integer range. Users with higher interaction frequency in a category
    receive higher ratings for that category.

    Args:
        df: Input DataFrame containing user-item interactions.
            Must include columns specified by user_col and category_col.
        user_col: Name of the user identifier column
        category_col: Name of the category/genre column
        seed: Random seed for reproducibility (currently unused but reserved for future use)

    Returns:
        DataFrame with an added 'rating' column containing generated ratings (1-5 scale)

    Raises:
        KeyError: If required columns are missing from the DataFrame
        ValueError: If DataFrame is empty or contains invalid data
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    if user_col not in df.columns or category_col not in df.columns:
        raise KeyError(f"Required columns '{user_col}' and/or '{category_col}' not found in DataFrame")

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    print("Generating frequency-based ratings...")

    # Calculate interaction counts per user-category combination
    user_category_counts = df.groupby([user_col, category_col]).size().reset_index(name="category_count")

    # Calculate normalization statistics per user
    user_stats = (
        user_category_counts.groupby(user_col)["category_count"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "user_min_count", "max": "user_max_count"})
    )

    # Merge counts with user statistics
    user_category_counts = pd.merge(user_category_counts, user_stats, on=user_col)

    # Generate normalized ratings
    ratings_df = _calculate_normalized_ratings(user_category_counts)

    # Merge ratings back into original DataFrame
    ratings_map = ratings_df[[user_col, category_col, "rating"]]
    df_with_ratings = pd.merge(df, ratings_map, on=[user_col, category_col], how="left")

    # Validate results
    _validate_generated_ratings(df_with_ratings, user_col, category_col)

    return df_with_ratings


def _calculate_normalized_ratings(user_category_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate normalized ratings (1-5 scale) based on user interaction frequencies.

    Args:
        user_category_counts: DataFrame with user-category counts and min/max statistics

    Returns:
        DataFrame with added normalized ratings
    """
    epsilon = 1e-6  # Small value to prevent division by zero
    user_range = user_category_counts["user_max_count"] - user_category_counts["user_min_count"]

    # Calculate normalized score (0-1 range)
    user_category_counts["normalized_score"] = np.where(
        user_range < epsilon,
        0.5,  # Default middle score if user has uniform interaction pattern
        (user_category_counts["category_count"] - user_category_counts["user_min_count"]) / user_range,
    )

    # Scale normalized score to 1-5 rating range
    user_category_counts["rating"] = np.where(
        user_range < epsilon,
        3,  # Default middle rating for uniform interaction patterns
        1 + 4 * user_category_counts["normalized_score"],  # Linear scaling to 1-5 range
    )

    # Ensure ratings are integers within valid range
    user_category_counts["rating"] = user_category_counts["rating"].round().astype(int).clip(1, 5)

    return user_category_counts


def _validate_generated_ratings(df_with_ratings: pd.DataFrame, user_col: str, category_col: str) -> None:
    """
    Validate the generated ratings and display summary statistics.

    Args:
        df_with_ratings: DataFrame with generated ratings
        user_col: Name of user column
        category_col: Name of category column
    """
    # Display sample of generated data
    display_cols = [user_col, category_col, "rating"]
    if "itemID" in df_with_ratings.columns:
        display_cols.insert(-1, "itemID")

    print("\nSample of generated ratings:")
    print(df_with_ratings[display_cols].head(10))

    # Show rating distribution
    print("\nRating distribution:")
    rating_counts = df_with_ratings["rating"].value_counts().sort_index()
    for rating, count in rating_counts.items():
        percentage = (count / len(df_with_ratings)) * 100
        print(f"  Rating {rating}: {count:,} ({percentage:.1f}%)")

    # Check for missing ratings
    missing_ratings = df_with_ratings["rating"].isnull().sum()
    if missing_ratings > 0:
        print(f"\nWarning: {missing_ratings:,} rows have missing ratings!")
    else:
        print(f"\nSuccessfully generated ratings for all {len(df_with_ratings):,} interactions")

    # Display rating statistics per user sample
    print("\nRating statistics summary:")
    rating_stats = df_with_ratings["rating"].describe()
    print(f"  Mean: {rating_stats['mean']:.2f}")
    print(f"  Std:  {rating_stats['std']:.2f}")
    print(f"  Range: {rating_stats['min']:.0f}-{rating_stats['max']:.0f}")


def generate_ratings_for_dataset(df: pd.DataFrame, user_col: str = "userID", category_col: str = "genres", seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Convenience wrapper function for frequency-based rating generation.

    This function provides a more descriptive name and can be used as an alternative
    entry point to the main rating generation functionality.

    Args:
        df: Input DataFrame
        user_col: User identifier column name
        category_col: Category/genre column name
        seed: Random seed for reproducibility

    Returns:
        DataFrame with generated frequency-based ratings
    """
    return frequency_based_rating_gen(df, user_col, category_col, seed)
