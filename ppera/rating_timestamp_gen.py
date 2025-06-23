import re
from datetime import datetime, timedelta
from typing import Optional

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


def rating_timestamp_gen(input_file: str, output_file: str, seed: int = 42) -> None:
    """
    Generate sentiment-based ratings and timestamps for Amazon review data.
    
    Preprocesses text data from a CSV, calculates sentiment scores,
    maps them to ratings (including half-star ratings), adds timestamps,
    and saves the result to a new CSV.

    Args:
        input_file: Path to the input CSV file (Amazon reviews)
        output_file: Path to the output CSV file
        seed: Random seed for reproducible timestamp generation

    Raises:
        ValueError: If input/output files are not CSV, or input CSV is empty
        FileNotFoundError: If the input CSV file is not found
        TypeError: If the seed is not an integer
    """
    # Input validation
    _validate_inputs(input_file, output_file, seed)
    
    # Load and validate data
    df = _load_and_validate_data(input_file)
    
    # Preprocess text data
    df = _preprocess_text_data(df)
    
    # Generate sentiment-based ratings
    df = _generate_sentiment_ratings(df)
    
    # Add random timestamps
    df = _add_timestamps(df, seed)
    
    # Save results
    _save_results(df, output_file)


def _validate_inputs(input_file: str, output_file: str, seed: int) -> None:
    """Validate input parameters."""
    if not isinstance(input_file, str) or not input_file.endswith(".csv"):
        raise ValueError("Input file must be a CSV file (ending with .csv)")
    if not isinstance(output_file, str) or not output_file.endswith(".csv"):
        raise ValueError("Output file must be a CSV file (ending with .csv)")
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")


def _load_and_validate_data(input_file: str) -> pd.DataFrame:
    """Load CSV data and validate it's not empty."""
    try:
        df = pd.read_csv(input_file)
        if df.empty:
            raise ValueError("Input CSV file is empty")
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError("Input CSV file is empty")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")


def _preprocess_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess review text data by cleaning and combining title and content."""
    # Fill missing values
    df["review_title"] = df["review_title"].fillna("")
    df["review_content"] = df["review_content"].fillna("")
    
    # Apply text preprocessing
    df["review_title"] = df["review_title"].apply(_clean_text)
    df["review_content"] = df["review_content"].apply(_clean_text)
    df["combined_review"] = df["review_title"] + " " + df["review_content"]
    
    return df


def _clean_text(text: str) -> str:
    """Clean and normalize text data."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def _generate_sentiment_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Generate sentiment-based ratings using VADER sentiment analyzer."""
    # Download VADER lexicon if not available
    nltk.download("vader_lexicon", quiet=True)
    
    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Calculate sentiment scores
    df["sentiment_score"] = df["combined_review"].apply(
        lambda text: sid.polarity_scores(text)["compound"]
    )
    
    # Map sentiment scores to ratings
    df["predicted_rating"] = df["sentiment_score"].apply(_map_sentiment_to_rating)
    
    # Remove existing rating column if present
    if "rating" in df.columns:
        df = df.drop(columns=["rating"])
    
    return df


def _map_sentiment_to_rating(score: float) -> float:
    """
    Map sentiment scores (-1 to 1) to star ratings (0.5 to 5.0).
    
    Args:
        score: Sentiment compound score from VADER (-1 to 1)
        
    Returns:
        Star rating from 0.5 to 5.0 in 0.5 increments
    """
    if score >= 0.9:
        return 5.0
    elif score >= 0.7:
        return 4.5
    elif score >= 0.5:
        return 4.0
    elif score >= 0.3:
        return 3.5
    elif score >= 0.1:
        return 3.0
    elif score >= -0.1:
        return 2.5
    elif score >= -0.3:
        return 2.0
    elif score >= -0.5:
        return 1.5
    elif score >= -0.7:
        return 1.0
    else:
        return 0.5


def _add_timestamps(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add random timestamps to the dataset within a specified date range."""
    num_rows = len(df)
    np.random.seed(seed)
    
    # Define date range (2018-2022)
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2022, 12, 31)
    total_seconds = int((end_date - start_date).total_seconds())
    
    # Generate random timestamps
    random_seconds = np.random.randint(0, total_seconds, size=num_rows)
    timestamps = [start_date + timedelta(seconds=int(seconds)) for seconds in random_seconds]
    
    # Sort in reverse chronological order (newest first)
    timestamps.sort(reverse=True)
    
    # Format timestamps
    formatted_timestamps = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]
    df["timestamp"] = formatted_timestamps
    
    return df


def _save_results(df: pd.DataFrame, output_file: str) -> None:
    """Save processed data and display summary information."""
    df.to_csv(output_file, index=False)
    
    # Display summary
    print("Sample of processed data:")
    print(df[["combined_review", "sentiment_score", "predicted_rating", "timestamp"]].head())
    
    print(f"\nRating distribution:")
    rating_counts = df["predicted_rating"].value_counts().sort_index()
    for rating, count in rating_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Rating {rating}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nProcessed {len(df):,} reviews with sentiment-based ratings and timestamps")
    print(f"Results saved to: {output_file}")


def generate_amazon_ratings(
    input_file: str, 
    output_file: str, 
    seed: int = 42,
    date_range: Optional[tuple] = None
) -> pd.DataFrame:
    """
    Convenience wrapper function for Amazon review rating generation.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        seed: Random seed for reproducibility
        date_range: Optional tuple of (start_date, end_date) as datetime objects
        
    Returns:
        Processed DataFrame with generated ratings and timestamps
    """
    # Use the main function for processing
    rating_timestamp_gen(input_file, output_file, seed)
    
    # Return the processed data
    return pd.read_csv(output_file)


def analyze_sentiment_distribution(df: pd.DataFrame) -> dict:
    """
    Analyze the distribution of sentiment scores and ratings.
    
    Args:
        df: DataFrame with sentiment_score and predicted_rating columns
        
    Returns:
        Dictionary with sentiment analysis statistics
    """
    if "sentiment_score" not in df.columns or "predicted_rating" not in df.columns:
        raise ValueError("DataFrame must contain 'sentiment_score' and 'predicted_rating' columns")
    
    stats = {
        "sentiment_stats": {
            "mean": df["sentiment_score"].mean(),
            "std": df["sentiment_score"].std(),
            "min": df["sentiment_score"].min(),
            "max": df["sentiment_score"].max()
        },
        "rating_distribution": df["predicted_rating"].value_counts().sort_index().to_dict(),
        "sentiment_ranges": {
            "positive": len(df[df["sentiment_score"] > 0.1]),
            "neutral": len(df[(df["sentiment_score"] >= -0.1) & (df["sentiment_score"] <= 0.1)]),
            "negative": len(df[df["sentiment_score"] < -0.1])
        }
    }
    
    return stats
