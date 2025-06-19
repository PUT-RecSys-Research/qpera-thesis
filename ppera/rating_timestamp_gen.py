import re
from datetime import datetime, timedelta

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


def rating_timestamp_gen(input_file, output_file, seed=42):
    """
    Preprocesses text data from a CSV, calculates sentiment scores,
    maps them to ratings (including half-star ratings), adds timestamps,
    and saves the result to a new CSV.

    Args:
        input_file (str): Path to the input CSV file (Amazon reviews).
        output_file (str): Path to the output CSV file.
        seed (int):  Seed for random timestamp generation.

    Raises:
        ValueError: If input/output files are not CSV, or input CSV is empty.
        FileNotFoundError: If the input CSV file is not found.
        TypeError: If the seed is not an integer.
        Exception: For other errors during CSV processing.
    """

    # --- Input Validation ---
    if not isinstance(input_file, str) or not input_file.endswith(".csv"):
        raise ValueError("Input file must be a CSV file (ending with .csv)")
    if not isinstance(output_file, str) or not output_file.endswith(".csv"):
        raise ValueError("Output file must be a CSV file (ending with .csv)")
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")

    # --- Load Data and Handle Errors ---
    try:
        df = pd.read_csv(input_file)
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("Input CSV file is empty.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

    if df.empty:
        raise ValueError("Input CSV file is empty.")

    # --- Preprocessing ---
    df["review_title"] = df["review_title"].fillna("")
    df["review_content"] = df["review_content"].fillna("")

    def preprocess_text(text):
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
        return text

    df["review_title"] = df["review_title"].apply(preprocess_text)
    df["review_content"] = df["review_content"].apply(preprocess_text)
    df["combined_review"] = df["review_title"] + " " + df["review_content"]

    # --- Sentiment Analysis and Rating ---
    nltk.download("vader_lexicon", quiet=True)
    sid = SentimentIntensityAnalyzer()

    def get_sentiment_score(text):
        return sid.polarity_scores(text)["compound"]

    df["sentiment_score"] = df["combined_review"].apply(get_sentiment_score)

    def map_score_to_rating(score):
        # Map sentiment scores to ratings (1-5 with 0.5 increments)
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
            return 0.5  # Added for very negative sentiments

    df["predicted_rating"] = df["sentiment_score"].apply(map_score_to_rating)
    if "rating" in df.columns:
        df = df.drop(columns=["rating"])

    # --- Add Timestamps ---
    num_rows = len(df)
    np.random.seed(seed)
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2022, 12, 31)
    time_deltas = [timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(num_rows)]
    timestamps = [start_date + delta for delta in time_deltas]
    timestamps.sort(reverse=True)
    formatted_timestamps = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]
    df["timestamp"] = formatted_timestamps

    # --- Save to CSV ---
    df.to_csv(output_file, index=False)
    print(df.head())  # Display the first few rows for verification
    print(f"Processed data with ratings and timestamps saved to '{output_file}'")
