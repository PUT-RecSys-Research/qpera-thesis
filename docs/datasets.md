# Datasets

This project uses three main datasets to evaluate recommendation algorithms across different domains and provides automatic download capabilities.

## Overview

| Dataset | Domain | Size | Users | Items | Ratings | Kaggle Source |
|---------|--------|------|-------|-------|---------|---------------|
| [MovieLens](#movielens) | Movies | ~20M | ~138K | ~27K | ~20M | `grouplens/movielens-20m-dataset` |
| [Amazon Sales](#amazon-sales) | E-commerce | ~1.4M | ~1M | ~200K | ~1M | `karkavelrajaj/amazon-sales-dataset` |
| [Post Recommendations](#post-recommendations) | Social Media | Variable | ~10K | ~50K | Generated | `vatsalparsaniya/post-pecommendation` |

## Automatic Download Setup

### Prerequisites

1. **Install Kaggle API**:
```bash
pip install kaggle
```

2. **Configure Kaggle credentials**:
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. **Download datasets automatically**:
```bash
# Download all datasets at once
make check-datasets

# Or download individual datasets
python -m ppera.datasets_downloader movielens
python -m ppera.datasets_downloader amazonsales  
python -m ppera.datasets_downloader postrecommendations
```

## Dataset Details

### MovieLens

**Kaggle Dataset**: `grouplens/movielens-20m-dataset`  
**Local Directory**: `datasets/MovieLens/`

**Required Files**:
- `rating.csv` - User ratings (userId, movieId, rating, timestamp)
- `movie.csv` - Movie metadata (movieId, title, genres)
- `tag.csv` - User tags (userId, movieId, tag, timestamp)

**Data Processing**:
```python
# Column mapping
{"userId": "userID", "movieId": "itemID"}

# Genre processing
"Action|Adventure|Sci-Fi" → "Action Adventure Sci-Fi"

# Timestamp conversion
datetime → Unix timestamp

# Deduplication
Drop duplicates on ["userID", "itemID", "rating"]
```

**Example loading**:
```python
from ppera.datasets_loader import loader

# Load MovieLens data
data = loader("movielens", 
              want_col=["userID", "itemID", "rating", "title", "genres"],
              num_rows=10000, 
              seed=42)
```

### Amazon Sales

**Kaggle Dataset**: `karkavelrajaj/amazon-sales-dataset`  
**Local Directory**: `datasets/AmazonSales/`

**Required Files**:
- `amazon.csv` - Product sales data with user interactions

**Column Mapping**:
```python
{
    "user_id": "userID",
    "product_id": "itemID", 
    "category": "genres",
    "product_name": "title",
    "predicted_rating": "rating"
}
```

**Data Processing**:
```python
# Genre combination
df["genres"] = df[["category", "about_product"]].agg(" | ".join, axis=1)
df["genres"] = df["genres"].str.replace("|", " ")

# Rating normalization
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Timestamp generation (if missing)
rating_timestamp_gen.rating_timestamp_gen(file_path, output_path)

# Removed columns
["discounted_price", "actual_price", "discount_percentage", 
 "rating_count", "about_product", "user_name", "review_id", 
 "review_title", "review_content", "img_link", "product_link"]
```

### Post Recommendations

**Kaggle Dataset**: `vatsalparsaniya/post-pecommendation`  
**Local Directory**: `datasets/PostRecommendations/`

**Required Files**:
- `user_data.csv` - User profiles
- `view_data.csv` - User-post interactions  
- `post_data.csv` - Post metadata

**Column Mapping**:
```python
{
    "user_id": "userID",
    "post_id": "itemID",
    "time_stamp": "timestamp", 
    "category": "genres"
}
```

**Special Processing**:
```python
# Rating generation (frequency-based)
# Since this dataset lacks explicit ratings, they are generated based on:
# - User interaction frequency with genres
# - Popularity-based scoring
final_df = frequency_based_rating_gen.frequency_based_rating_gen(
    final_df, 
    user_col="userID", 
    category_col="genres"
)

# Deduplication
df.drop_duplicates(subset=["userID", "itemID"], keep="first")
```

## File Structure

```
personalization-privacy-and-explainability-of-recommendation-algorithms/
├── datasets/                           # Raw dataset storage
│   ├── MovieLens/
│   │   ├── rating.csv                 # User ratings
│   │   ├── movie.csv                  # Movie metadata  
│   │   └── tag.csv                    # User tags
│   ├── AmazonSales/
│   │   └── amazon.csv                 # Product interactions
│   └── PostRecommendations/
│       ├── user_data.csv              # User profiles
│       ├── view_data.csv              # Interactions
│       └── post_data.csv              # Post metadata
│
├── ppera/datasets/                     # Processed datasets cache
│   ├── MovieLens/
│   │   ├── merge_file.csv             # Full merged dataset
│   │   ├── merge_file_r1000_s42.csv   # Limited dataset (1000 rows, seed 42)
│   │   └── merge_file_r5000_s123.csv  # Limited dataset (5000 rows, seed 123)
│   ├── AmazonSales/
│   │   └── merge_file.csv
│   └── PostRecommendations/
│       └── merge_file.csv
│
└── ppera/rl_tmp/                       # RL-specific processed data
    ├── Movielens/
    │   ├── processed_dataset.pkl       # Knowledge graph entities/relations
    │   ├── kg.pkl                      # Knowledge graph structure
    │   ├── transe_embed.pkl            # TransE embeddings
    │   ├── train_label.pkl             # Training labels
    │   └── test_label.pkl              # Testing labels
    ├── AmazonSales/
    └── PostRecommendations/
```

## Dataset Loading API

### Basic Loading

```python
from ppera.datasets_loader import loader

# Load full dataset
data = loader("movielens")

# Load specific columns
data = loader("movielens", 
              want_col=["userID", "itemID", "rating", "genres"])

# Load limited rows (sequential, not random sampling)
data = loader("movielens", num_rows=5000, seed=42)
```

### Advanced Options

```python
# For specific algorithms
want_col_standard = ["userID", "itemID", "rating", "timestamp", "title", "genres"]

# CF/CBF algorithms
data_cf = loader("movielens", want_col_standard, num_rows=10000, seed=42)

# RL algorithms (often need more data)
data_rl = loader("movielens", want_col_standard, num_rows=14000, seed=42)
```

### Caching Behavior

The dataset loader implements intelligent caching:

1. **First load**: Merges raw files, saves `merge_file.csv`
2. **Subsequent loads**: Loads from cached merged file
3. **Row limiting**: Creates separate cached files with naming `merge_file_r{rows}_s{seed}.csv`
4. **Column selection**: Applied after loading, no separate caching

## Dataset Validation

### Automatic Validation

```python
# Built into each dataset loader
class MovieLensDataset(BaseDatasetLoader):
    def _check_local_files_exist(self) -> bool:
        return (os.path.exists(self.ratings_file) and 
                os.path.exists(self.movies_file) and 
                os.path.exists(self.tag_file))
```

### Manual Validation

```bash
# Check if datasets are properly downloaded
make check-datasets

# This will download missing datasets automatically
```

## Knowledge Graph Processing (RL)

For reinforcement learning algorithms, datasets undergo additional processing:

### Entity Extraction

```python
# From ppera/rl_preprocess.py
ENTITY_TYPES = {
    USERID: "user_id",      # User entities
    ITEMID: "item_id",      # Item entities  
    TITLE: "title",         # Title entities
    GENRES: "genres",       # Genre entities
    RATING: "rating"        # Rating value entities
}
```

### Relation Construction

```python
# Knowledge graph relations
RELATIONS = {
    WATCHED: [(user_idx, item_idx), ...],                    # User watched item
    BELONG_TO: [(item_idx, genre_idx), ...],                 # Item belongs to genre
    DESCRIBED_AS: [(title_idx, item_idx), ...],              # Title describes item
    RATED: [(user_idx, title_idx), ...],                     # User rated title
    USER_RATED_WITH_VALUE: [(user_idx, rating_idx), ...],    # User rated with value
    RATING_VALUE_FOR_ITEM: [(rating_idx, item_idx), ...]     # Rating value for item
}
```

### Path Patterns

```python
# Example path patterns for recommendation
PATH_PATTERNS = {
    1: ((None, USERID), (RATED, TITLE), (DESCRIBED_AS, ITEMID)),
    2: ((None, USERID), (USER_RATED_WITH_VALUE, RATING), (RATING_VALUE_FOR_ITEM, ITEMID)),
    3: ((None, USERID), (WATCHED, ITEMID), (BELONG_TO, GENRES)),
    # ... up to 15 different path types
}
```

## Custom Dataset Integration

### Step 1: Create Dataset Loader

```python
# ppera/datasets_loader.py
class CustomDatasetLoader(BaseDatasetLoader):
    def __init__(self, raw_data_path="datasets/CustomDataset", 
                 processed_data_path="ppera/datasets/CustomDataset"):
        # Set file paths BEFORE calling super().__init__
        self.data_file = os.path.join(raw_data_path, "interactions.csv")
        self.metadata_file = os.path.join(raw_data_path, "metadata.csv")
        self.column_mapping = {
            "user": "userID",
            "item": "itemID",
            "score": "rating",
            "category": "genres"
        }
        super().__init__(raw_data_path, processed_data_path)

    def _check_local_files_exist(self) -> bool:
        return (os.path.exists(self.data_file) and 
                os.path.exists(self.metadata_file))

    def merge_datasets(self) -> pd.DataFrame:
        # Load and merge your data files
        interactions = pd.read_csv(self.data_file)
        metadata = pd.read_csv(self.metadata_file)
        
        # Apply column mapping
        interactions = self.normalize_column_names(interactions, self.column_mapping)
        
        # Merge datasets
        merged = interactions.merge(metadata, on='itemID', how='left')
        
        # Standard processing
        merged = merged.drop_duplicates(subset=["userID", "itemID"], keep="first")
        merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce")
        
        # Genre processing (standardize format)
        if "genres" in merged.columns:
            merged["genres"] = merged["genres"].str.replace("|", " ", regex=False)
        
        # Timestamp processing (if needed)
        if "timestamp" not in merged.columns:
            # Generate timestamps if missing
            merged = rating_timestamp_gen.rating_timestamp_gen_df(merged)
        
        return merged
```

### Step 2: Register Dataset

```python
# Update loader() function in datasets_loader.py
def loader(dataset_name: str = "movielens", ...):
    loaders = {
        "amazonsales": AmazonSalesDataset,
        "movielens": MovieLensDataset,
        "postrecommendations": PostRecommendationsDataset,
        "customdataset": CustomDatasetLoader,  # Add here
    }
```

### Step 3: Add RL Support (Optional)

```python
# ppera/rl_utils.py
DATASET_DIR = {
    # ... existing datasets ...
    "customdataset": "./datasets/CustomDataset",
}

TMP_DIR = {
    # ... existing datasets ...
    "customdataset": "ppera/rl_tmp/CustomDataset", 
}

LABELS = {
    # ... existing datasets ...
    "customdataset": (
        TMP_DIR["customdataset"] + "/train_label.pkl",
        TMP_DIR["customdataset"] + "/test_label.pkl"
    ),
}
```

### Step 4: Add Downloader Support (Optional)

```python
# ppera/datasets_downloader.py
DATASET_CONFIG = {
    # ... existing datasets ...
    "customdataset": {
        "kaggle_dataset": "username/dataset-name",
        "local_dir": "datasets/CustomDataset",
        "expected_files": ["interactions.csv", "metadata.csv"],
        "extract_patterns": ["*.csv"]
    }
}
```

## Performance Considerations

### Memory Management

```python
# Sequential row limiting (memory efficient)
data = loader("movielens", num_rows=1000)  # Loads only first 1000 rows

# For large datasets, use smaller chunks
for chunk_size in [1000, 5000, 10000]:
    data_chunk = loader("amazonsales", num_rows=chunk_size, seed=42)
    # Process chunk
```

### Caching Optimization

```python
# Pre-generate common dataset sizes
common_sizes = [1000, 5000, 10000, 14000]
for size in common_sizes:
    for seed in [42, 123, 456]:
        data = loader("movielens", num_rows=size, seed=seed)
        # This creates cached files for faster future access
```

## Troubleshooting

### Common Issues

#### 1. Missing Dataset Files
```bash
Error: Required dataset files not found in datasets/MovieLens
Solution: Run 'make check-datasets' to download automatically
```

#### 2. Kaggle API Not Configured
```bash
Error: 401 - Unauthorized
Solution: Configure Kaggle API credentials (see setup instructions above)
```

#### 3. Memory Issues with Large Datasets
```python
# Use row limiting
data = loader("amazonsales", num_rows=5000)  # Instead of loading full dataset
```

#### 4. Column Name Mismatches
```python
# Check available columns
data = loader("movielens")
print(data.columns.tolist())
# Expected: ['userID', 'itemID', 'rating', 'timestamp', 'title', 'genres']
```

#### 5. Knowledge Graph Processing Errors
```bash
# Clear RL cache if issues occur
rm -rf ppera/rl_tmp/Movielens/
# Re-run RL experiment to regenerate
```

### Validation Commands

```bash
# Test dataset loading
python -c "from ppera.datasets_loader import loader; print(loader('movielens', num_rows=10).shape)"

# Test all datasets
for dataset in movielens amazonsales postrecommendations; do
    python -c "from ppera.datasets_loader import loader; print('$dataset:', loader('$dataset', num_rows=100).shape)"
done

# Check processed files
ls -la ppera/datasets/*/
ls -la ppera/rl_tmp/*/
```

## Data Format Requirements

For any custom dataset to work with PPERA, ensure:

1. **Minimum required columns**: `userID`, `itemID`, `rating`
2. **Optional but recommended**: `timestamp`, `title`, `genres`
3. **Data types**: 
   - `userID`: string/int (will be converted to string)
   - `itemID`: string/int (will be converted to string)
   - `rating`: numeric (float/int)
   - `timestamp`: Unix timestamp (int) or datetime string
   - `genres`: space-separated string (e.g., "Action Adventure Sci-Fi")

4. **No missing values** in required columns
5. **Consistent ID formats** throughout the dataset

---

*For more details on specific data processing functions, see the [API Reference](api.md#datasets-module).*