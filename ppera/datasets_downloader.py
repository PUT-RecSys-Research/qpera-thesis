import os
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import shutil


class KaggleDatasetDownloader:
    """Handles automatic dataset downloads from Kaggle."""
    
    DATASET_CONFIG = {
        "amazonsales": {
            "kaggle_dataset": "karkavelrajaj/amazon-sales-dataset",
            "local_dir": "datasets/AmazonSales",
            "expected_files": ["amazon.csv"],
            "extract_patterns": ["*.csv"]
        },
        "movielens": {
            "kaggle_dataset": "grouplens/movielens-20m-dataset", 
            "local_dir": "datasets/MovieLens",
            "expected_files": ["rating.csv", "movie.csv", "tag.csv"],
            "extract_patterns": ["*.csv"]
        },
        "postrecommendations": {
            "kaggle_dataset": "vatsalparsaniya/post-pecommendation",
            "local_dir": "datasets/PostRecommendations", 
            "expected_files": ["post_data.csv", "user_data.csv", "view_data.csv"],
            "extract_patterns": ["*.csv"]
        }
    }
    
    def __init__(self):
        """Initialize the downloader and check Kaggle CLI availability."""
        self._check_kaggle_cli()
        self._check_kaggle_auth()
    
    def _check_kaggle_cli(self) -> None:
        """Check if Kaggle CLI is installed."""
        try:
            result = subprocess.run(
                ["kaggle", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print(f"Kaggle CLI found: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Kaggle CLI not found. Please install it:\n"
                "pip install kaggle\n"
                "Or activate the conda environment: conda activate ppera-env"
            )
    
    def _check_kaggle_auth(self) -> None:
        """Check if Kaggle authentication is configured."""
        kaggle_config_path = Path.home() / ".kaggle" / "kaggle.json"
        
        if not kaggle_config_path.exists():
            raise RuntimeError(
                f"Kaggle authentication not found at {kaggle_config_path}\n"
                "Please follow these steps:\n"
                "1. Go to https://www.kaggle.com/account\n"
                "2. Click 'Create New API Token'\n"
                "3. Download kaggle.json\n"
                "4. Place it at ~/.kaggle/kaggle.json\n"
                "5. Run: chmod 600 ~/.kaggle/kaggle.json"
            )
        
        # Test authentication
        try:
            subprocess.run(
                ["kaggle", "datasets", "list", "--max-size", "1"],
                capture_output=True,
                check=True,
                timeout=30
            )
            print("Kaggle authentication verified")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Kaggle authentication failed: {e}\n"
                "Please check your kaggle.json credentials"
            )
        except subprocess.TimeoutExpired:
            print("Kaggle API timeout (but credentials seem valid)")
    
    def _create_dataset_dir(self, dataset_name: str) -> Path:
        """Create dataset directory if it doesn't exist."""
        config = self.DATASET_CONFIG[dataset_name]
        dataset_dir = Path(config["local_dir"])
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir
    
    def _is_dataset_downloaded(self, dataset_name: str) -> bool:
        """Check if dataset is already downloaded and valid."""
        config = self.DATASET_CONFIG[dataset_name]
        dataset_dir = Path(config["local_dir"])
        
        if not dataset_dir.exists():
            return False
        
        # Check if all expected files exist
        for expected_file in config["expected_files"]:
            file_path = dataset_dir / expected_file
            if not file_path.exists() or file_path.stat().st_size == 0:
                print(f"Missing or empty file: {file_path}")
                return False
        
        print(f"Dataset '{dataset_name}' already downloaded and valid")
        return True
    
    def _download_from_kaggle(self, dataset_name: str, force: bool = False) -> None:
        """Download dataset from Kaggle."""
        config = self.DATASET_CONFIG[dataset_name]
        dataset_dir = self._create_dataset_dir(dataset_name)
        
        print(f"Downloading {config['kaggle_dataset']} to {dataset_dir}")
        
        try:
            # Download dataset
            cmd = [
                "kaggle", "datasets", "download",
                config["kaggle_dataset"],
                "--path", str(dataset_dir),
                "--unzip"
            ]
            
            if force:
                cmd.append("--force")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10 minute timeout
            )
            
            print(f"Download completed: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to download {dataset_name}:\n"
                f"Command: {' '.join(e.cmd)}\n"
                f"Error: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Download timeout for {dataset_name} (>10 minutes)\n"
                "Please check your internet connection"
            )
    
    def _post_process_dataset(self, dataset_name: str) -> None:
        """Post-process downloaded dataset (rename files, cleanup, etc.)."""
        config = self.DATASET_CONFIG[dataset_name]
        dataset_dir = Path(config["local_dir"])
        
        # Dataset-specific post-processing
        if dataset_name == "movielens":
            self._process_movielens(dataset_dir)
        elif dataset_name == "amazonsales":
            self._process_amazonsales(dataset_dir)
        elif dataset_name == "postrecommendations":
            self._process_postrecommendations(dataset_dir)
    
    def _process_movielens(self, dataset_dir: Path) -> None:
        """Process MovieLens dataset files."""
        # The files are already correctly named, just verify they exist
        expected_files = ["rating.csv", "movie.csv", "tag.csv"]
        
        for file_name in expected_files:
            file_path = dataset_dir / file_name
            if file_path.exists():
                print(f"Found {file_name}")
            else:
                print(f"Missing {file_name}")
        
        # Remove unnecessary files that might come with the download
        unnecessary_files = [
            "genome_scores.csv", "genome_tags.csv", 
            "README.txt", "ml-20m", "link.csv"
        ]
        
        for file_name in unnecessary_files:
            file_path = dataset_dir / file_name
            if file_path.exists():
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
                print(f"Removed {file_name}")
    
    def _process_amazonsales(self, dataset_dir: Path) -> None:
        """Process Amazon Sales dataset files."""
        # The main file should be amazon.csv
        # Verify amazon.csv exists
        amazon_file = dataset_dir / "amazon.csv"
        if amazon_file.exists():
            print("Found amazon.csv")
        else:
            print("Missing amazon.csv")
        
        # Remove any unnecessary files (but keep .gitkeep)
        all_files = list(dataset_dir.glob("*"))
        for file_path in all_files:
            if (file_path.name != "amazon.csv" and 
                file_path.name != ".gitkeep" and 
                file_path.is_file()):
                file_path.unlink()
                print(f"Removed unnecessary file: {file_path.name}")
    
    def _process_postrecommendations(self, dataset_dir: Path) -> None:
        """Process Post Recommendations dataset files."""
        # Verify the expected files exist
        expected_files = ["post_data.csv", "user_data.csv", "view_data.csv"]
        
        for file_name in expected_files:
            file_path = dataset_dir / file_name
            if file_path.exists():
                print(f"Found {file_name}")
            else:
                print(f"Missing {file_name}")
        
        # Remove any unnecessary files
        all_files = list(dataset_dir.glob("*.csv"))
        for file_path in all_files:
            if file_path.name not in expected_files:
                file_path.unlink()
                print(f"Removed unexpected file: {file_path.name}")
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Download a specific dataset.
        
        Args:
            dataset_name: Name of dataset ('amazonsales', 'movielens', 'postrecommendations')
            force: Force re-download even if dataset exists
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.DATASET_CONFIG:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(self.DATASET_CONFIG.keys())}"
            )
        
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Check if already downloaded
        if not force and self._is_dataset_downloaded(dataset_name):
            return True
        
        try:
            # Download from Kaggle
            self._download_from_kaggle(dataset_name, force)
            
            # Post-process
            self._post_process_dataset(dataset_name)
            
            # Verify download
            if self._is_dataset_downloaded(dataset_name):
                print(f"Successfully downloaded and verified: {dataset_name}")
                return True
            else:
                print(f"Download verification failed: {dataset_name}")
                return False
                
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return False
    
    def download_all_datasets(self, force: bool = False) -> Dict[str, bool]:
        """
        Download all datasets.
        
        Returns:
            Dictionary with dataset names and success status
        """
        print("Starting bulk dataset download...")
        results = {}
        
        for dataset_name in self.DATASET_CONFIG.keys():
            results[dataset_name] = self.download_dataset(dataset_name, force)
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        
        print(f"\nDownload Summary: {successful}/{total} successful")
        for dataset_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {status}: {dataset_name}")
        
        return results
    
    def get_setup_instructions(self) -> str:
        """Get setup instructions for Kaggle authentication."""
        return """
Kaggle Setup Instructions:

1. Create Kaggle Account: https://www.kaggle.com/account/login

2. Generate API Token:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Download kaggle.json file

3. Setup Authentication:
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

4. Test Setup:
   kaggle datasets list --max-size 1
"""


def main():
    """CLI interface for dataset downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets from Kaggle")
    parser.add_argument(
        "--setup-help",
        action="store_true", 
        help="Show Kaggle setup instructions"
    )
    
    args = parser.parse_args()
    
    if args.setup_help:
        downloader = KaggleDatasetDownloader()
        print(downloader.get_setup_instructions())
        return
    
    try:
        downloader = KaggleDatasetDownloader()
        
        results = downloader.download_all_datasets()
        success = all(results.values())
        
        if success:
            print("\nAll downloads completed successfully!")
            exit(0)
        else:
            print("\nSome downloads failed. Check the logs above.")
            exit(1)
            
    except Exception as e:
        print(f"\nFatal error: {e}")
        print("\nFor setup help, run: python -m ppera.dataset_downloader --setup-help")
        exit(1)


if __name__ == "__main__":
    main()