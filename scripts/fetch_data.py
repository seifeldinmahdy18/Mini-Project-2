import os
import subprocess
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME', '')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY', '')

DATASET_NAME = "forgemaster/steam-reviews-dataset"
DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "data"

def fetch_dataset():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run([
            "kaggle", "datasets", "download", "-d", DATASET_NAME,
            "-p", str(DOWNLOAD_DIR)
        ], check=True)
    except FileNotFoundError:
        return
    except subprocess.CalledProcessError:
        return

    zip_path = DOWNLOAD_DIR / "steam-reviews-dataset.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)
        zip_path.unlink()

if __name__ == "__main__":
    fetch_dataset()
