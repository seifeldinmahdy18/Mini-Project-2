import os
import subprocess
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME', '')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY', '')

DATASET_NAME = "forgemaster/steam-reviews-dataset"
TARGET_FILE  = "reviews-1-115.csv"
DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_dataset():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", DATASET_NAME,
        "-p", str(DOWNLOAD_DIR),
        "--file", TARGET_FILE,
    ], check=True)

    zip_path = DOWNLOAD_DIR / f"{TARGET_FILE}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DOWNLOAD_DIR)
        zip_path.unlink()


if __name__ == "__main__":
    fetch_dataset()
