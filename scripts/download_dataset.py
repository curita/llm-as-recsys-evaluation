import requests
import zipfile
import logging
from pathlib import Path

# Define the URL for the MovieLens small dataset
DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Define the output directory and file
OUTPUT_DIR = Path("./data")
OUTPUT_FILE = OUTPUT_DIR / "ml-latest-small.zip"

logger = logging.getLogger(__name__)


def download_dataset(url: str, output_file: Path) -> None:
    """
    Download the dataset from the given URL to the specified output file.

    :param url: The URL of the dataset.
    :param output_file: The path to save the downloaded file.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Download complete.")


def unzip_dataset(zip_file: Path, output_dir: Path) -> None:
    """
    Unzip the dataset to the specified output directory.

    :param zip_file: The path to the zip file.
    :param output_dir: The directory to extract the files to.
    """
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    logger.info("Unzipping complete.")


def main() -> None:
    """
    Main function to download and extract the MovieLens small dataset.
    """
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    logger.info("Downloading MovieLens small dataset...")
    download_dataset(DATASET_URL, OUTPUT_FILE)

    # Unzip the dataset
    logger.info("Unzipping the dataset...")
    unzip_dataset(OUTPUT_FILE, OUTPUT_DIR)

    # Remove the zip file
    logger.info("Cleaning up...")
    OUTPUT_FILE.unlink()

    logger.info("Download and extraction complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT
    )
    main()
