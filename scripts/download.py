import logging
import os
import tarfile

import requests
import tyro
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("download_script")


def download_and_extract(url: str, download_dir: str):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    file_name = os.path.join(download_dir, url.split("/")[-1])

    logger.info(f"Downloading from {url} to {file_name}...")
    response = requests.get(url)

    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.content)
        logger.info("Download completed.")

        if file_name.endswith(".tar.gz"):
            logger.info(f"Extracting {file_name}...")
            with tarfile.open(file_name, "r:gz") as tar:
                tar.extractall(path=download_dir)
            logger.info("Extraction completed.")
    else:
        logger.error(f"Failed to download file: {response.status_code}")


def main(url: str, download_dir: str):
    download_and_extract(url, download_dir)


if __name__ == "__main__":
    tyro.cli(main)
