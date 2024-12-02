import logging

import click
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model", type=str, help="Hugging Face model repository ID")
def main(model):
    """Download a Hugging Face model repository.

    This function downloads the models config and parameters without loading it into memory.
    """
    try:
        snapshot_download(
            repo_id=model, allow_patterns=["*.json", "*.model", "*.safetensors"]
        )
        logger.info(f"Model '{model}' downloaded successfully.")
    except Exception:
        logger.exception(f"Error downloading model '{model}'")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT
    )
    main()
