from dataclasses import asdict
import logging

import click

from llm_rec_eval.config import Config
from llm_rec_eval.pipeline import load_pipeline
from llm_rec_eval.runner import ExperimentRunner
from llm_rec_eval.constants import FREQUENCY_CATEGORIES

logger = logging.getLogger(__name__)


@click.command(context_settings={"default_map": asdict(Config())})
@click.option(
    "--runs", type=int, help="Number of experiment iterations for cross-validation."
)
@click.option("--initial-run-seed", type=int, help="Initial random seed.")
@click.option(
    "--testing-ratio",
    type=float,
    help="Proportion of dataset reserved for testing (between 0 and 1).",
)
@click.option(
    "--popularity",
    multiple=True,
    type=click.Choice(FREQUENCY_CATEGORIES),
    help="Filter dataset by selected movie popularity categories.",
)
@click.option(
    "--training-popularity",
    multiple=True,
    type=click.Choice(FREQUENCY_CATEGORIES),
    help="Filter training dataset by selected movie popularity categories.",
)
@click.option(
    "--model",
    type=str,
    help="HuggingFace identifier or file path of the model to be evaluated.",
)
@click.option(
    "--precision",
    type=click.Choice(["default", "16", "8", "4"]),
    help="Numerical precision to load the model weights.",
)
@click.option("--use-flash-attention-2", is_flag=True, help="Enable Flash Attention 2.")
@click.option(
    "--batch-size",
    type=int,
    help="Inference batch size determining how many prompts are processed at the same time.",
)
@click.option(
    "--temperature",
    type=float,
    help="Sampling temperature controlling randomness of model outputs.",
)
@click.option("--shots", type=int, help="Number of shots for in-context learning.")
@click.option(
    "--likes-count",
    type=int,
    help="Number of top-rated movies to use in user history samples.",
)
@click.option(
    "--dislikes-count",
    type=int,
    help="Number of bottom-rated movies to use in user history samples.",
)
@click.option(
    "--double-range/--single-range",
    help="Use rating scale of 1-10 (double) or 0.5-5 (single).",
)
@click.option(
    "--with-context/--without-context",
    help="Include or exclude user historical context in prompts.",
)
@click.option(
    "--with-genre/--without-genre", help="Include or exclude movie genres in prompts."
)
@click.option(
    "--with-global-rating-in-context/--without-global-rating-in-context",
    help="Include or exclude global ratings for movies in user history.",
)
@click.option(
    "--with-global-rating-in-task/--without-global-rating-in-task",
    help="Include or exclude global rating for target movie.",
)
@click.option(
    "--context-header-version",
    type=int,
    help="Formatting version of user historical context header.",
)
@click.option(
    "--sample-header-version", type=int, help="Formatting version for sample header."
)
@click.option(
    "--rating-listing-version", type=int, help="Formatting version for rating listing."
)
@click.option(
    "--task-desc-version", type=int, help="Formatting version for task description."
)
@click.option(
    "--answer-mark-version", type=int, help="Formatting version for answer mark."
)
@click.option(
    "--numeric-user-identifier/--alphabetic-user-identifier",
    help="Use numeric or alphabetic user identifiers.",
)
@click.option(
    "--likes-first/--dislikes-first",
    help="Determine order of movies in user history: top-rated first or bottom-rated first.",
)
@click.option(
    "--keep-trailing-zeroes/--strip-trailing-zeroes",
    help="Preserve or remove trailing zeroes in ratings.",
)
def main(**kwargs):
    logger.info(
        f"Script parameters {' '.join(str(k) + '=' + str(v) for k, v in kwargs.items())}."
    )
    config = Config(**kwargs)
    predictor = load_pipeline(
        precision=config.precision,
        use_flash_attention_2=config.use_flash_attention_2,
        model=config.model,
    )
    runner = ExperimentRunner(predictor, config=config)
    runner.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT
    )
    main()
