from dataclasses import asdict
import logging

import click

from llm_rec_eval.config import Config
from llm_rec_eval.pipeline import load_pipeline
from llm_rec_eval.runner import ExperimentRunner
from llm_rec_eval.constants import FREQUENCY_CATEGORIES

logger = logging.getLogger(__name__)


@click.command(context_settings={"default_map": asdict(Config())})
@click.option("--testing-ratio", type=float)
@click.option("--batch-size", type=int)
@click.option("--initial-run-seed", type=int)
@click.option("--model", type=str)
@click.option("--likes-count", type=int)
@click.option("--dislikes-count", type=int)
@click.option("--with-context/--without-context")
@click.option("--likes-first/--dislikes-first")
@click.option("--task-desc-version", type=int)
@click.option("--shots", type=int)
@click.option("--with-genre/--without-genre")
@click.option("--with-global-rating-in-context/--without-global-rating-in-context")
@click.option("--with-global-rating-in-task/--without-global-rating-in-task")
@click.option("--temperature", type=float)
@click.option("--popularity", multiple=True, type=click.Choice(FREQUENCY_CATEGORIES))
@click.option(
    "--training-popularity", multiple=True, type=click.Choice(FREQUENCY_CATEGORIES)
)
@click.option("--runs", type=int)
@click.option("--keep-trailing-zeroes/--strip-trailing-zeroes")
@click.option("--double-range/--single-range")
@click.option("--sample-header-version", type=int)
@click.option("--rating-listing-version", type=int)
@click.option("--context-header-version", type=int)
@click.option("--answer-mark-version", type=int)
@click.option("--numeric-user-identifier/--alphabetic-user-identifier")
@click.option("--precision", type=click.Choice(["default", "16", "8", "4"]))
@click.option("--use-flash-attention-2", is_flag=True)
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
