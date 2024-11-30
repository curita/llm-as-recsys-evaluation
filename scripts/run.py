import logging

import click

from llm_rec_eval.pipeline import load_pipeline
from llm_rec_eval.runner import ExperimentRunner
from llm_rec_eval.constants import FREQUENCY_CATEGORIES

logger = logging.getLogger(__name__)


@click.command()
@click.option("--testing-ratio", default=0.2, type=float)
@click.option("--batch-size", default=8, type=int)
@click.option("--initial-run-seed", default=0, type=int)
@click.option("--model", default="google/flan-t5-base", type=str)
@click.option("--likes-count", default=10, type=int)
@click.option("--dislikes-count", default=10, type=int)
@click.option("--with-context/--without-context", default=True)
@click.option("--likes-first/--dislikes-first", default=True)
@click.option("--task-desc-version", default=1, type=int)
@click.option("--shots", default=0, type=int)
@click.option("--with-genre/--without-genre", default=False)
@click.option(
    "--with-global-rating-in-context/--without-global-rating-in-context", default=False
)
@click.option(
    "--with-global-rating-in-task/--without-global-rating-in-task", default=False
)
@click.option("--temperature", default=0, type=float)
@click.option("--popularity", multiple=True, type=click.Choice(FREQUENCY_CATEGORIES))
@click.option(
    "--training-popularity", multiple=True, type=click.Choice(FREQUENCY_CATEGORIES)
)
@click.option("--runs", default=1, type=int)
@click.option("--keep-trailing-zeroes/--strip-trailing-zeroes", default=True)
@click.option("--double-range/--single-range", default=False)
@click.option("--sample-header-version", default=1, type=int)
@click.option("--rating-listing-version", default=1, type=int)
@click.option("--context-header-version", default=1, type=int)
@click.option("--answer-mark-version", default=1, type=int)
@click.option("--numeric-user-identifier/--alphabetic-user-identifier", default=False)
@click.option(
    "--precision", default="default", type=click.Choice(["default", "16", "8", "4"])
)
@click.option("--use-flash-attention-2", is_flag=True, default=False)
def main(
    testing_ratio,
    batch_size,
    initial_run_seed,
    model,
    likes_count,
    dislikes_count,
    with_context,
    likes_first,
    task_desc_version,
    shots,
    with_genre,
    with_global_rating_in_context,
    with_global_rating_in_task,
    temperature,
    popularity,
    training_popularity,
    runs,
    keep_trailing_zeroes,
    double_range,
    sample_header_version,
    rating_listing_version,
    context_header_version,
    answer_mark_version,
    numeric_user_identifier,
    precision,
    use_flash_attention_2,
):
    params = locals()
    logger.info(
        f"Script parameters {' '.join(str(k) + '=' + str(v) for k, v in params.items())}."
    )
    predictor = load_pipeline(
        precision=precision,
        use_flash_attention_2=use_flash_attention_2,
        model=model,
    )
    runner = ExperimentRunner(predictor, **params)
    runner.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT
    )
    main()
