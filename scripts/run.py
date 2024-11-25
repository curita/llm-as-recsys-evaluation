import logging
import re

import click
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers.pipelines.base import Pipeline

from llm_rec_eval.constants import FREQUENCY_CATEGORIES, POSSIBLE_VALUES
from llm_rec_eval.dataset import MovieLensDataSet, MockListDataset
from llm_rec_eval.metrics import AggregatedStats, report_metrics
from llm_rec_eval.pipeline import get_inference_kwargs, load_pipeline
from llm_rec_eval.prompts import PromptGenerator
from llm_rec_eval.save import save_results

logger = logging.getLogger(__name__)


def parse_model_output(output: str, double_range: bool) -> float:
    original_output = output

    try:
        output = re.sub(
            r"^[^\d\w]+", "", output
        )  # Strip leading puntuation, spaces or emojis
        value = float(re.findall(r"^(\d+(?:\.\d+)?)", output)[0])

        min_value, max_value = POSSIBLE_VALUES[0], POSSIBLE_VALUES[-1]
        if double_range:
            min_value *= 2
            max_value *= 2

        assert value >= min_value
        assert value <= max_value

        if double_range:
            value /= 2

        return value

    except Exception as err:
        msg = f"Can't parse: {original_output!r}"
        logger.exception(msg)
        raise ValueError(msg) from err


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
    stats = AggregatedStats()
    predictor = load_pipeline(
        stats=stats,
        precision=precision,
        use_flash_attention_2=use_flash_attention_2,
        model=model,
    )
    run_experiment(predictor=predictor, stats=stats, **params)


class ExperimentRunner:
    def __init__(self, predictor, stats, **params):
        self.predictor = predictor
        self.stats = stats
        self.params = params

    def run(self):
        for x in range(self.params["runs"]):
            self.run_single_experiment(x)
        self.stats.report()

    def run_single_experiment(self, run_index):
        run_params = self.params.copy()
        run_seed = self.params["initial_run_seed"] + run_index
        run_params["run_seed"] = run_seed

        logger.info(f"Run {run_seed=}.")
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        dataset = self.create_dataset(run_params)
        prompts = self.generate_prompts(dataset, run_params)
        outputs = self.run_model(prompts)
        predictions, unpredicted_indexes = self.parse_outputs(outputs, prompts)
        truth = [row.rating for row in dataset.testing_df.itertuples()]

        save_results(prompts, outputs, predictions, dataset, run_params)
        self.remove_unpredicted_items(truth, predictions, unpredicted_indexes)
        self.report_metrics(truth, predictions)

    def create_dataset(self, run_params):
        logger.info("Creating dataset...")
        return MovieLensDataSet(
            testing_ratio=self.params["testing_ratio"],
            training_popularity=self.params["training_popularity"],
            popularity=self.params["popularity"],
        )

    def generate_prompts(self, dataset, run_params):
        logger.info("Generating prompts...")
        prompt_generator = PromptGenerator(dataset=dataset, **run_params)
        prompts = [
            prompt_generator(user_id=row.userId, movie_id=row.movieId)
            for row in dataset.testing_df.itertuples()
        ]
        logger.info(f"Prompt Example:\n{prompts[0]}")
        self.stats.increment_prompts_count(len(prompts))
        return prompts

    def run_model(self, prompts):
        logger.info("Running model...")
        inference_kwargs = get_inference_kwargs(
            model=self.params["model"], temperature=self.params["temperature"]
        )
        return [
            p[0]["generated_text"]
            for p in tqdm(
                self.predictor(
                    MockListDataset(prompts),
                    batch_size=self.params["batch_size"],
                    **inference_kwargs,
                ),
                total=len(prompts),
            )
        ]

    def parse_outputs(self, outputs, prompts):
        logger.info("Parsing outputs...")
        retried_prompts = 0
        retries = 0
        max_retries = 3
        predictions = []
        unpredicted_indexes = set()
        for index, out in enumerate(outputs):
            try:
                pred = parse_model_output(out, double_range=self.params["double_range"])
            except ValueError:
                try:
                    retried_prompts += 1
                    (
                        retried_output,
                        retried_prediction,
                        retried_attempts,
                    ) = self.retry_inference(
                        prompt=prompts[index],
                        max_retries=max_retries,
                    )
                    retries += retried_attempts
                    outputs[index] = retried_output
                    pred = retried_prediction
                except ValueError:
                    retries += max_retries
                    unpredicted_indexes.add(index)
                    pred = "N/A"
            predictions.append(pred)

        logger.info(f"Retried prompts: {retried_prompts}")
        logger.info(f"Retries: {retries}")
        self.stats.increment_retried_prompts(retried_prompts)
        self.stats.increment_retries(retries)
        return predictions, unpredicted_indexes

    def retry_inference(self, prompt, max_retries=3):
        inference_kwargs = get_inference_kwargs(
            model=self.params["model"], temperature=self.params["temperature"]
        )
        inference_kwargs["do_sample"] = True

        for attempt in range(1, max_retries + 1):
            logger.info(f"Retrying, {attempt=}")
            output = self.predictor(prompt, **inference_kwargs)[0]["generated_text"]
            try:
                pred = parse_model_output(
                    output, double_range=self.params["double_range"]
                )
            except ValueError:
                continue
            else:
                return output, pred, attempt

        raise ValueError("Couldn't get prediction")

    def remove_unpredicted_items(self, truth, predictions, unpredicted_indexes):
        logger.info("Removing unpredicted items...")
        truth[:] = [
            value for i, value in enumerate(truth) if i not in unpredicted_indexes
        ]
        predictions[:] = [
            value for i, value in enumerate(predictions) if i not in unpredicted_indexes
        ]
        logger.info(f"Unknown predictions: {len(unpredicted_indexes)}")
        self.stats.increment_unpredicted(len(unpredicted_indexes))

    def report_metrics(self, truth, predictions):
        if not predictions:
            logger.info("All predictions are unknown, stopping experiment...")
            return

        stats = report_metrics(truth, predictions)
        self.stats.add_rmse(stats.rmse)
        self.stats.add_precision(stats.precision)
        self.stats.add_recall(stats.recall)
        self.stats.add_f1(stats.f1)
        self.stats.update_value_count(stats.histogram)


def run_experiment(predictor: Pipeline, stats: AggregatedStats, **params):
    runner = ExperimentRunner(predictor, stats, **params)
    runner.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT
    )
    main()
