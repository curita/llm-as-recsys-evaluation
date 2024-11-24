import csv
import json
import logging
import re
from pathlib import Path

import click
import numpy as np
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.base import Pipeline

from llm_rec_eval.constants import FREQUENCY_CATEGORIES, POSSIBLE_VALUES
from llm_rec_eval.dataset import MovieLensDataSet
from llm_rec_eval.metrics import AggregatedStats, report_metrics
from llm_rec_eval.prompts import PromptGenerator

logger = logging.getLogger(__name__)


class MockListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


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


FILENAME_PARAMETERS = {
    "RATIO": "testing_ratio",
    "SEED": "run_seed",
    "M": "model",
    "L": "likes_count",
    "D": "dislikes_count",
    "C": "with_context",
    "F": "likes_first",
    "V": "task_desc_version",
    "S": "shots",
    "G": "with_genre",
    "CR": "with_global_rating_in_context",
    "TR": "with_global_rating_in_task",
    "T": "temperature",
    "P": "popularity",
    "TP": "training_popularity",
    "Z": "keep_trailing_zeroes",
    "DO": "double_range",
    "SH": "sample_header_version",
    "RL": "rating_listing_version",
    "H": "context_header_version",
    "AM": "answer_mark_version",
    "N": "numeric_user_identifier",
    "TK": "task",
    "B": "batch_size",
    "PR": "precision",
    "FL": "use_flash_attention_2",
}


@retry(stop=stop_after_attempt(5), wait=wait_exponential())
def get_pipeline(task: str, model: str, model_parameters: dict) -> Pipeline:
    return pipeline(
        task, model=model, device_map="auto", token=True, **model_parameters
    )


@click.command()
@click.option("--testing-ratio", default=0.2, type=float)
@click.option("--batch-size", default=8, type=int)
@click.option("--initial-run-seed", default=0, type=int)
@click.option("--model", default="google/flan-t5-base", type=str)
@click.option(
    "--task",
    default="text2text-generation",
    type=click.Choice(["text2text-generation", "text-generation"]),
)
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
    task,
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
        task=task,
        precision=precision,
        use_flash_attention_2=use_flash_attention_2,
        model=model,
    )
    run_experiment(predictor=predictor, stats=stats, **params)


def get_default_task(model: str) -> str:
    if "t5" in model.lower():
        return "text2text-generation"
    return "text-generation"


def load_pipeline(
    stats: AggregatedStats,
    task: str = "text2text-generation",
    precision: str = "default",
    use_flash_attention_2: bool = False,
    model: str = "google/flan-t5-base",
) -> Pipeline:
    logger.info(f"Initializing {task} pipeline...")

    model_parameters = {}
    if precision == "16":
        model_parameters["torch_dtype"] = torch.float16
    elif precision == "8":
        model_parameters["model_kwargs"] = {"load_in_8bit": True}
    elif precision == "4":
        model_parameters["model_kwargs"] = {"load_in_4bit": True}

    if use_flash_attention_2:
        model_parameters["torch_dtype"] = torch.float16
        model_parameters.setdefault("model_kwargs", {})["attn_implementation"] = (
            "flash_attention_2"
        )

    predictor = get_pipeline(task=task, model=model, model_parameters=model_parameters)
    preprocess_method_name = (
        "_parse_and_tokenize"
        if hasattr(predictor, "_parse_and_tokenize")
        else "preprocess"
    )
    original_preprocess = getattr(predictor, preprocess_method_name)
    # This is the max sequence length. Does it mean the model doesn't output responses longer than this, or it doesn't work well in contexts longer than this?
    max_token_length = getattr(predictor.model.config, "max_position_embeddings", None)
    if not max_token_length and "t5" in model.lower():
        # https://huggingface.co/google/flan-t5-xxl/discussions/41#65c3c3706b793334ef78dffc
        max_token_length = 1024

    logger.info(f"Model context limit: {max_token_length}")

    def _patched_preprocess(*args, **kwargs):
        inputs = original_preprocess(*args, **kwargs)
        # NOTE: Only valid for PyTorch tensors
        input_length = inputs["input_ids"].shape[-1]

        if input_length > max_token_length:
            stats.increment_over_token_limit()

        return inputs

    setattr(predictor, preprocess_method_name, _patched_preprocess)
    return predictor


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

        self.save_results(prompts, outputs, predictions, dataset, run_params)
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
        model_parameters = self.get_model_parameters()
        return [
            p[0]["generated_text"]
            for p in tqdm(
                self.predictor(
                    MockListDataset(prompts),
                    batch_size=self.params["batch_size"],
                    **model_parameters,
                ),
                total=len(prompts),
            )
        ]

    def get_model_parameters(self):
        model_parameters = {}
        if self.params["temperature"] == 0.0:
            model_parameters["do_sample"] = False
        else:
            model_parameters["do_sample"] = True
            model_parameters["temperature"] = self.params["temperature"]

        if self.params["task"] == "text-generation":
            model_parameters["return_full_text"] = False
            model_parameters["max_new_tokens"] = 20
            if (
                self.predictor.tokenizer.pad_token_id
                and not self.predictor.model.config.pad_token_id
            ):
                self.predictor.model.config.pad_token_id = (
                    self.predictor.tokenizer.pad_token_id
                )
            elif (
                not self.predictor.tokenizer.pad_token_id
                and self.predictor.model.config.pad_token_id
            ):
                self.predictor.tokenizer.pad_token_id = (
                    self.predictor.model.config.pad_token_id
                )
            else:
                if "llama-2" in self.params["model"].lower():
                    self.predictor.tokenizer.pad_token = "[PAD]"
                    self.predictor.tokenizer.padding_side = "left"
                else:
                    self.predictor.tokenizer.pad_token_id = (
                        self.predictor.model.config.eos_token_id
                    )
                    self.predictor.model.config.pad_token_id = (
                        self.predictor.model.config.eos_token_id
                    )
        return model_parameters

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
        model_parameters = self.get_model_parameters()
        model_parameters["do_sample"] = True

        for attempt in range(1, max_retries + 1):
            logger.info(f"Retrying, {attempt=}")
            output = self.predictor(prompt, **model_parameters)[0]["generated_text"]
            try:
                pred = parse_model_output(
                    output, double_range=self.params["double_range"]
                )
            except ValueError:
                continue
            else:
                return output, pred, attempt

        raise ValueError("Couldn't get prediction")

    def save_results(self, prompts, outputs, predictions, dataset, run_params):
        logger.info("Dumping results...")
        folder_name = f"experiment_{'_'.join(k + '=' + str(run_params[v]) for k, v in FILENAME_PARAMETERS.items())}".replace(
            "/", ":"
        )
        output_folder = Path("results") / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / "results.csv"

        logger.info(f"Path: {output_file}")

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "Prompt",
                    "Movie",
                    "MovieID",
                    "UserID",
                    "Output",
                    "Prediction",
                    "Truth",
                    "Parameters",
                ],
            )

            writer.writeheader()
            parameters = json.dumps(run_params)
            for prmpt, out, pred, row in zip(
                prompts, outputs, predictions, dataset.testing_df.itertuples()
            ):
                writer.writerow(
                    {
                        "Prompt": prmpt,
                        "Movie": dataset.get_movie_name(row.movieId),
                        "MovieID": row.movieId,
                        "UserID": row.userId,
                        "Output": out,
                        "Prediction": str(pred),
                        "Truth": str(row.rating),
                        "Parameters": parameters,
                    }
                )
                parameters = ""

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
