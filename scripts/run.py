from collections import defaultdict
from enum import Enum
import json
from pathlib import Path
import re
import csv
import logging

import click
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.base import Pipeline
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    mean_squared_error,
    classification_report,
    precision_recall_fscore_support,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from src.constants import FREQUENCY_CATEGORIES, POSSIBLE_VALUES
from src.dataset import MovieLensDataSet


logger = logging.getLogger(__name__)


class SampleKind(Enum):
    lowest = "lowest"
    highest = "highest"


class PromptGenerator:
    def __init__(
        self,
        dataset: MovieLensDataSet,
        with_genre: bool,
        with_global_rating_in_context: bool,
        with_global_rating_in_task: bool,
        likes_first: bool,
        likes_count: int,
        dislikes_count: int,
        task_desc_version: int,
        with_context: bool,
        shots: int,
        keep_trailing_zeroes: bool,
        double_range: bool,
        sample_header_version: int,
        rating_listing_version: int,
        context_header_version: int,
        answer_mark_version: int,
        numeric_user_identifier: bool,
        task: str,
        **kwargs,
    ) -> None:
        self.dataset = dataset
        self.with_genre = with_genre
        self.with_global_rating_in_context = with_global_rating_in_context
        self.with_global_rating_in_task = with_global_rating_in_task
        self.likes_first = likes_first
        self.likes_count = likes_count
        self.dislikes_count = dislikes_count
        self.task_desc_version = task_desc_version
        self.with_context = with_context
        self.shots = shots
        self.keep_trailing_zeroes = keep_trailing_zeroes
        self.double_range = double_range
        self.sample_header_version = sample_header_version
        self.rating_listing_version = rating_listing_version
        self.context_header_version = context_header_version
        self.answer_mark_version = answer_mark_version
        self.numeric_user_identifier = numeric_user_identifier
        self.task = task

    def get_movie_info(
        self, movie_id: int, with_genre: bool, with_global_rating: bool
    ) -> str:
        info = f'"{self.dataset.get_movie_name(movie_id)}"'
        if with_genre:
            info += f' ({"|".join(self.dataset.get_movie_genres(movie_id))})'
        if with_global_rating and (
            global_rating := self.dataset.get_movie_global_rating(movie_id)
        ):
            info += f" (Average rating: {global_rating} stars out of 5)"
        return info

    def convert_rating_to_str(self, rating: float) -> str:
        if self.double_range:
            rating *= 2

        if self.keep_trailing_zeroes:
            return str(rating)
        else:
            return f"{rating:g}"

    def get_user_identifier(self, shot: int) -> str:
        if self.numeric_user_identifier:
            _id = shot + 1
        else:
            _id = chr(65 + shot)

        return f'User "{_id}"'

    def get_user_rating_display(self, shot: int, rating: float, movie_id: int) -> str:
        movie_info = self.get_movie_info(
            movie_id=movie_id,
            with_genre=self.with_genre,
            with_global_rating=self.with_global_rating_in_context,
        )

        # NOTE: "\n" and " " separators between ratings in the listing are treated the same
        user_rating_versioned = {
            1: f"{self.get_user_identifier(shot=shot)} rated with {self.convert_rating_to_str(rating)} stars the movie {movie_info}.\n",
            2: f"- {movie_info}: {self.convert_rating_to_str(rating)} stars.\n",
            3: f"* {movie_info} - {self.convert_rating_to_str(rating)} stars.\n",
            4: f"* {movie_info} ({self.convert_rating_to_str(rating)} stars).\n",
        }

        return user_rating_versioned[self.rating_listing_version]

    def get_rated_movies_context(self, ratings_sample: pd.DataFrame, shot: int) -> str:
        context = ""
        for rating in ratings_sample.itertuples():
            context += self.get_user_rating_display(
                shot=shot, rating=rating.rating, movie_id=rating.movieId
            )

        return context.strip()

    def get_sample_header(self, kind: SampleKind, shot: int) -> str:
        versioned_headers = {
            1: "",
            2: f"Some of {self.get_user_identifier(shot=shot)}'s {kind.value}-rated movies:\n",
            3: f"Some {kind.value}-rated movies by {self.get_user_identifier(shot=shot)} include:\n",
        }

        return versioned_headers[self.sample_header_version]

    def get_context(self, user_id: int, shot: int) -> str:
        # Shuffled user ratings
        user_ratings = (
            self.dataset.training_df[self.dataset.training_df["userId"] == user_id]
            .sample(frac=1)
            .sort_values("rating", ascending=False)
        )

        likes_sample = user_ratings[user_ratings.rating >= 4][: self.likes_count]
        dislikes_sample = user_ratings[user_ratings.rating <= 2][::-1][
            : self.dislikes_count
        ]

        assert len(likes_sample) or len(dislikes_sample)

        rated_context_data = [
            (likes_sample, SampleKind.highest),
            (dislikes_sample, SampleKind.lowest),
        ]

        if not self.likes_first:
            rated_context_data = reversed(rated_context_data)

        context = ""
        for sample, kind in rated_context_data:
            if not len(sample):
                continue

            if context:
                context += "\n\n"

            context += self.get_sample_header(kind=kind, shot=shot)
            context += self.get_rated_movies_context(ratings_sample=sample, shot=shot)

        return self.get_context_header(shot=shot) + context

    def get_context_header(self, shot: int) -> str:
        header_versioned = {
            1: "",
            2: f"Here are some movie ratings from {self.get_user_identifier(shot=shot)}.\n\n",
            3: f"{self.get_user_identifier(shot=shot)} has provided ratings for various movies.\n\n",
            4: f"This is a selection of {self.get_user_identifier(shot=shot)}'s history of movie ratings.\n\n",
            5: f"Here are some of the highest and lowest ratings that {self.get_user_identifier(shot=shot)} has given to movies.\n\n",
        }
        return header_versioned[self.context_header_version]

    def get_task_description(self, movie_id: int, shot: int) -> str:
        versioned_descriptions = {
            1: f"On a scale of {', '.join(self.convert_rating_to_str(x) for x in POSSIBLE_VALUES)}, how would {self.get_user_identifier(shot=shot)} rate the movie {{}}?",
            2: f"How would {self.get_user_identifier(shot=shot)} rate the movie {{}} on a scale of {', '.join(self.convert_rating_to_str(x) for x in POSSIBLE_VALUES)}?",
            3: f"On a scale of {self.convert_rating_to_str(min(POSSIBLE_VALUES))} to {self.convert_rating_to_str(max(POSSIBLE_VALUES))}, how would {self.get_user_identifier(shot=shot)} rate the movie {{}}?",
            # NOTE: Using chr(10) (equivalent to '\n') circumvents Python's restriction on employing backslashes within f-string expressions.
            4: f"How would {self.get_user_identifier(shot=shot)} rate the movie {{}}?\nOPTIONS:\n- {(chr(10) + '- ').join(self.convert_rating_to_str(x) for x in POSSIBLE_VALUES)}",
            5: f"How would {self.get_user_identifier(shot=shot)} rate the movie {{}}?",
            6: f"Predict {self.get_user_identifier(shot=shot)}'s likely rating for the movie {{}} on a scale from {self.convert_rating_to_str(min(POSSIBLE_VALUES))} to {self.convert_rating_to_str(max(POSSIBLE_VALUES))}.",
            7: f"{self.get_user_identifier(shot=shot)} hasn't seen the movie {{}} yet. Predict how {self.get_user_identifier(shot=shot)} will likely rate the movie on a scale from {self.convert_rating_to_str(min(POSSIBLE_VALUES))} to {self.convert_rating_to_str(max(POSSIBLE_VALUES))}.",
            8: f"How would {self.get_user_identifier(shot=shot)} rate the movie {{}} on a scale of {self.convert_rating_to_str(min(POSSIBLE_VALUES))} to {self.convert_rating_to_str(max(POSSIBLE_VALUES))}?",

        }

        movie_info = self.get_movie_info(
            movie_id=movie_id,
            with_genre=self.with_genre,
            with_global_rating=self.with_global_rating_in_task,
        )
        return versioned_descriptions[self.task_desc_version].format(movie_info)

    def generate_zeroshot_prompt(self, user_id: int, movie_id: int, shot: int) -> str:
        task_description = self.get_task_description(movie_id=movie_id, shot=shot)
        task_description += self.get_answer_mark()

        if self.with_context:
            context = self.get_context(user_id=user_id, shot=shot)
            return f"{context}\n\n{task_description}"

        return task_description

    def get_answer_mark(self) -> str:
        mark_versioned = {
            1: "\n\n",
            2: "\n\nRating: ",
            3: "\n\nEstimated rating: ",
            4: "\n\nPredicted rating: ",
        }
        return mark_versioned[self.answer_mark_version]

    def __call__(self, user_id: int, movie_id: int) -> str:
        prompt = ""
        movie_ratings = self.dataset.training_df[
            self.dataset.training_df["movieId"] == movie_id
        ]
        example_ratings = movie_ratings.sample(
            n=min(self.shots, len(movie_ratings)), replace=False
        )
        i = -1
        for i, example in enumerate(example_ratings.itertuples()):
            prompt += self.generate_zeroshot_prompt(
                user_id=example.userId, movie_id=example.movieId, shot=i
            )
            prompt += f"{self.convert_rating_to_str(example.rating)}\n\n\n"

        zero_shot = self.generate_zeroshot_prompt(
            user_id=user_id, movie_id=movie_id, shot=i + 1
        )
        prompt += zero_shot

        return prompt


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

    except Exception:
        msg = f"Can't parse: {original_output!r}"
        logger.exception(msg)
        raise ValueError(msg)


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
    predictor = load_pipeline(task=task, precision=precision, use_flash_attention_2=use_flash_attention_2, model=model)
    run_experiment(predictor, **params)


def get_default_task(model: str) -> str:
    if "t5" in model.lower():
        return "text2text-generation"
    return "text-generation"


def load_pipeline(
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
        model_parameters.setdefault("model_kwargs", {})[
            "attn_implementation"
        ] = "flash_attention_2"

    predictor = get_pipeline(task=task, model=model, model_parameters=model_parameters)
    predictor.over_token_limit_count = 0
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
            predictor.over_token_limit_count += 1

        return inputs

    setattr(predictor, preprocess_method_name, _patched_preprocess)
    return predictor


def run_experiment(
    predictor: Pipeline,
    testing_ratio: float = 0.2,
    batch_size: int = 8,
    initial_run_seed: int = 0,
    model: str = "google/flan-t5-base",
    task: str = "text2text-generation",
    likes_count: int = 10,
    dislikes_count: int = 10,
    with_context: bool = True,
    likes_first: bool = True,
    task_desc_version: int = 1,
    shots: int = 0,
    with_genre: bool = False,
    with_global_rating_in_context: bool = False,
    with_global_rating_in_task: bool = False,
    temperature: float = 0.0,
    popularity: list = None,
    training_popularity: list = None,
    runs: int = 1,
    keep_trailing_zeroes: bool = True,
    double_range: bool = False,
    sample_header_version: int = 1,
    rating_listing_version: int = 1,
    context_header_version: int = 1,
    answer_mark_version: int = 1,
    numeric_user_identifier: bool = False,
    precision: str = "default",
    use_flash_attention_2: bool = False,
) -> float:
    initial_params = locals()
    del initial_params["predictor"]
    logger.info(
        f"Script parameters {' '.join(str(k) + '=' + str(v) for k, v in initial_params.items())}."
    )

    aggregated_rmse = []
    aggregated_precision = []
    aggregated_recall = []
    aggregated_f1 = []
    aggregated_value_counts = defaultdict(int, {v: 0 for v in POSSIBLE_VALUES})
    aggregated_retried_prompts = 0
    aggregated_retries = 0
    aggregated_unpredicted = 0

    for x in range(runs):
        run_params = initial_params.copy()
        run_seed = initial_run_seed + x
        run_params["run_seed"] = run_seed

        logger.info(f"Run {run_seed=}.")

        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        logger.info("Creating dataset...")
        dataset = MovieLensDataSet(
            testing_ratio=testing_ratio,
            training_popularity=training_popularity,
            popularity=popularity,
        )

        logger.info("Generating prompts...")

        prompt_generator = PromptGenerator(dataset=dataset, **run_params)
        prompts = [
            prompt_generator(user_id=row.userId, movie_id=row.movieId)
            for row in dataset.testing_df.itertuples()
        ]
        logger.info(f"Prompt Example:\n{prompts[0]}")

        logger.info("Running model...")
        model_parameters = {}
        if temperature == 0.0:
            model_parameters["do_sample"] = False
        else:
            model_parameters["do_sample"] = True
            model_parameters["temperature"] = temperature

        if task == "text-generation":
            model_parameters["return_full_text"] = False
            model_parameters["max_new_tokens"] = 20
            # NOTE: Needed for batching, as it's not set automatically in the pipeline like with other tasks

            if (
                predictor.tokenizer.pad_token_id
                and not predictor.model.config.pad_token_id
            ):
                predictor.model.config.pad_token_id = predictor.tokenizer.pad_token_id
            elif (
                not predictor.tokenizer.pad_token_id
                and predictor.model.config.pad_token_id
            ):
                predictor.tokenizer.pad_token_id = predictor.model.config.pad_token_id
            else:
                if "llama-2" in model.lower():
                    # Reference: https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020/2
                    predictor.tokenizer.pad_token = "[PAD]"
                    predictor.tokenizer.padding_side = "left"
                else:
                    predictor.tokenizer.pad_token_id = (
                        predictor.model.config.eos_token_id
                    )
                    predictor.model.config.pad_token_id = (
                        predictor.model.config.eos_token_id
                    )

        outputs = [
            p[0]["generated_text"]
            for p in tqdm(
                predictor(
                    MockListDataset(prompts), batch_size=batch_size, **model_parameters
                ),
                total=len(prompts),
            )
        ]
        logger.info("Parsing outputs...")

        def retry_inference(prompt, max_retries=3):
            model_parameters["do_sample"] = True

            for attempt in range(1, max_retries + 1):
                logger.info(f"Retrying, {attempt=}")
                output = predictor(prompt, **model_parameters)[0]["generated_text"]
                try:
                    pred = parse_model_output(output, double_range=double_range)
                except ValueError:
                    continue
                else:
                    return output, pred, attempt

            raise ValueError("Couldn't get prediction")

        retried_prompts = 0
        retries = 0
        max_retries = 3
        predictions = []
        unpredicted_indexes = set()
        for index, out in enumerate(outputs):
            try:
                pred = parse_model_output(out, double_range=double_range)
            except ValueError:
                try:
                    retried_prompts += 1
                    (
                        retried_output,
                        retried_prediction,
                        retried_attempts,
                    ) = retry_inference(prompt=prompts[index], max_retries=max_retries)
                    retries += retried_attempts
                    outputs[index] = retried_output
                    pred = retried_prediction
                except ValueError:
                    retries += max_retries
                    unpredicted_indexes.add(index)
                    pred = "N/A"

            predictions.append(pred)

        truth = [row.rating for row in dataset.testing_df.itertuples()]

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

        logger.info("Removing unpredicted items...")
        truth = [value for i, value in enumerate(truth) if i not in unpredicted_indexes]
        predictions = [
            value for i, value in enumerate(predictions) if i not in unpredicted_indexes
        ]
        logger.info(f"Unknown predictions: {len(unpredicted_indexes)}")
        aggregated_unpredicted += len(unpredicted_indexes)

        if not predictions:
            logger.info("All predictions are unknown, stopping experiment...")
            return [float("inf"), 0.]

        logger.info("Reporting metrics...")
        rmse = mean_squared_error(truth, predictions, squared=False)
        logger.info(f"RMSE: {rmse}")
        aggregated_rmse.append(rmse)

        predictions_df = pd.DataFrame(
            {
                "Truth": truth,
                "Prediction": predictions,
                "UserID": [
                    u
                    for i, u in enumerate(dataset.testing_df["userId"])
                    if i not in unpredicted_indexes
                ],
            }
        )

        logger.info(
            f"Classification report:\n{classification_report(predictions_df['Truth'] >= 4.0, predictions_df['Prediction'] >= 4.0)}"
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            predictions_df["Truth"] >= 4.0,
            predictions_df["Prediction"] >= 4.0,
            average="macro",
            zero_division=0.0,
        )
        aggregated_precision.append(precision)
        aggregated_recall.append(recall)
        aggregated_f1.append(f1)

        value_counts = defaultdict(int, {v: 0 for v in POSSIBLE_VALUES})
        for p in predictions:
            value_counts[p] += 1
            aggregated_value_counts[p] += 1
        distribution = {
            rating: round((count * 100 / len(predictions)), 2)
            for rating, count in sorted(value_counts.items())
        }
        logger.info(f"Distribution: {distribution}")

        aggregated_retried_prompts += retried_prompts
        logger.info(f"Retried prompts: {retried_prompts}")

        aggregated_retries += retries
        logger.info(f"Retries: {retries}")

    logger.info("Aggregated stats.")
    rmse_s = pd.Series(aggregated_rmse)
    logger.info(
        f"Aggregated RMSE. Median: {rmse_s.median()}. STD: {rmse_s.std(ddof=1)}"
    )

    precision_s = pd.Series(aggregated_precision)
    logger.info(
        f"Aggregated Precision. Median: {precision_s.median()}. STD: {precision_s.std(ddof=1)}"
    )

    recall_s = pd.Series(aggregated_recall)
    logger.info(
        f"Aggregated Recall. Median: {recall_s.median()}. STD: {recall_s.std(ddof=1)}"
    )

    f1_s = pd.Series(aggregated_f1)
    logger.info(f"Aggregated F1. Median: {f1_s.median()}. STD: {f1_s.std(ddof=1)}")

    total = sum(aggregated_value_counts.values())
    aggregated_distribution = {
        rating: round((count * 100 / total), 2)
        for rating, count in sorted(aggregated_value_counts.items())
    }
    logger.info(f"Aggregated Distribution: {aggregated_distribution}")

    aggregated_prompts = len(prompts) * runs

    logger.info(
        f"Aggregated Retried Prompts: {aggregated_retried_prompts} ({round(aggregated_retried_prompts * 100 / aggregated_prompts, 2)}%)"
    )
    logger.info(f"Aggregated Retries: {aggregated_retries}")

    logger.info(
        f"Aggregated Unknown Predictions: {aggregated_unpredicted} ({round(aggregated_unpredicted * 100 / aggregated_prompts, 2)}%)"
    )

    logger.info(
        f"Aggregated Over Limit Prompts: {predictor.over_token_limit_count} ({round(predictor.over_token_limit_count * 100 / aggregated_prompts, 2)}%)"
    )

    return [rmse_s.median(), f1_s.median()]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT
    )
    main()
