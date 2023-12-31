from collections import defaultdict
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
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, classification_report, precision_recall_fscore_support


logger = logging.getLogger(__name__)


def round_to_nearest_half(number):
    return round(number * 2) / 2


POSSIBLE_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
FREQUENCY_CATEGORIES = ["rare", "unfrequent", "normal", "very_frequent"]

class MovieLensDataSet:
    def __init__(self, training_ratio: float, training_popularity: tuple[str], popularity: tuple[str]) -> None:
        self.ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
        self.movies_df = pd.read_csv("ml-latest-small/movies.csv")
        self.normalize_movie_titles()
        self.categorize_movie_popularity()

        if popularity:
            self.ratings_df = self.ratings_df[self.ratings_df.popularity.isin(popularity)]

        self.training_df = self.ratings_df.sample(frac=training_ratio, replace=False)
        self.testing_df = self.ratings_df.loc[self.ratings_df.index.difference(self.training_df.index)]

        if training_popularity:
            self.training_df = self.training_df[self.training_df.popularity.isin(training_popularity)]

    def normalize_movie_titles(self):
        self.movies_df["normalize_title"] = self.movies_df["title"].str.replace(
            r"^(.+), The (\(\d{4}\))$", r"The \1 \2", regex=True
        )
        self.movies_df["normalize_title"] = self.movies_df[
            "normalize_title"
        ].str.replace(r"^(.+), An (\(\d{4}\))$", r"An \1 \2", regex=True)
        self.movies_df["normalize_title"] = self.movies_df[
            "normalize_title"
        ].str.replace(r"^(.+), A (\(\d{4}\))$", r"A \1 \2", regex=True)

    def categorize_movie_popularity(self):
        rating_counts = self.ratings_df.groupby('movieId').count().userId.to_frame().rename(columns={'userId': 'ratingCount'})
        rating_counts["popularity"] = pd.cut(
            rating_counts.ratingCount, [0, 2, 10, 50, 300],
            labels=FREQUENCY_CATEGORIES,
        )
        self.ratings_df = self.ratings_df.merge(rating_counts, on='movieId', how="left")

    def get_movie_name(self, movie_id: int) -> str:
        return self.movies_df[self.movies_df["movieId"] == movie_id][
            "normalize_title"
        ].iloc[0]

    def get_movie_genres(self, movie_id: int) -> list[str]:
        return self.movies_df[self.movies_df["movieId"] == movie_id]["genres"].iloc[0].split("|")

    def get_movie_global_rating(self, movie_id: int) -> float | None:
        movie_ratings = self.training_df[self.training_df["movieId"] == movie_id]["ratings"]
        if not len(movie_ratings):
            return
        return round_to_nearest_half(movie_ratings.median())
    

class PromptGenerator:

    def __init__(self, dataset: MovieLensDataSet, with_genre: bool, with_global_rating: bool, likes_first: bool, likes_count: int, dislikes_count: int, task_desc_version: int, with_context: bool, shots: int, keep_trailing_zeroes: bool, **kwargs) -> None:
        self.dataset = dataset
        self.with_genre = with_genre
        self.with_global_rating = with_global_rating
        self.likes_first = likes_first
        self.likes_count = likes_count
        self.dislikes_count = dislikes_count
        self.task_desc_version = task_desc_version
        self.with_context = with_context
        self.shots = shots
        self.keep_trailing_zeroes = keep_trailing_zeroes

    def get_movie_info(self, movie_id: int, with_genre: bool, with_global_rating: bool) -> str:
        info = f'"{self.dataset.get_movie_name(movie_id)}"'
        if with_genre:
            info += f' ({"|".join(self.dataset.get_movie_genres(movie_id))})'
        if with_global_rating and (global_rating := self.dataset.get_movie_global_rating(movie_id)):
            info += f' (Average rating: {global_rating} stars out of 5)'
        return info

    def convert_rating_to_str(self, rating: float) -> str:
        if self.keep_trailing_zeroes:
            return str(rating)
        else:
            return f"{rating:g}"

    def get_rated_movies_context(self, ratings_sample: pd.DataFrame, initial_prefix: str = "A") -> str:
        context = ""
        prefix = initial_prefix
        for rating in ratings_sample.itertuples():
            movie_info = self.get_movie_info(movie_id=rating.movieId, with_genre=self.with_genre, with_global_rating=False)
            context += f'{prefix} user rated with {self.convert_rating_to_str(rating.rating)} stars the movie {movie_info}.'
            prefix = " The"

        return context

    def get_context(self, user_id: int) -> str:
        # Shuffled user ratings
        user_ratings = self.dataset.training_df[self.dataset.training_df["userId"] == user_id].sample(frac=1).sort_values("rating", ascending=False)

        likes_sample = user_ratings[user_ratings.rating >= 4][:self.likes_count]
        dislikes_sample = user_ratings[user_ratings.rating <= 2][::-1][:self.dislikes_count]

        assert len(likes_sample) or len(dislikes_sample)

        rated_context_data = [likes_sample, dislikes_sample]

        if not self.likes_first:
            rated_context_data = reversed(rated_context_data)

        context = ""
        for sample in rated_context_data:
            if not len(sample):
                continue

            if not context:
                prefix = "A"
            else:
                prefix = "\n\nThe"
            context += self.get_rated_movies_context(ratings_sample=sample, initial_prefix=prefix)

        return context

    def get_task_description(self, movie_id: int) -> str:
        versioned_descriptions = {
            1: f"On a scale of {', '.join(self.convert_rating_to_str(x) for x in POSSIBLE_VALUES)}, how would the user rate the movie {{}}?",
            2: f"How would the user rate the movie {{}} on a scale of {', '.join(self.convert_rating_to_str(x) for x in POSSIBLE_VALUES)}?",
        }

        movie_info = self.get_movie_info(movie_id=movie_id, with_genre=self.with_genre, with_global_rating=self.with_global_rating)
        return versioned_descriptions[self.task_desc_version].format(movie_info)

    def generate_zeroshot_prompt(self, user_id: int, movie_id: int) -> str:
        task_description = self.get_task_description(movie_id=movie_id)

        if self.with_context:
            context = self.get_context(user_id=user_id)
            return f"{context}\n\n{task_description}"

        return task_description

    def __call__(self, user_id: int, movie_id: int) -> str:
        prompt = ""
        movie_ratings = self.dataset.training_df[self.dataset.training_df["movieId"] == movie_id]
        example_ratings = movie_ratings.sample(n=min(self.shots, len(movie_ratings)), replace=False)
        for example in example_ratings.itertuples():
            prompt += self.generate_zeroshot_prompt(user_id=example.userId, movie_id=example.movieId)
            prompt += f'\n{example.rating}\n\n\n'

        zero_shot = self.generate_zeroshot_prompt(user_id=user_id, movie_id=movie_id)
        prompt += zero_shot
        return prompt

class MockListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def parse_model_output(output: str) -> bool:
    try:
        for string, replacement in {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", ",": "."}.items():
            output = output.replace(string, replacement)
        return float(re.findall(r"(\d+(?:.\d+)?)", output)[0])
    except Exception:
        raise ValueError(output)

FILENAME_PARAMETERS = {
    "RATIO": "training_ratio",
    "SEED": "run_seed",
    "M": "model",
    "L": "likes_count",
    "D": "dislikes_count",
    "C": "with_context",
    "F": "likes_first",
    "V": "task_desc_version",
    "S": "shots",
    "G": "with_genre",
    "R": "with_global_rating",
    "T": "temperature",
    "P": "popularity",
    "TP": "training_popularity",
    "Z": "keep_trailing_zeroes",
}


@click.command()
@click.option("--dataset-seed", default=0, type=int)
@click.option("--training-ratio", default=0.8, type=float)
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
@click.option("--with-global-rating/--without-global-rating", default=False)
@click.option("--temperature", default=0, type=float)
@click.option("--popularity", multiple=True, type=click.Choice(FREQUENCY_CATEGORIES))
@click.option("--training-popularity", multiple=True, type=click.Choice(FREQUENCY_CATEGORIES))
@click.option("--runs", default=1, type=int)
@click.option("--keep-trailing-zeroes/--strip-trailing-zeros", default=True)
@click.pass_context
def main(ctx, dataset_seed, training_ratio, batch_size, initial_run_seed, model, likes_count, dislikes_count, with_context, likes_first, task_desc_version, shots, with_genre, with_global_rating, temperature, popularity, training_popularity, runs, keep_trailing_zeroes):

    logger.info(f"Script parameters {' '.join(str(k) + '=' + str(v) for k, v in ctx.params.items())}.")

    logger.info("Creating dataset...")
    np.random.seed(dataset_seed)
    dataset = MovieLensDataSet(training_ratio=training_ratio, training_popularity=training_popularity, popularity=popularity)

    aggregated_rmse = []
    aggregated_precision = []
    aggregated_recall = []
    aggregated_f1 = []
    aggregated_value_counts = defaultdict(int, {v: 0 for v in POSSIBLE_VALUES})

    for x in range(runs):

        run_params = ctx.params.copy()
        run_seed = initial_run_seed + x
        run_params["run_seed"] = run_seed

        logger.info(f"Run {run_seed=}.")

        logger.info("Generating prompts...")
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        prompt_generator = PromptGenerator(dataset=dataset, **run_params)
        prompts = [
            prompt_generator(user_id=row.userId, movie_id=row.movieId)
            for row in dataset.testing_df.itertuples()
        ]
        logger.info(f"Prompt Example:\n{prompts[0]}")
        logger.info("Initializing text-generation pipeline...")
        text2textgenerator = pipeline("text2text-generation", model=model, device_map="auto")
        logger.info("Running model...")
        model_parameters = {}
        if temperature == 0.0:
            model_parameters["do_sample"] = False
        else:
            model_parameters["do_sample"] = True
            model_parameters["temperature"] = temperature
        outputs = [p[0]["generated_text"] for p in tqdm(text2textgenerator(MockListDataset(prompts), batch_size=batch_size, **model_parameters), total=len(prompts))]
        logger.info("Parsing outputs...")
        predictions = [parse_model_output(o) for o in outputs]
        truth = [row.rating for row in dataset.testing_df.itertuples()]

        logger.info("Dumping results...")

        folder_name = f"experiment_{'_'.join(k + '=' + str(run_params[v]) for k, v in FILENAME_PARAMETERS.items())}".replace("/", ":")
        output_folder = Path(f"results") / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / "results.csv"

        logger.info(f"Path: {output_file}")

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Prompt", "Movie", "MovieID", "UserID", "Output", "Prediction", "Truth", "Parameters"])

            writer.writeheader()
            parameters = json.dumps(run_params)
            for prmpt, out, pred, row in zip(prompts, outputs, predictions, dataset.testing_df.itertuples()):
                writer.writerow({'Prompt': prmpt, "Movie": dataset.get_movie_name(row.movieId), "MovieID": row.movieId, "UserID": row.userId, "Output": out, "Prediction": str(pred), "Truth": str(row.rating), "Parameters": parameters})
                parameters = ""

        logger.info("Reporting metrics...")
        rmse = mean_squared_error(truth, predictions, squared=False)
        logger.info(f"RMSE: {rmse}")
        aggregated_rmse.append(rmse)

        logger.info(f"Classification report:\n{classification_report([str(x) for x in truth], [str(x) for x in predictions])}")
        precision, recall, f1, _ = precision_recall_fscore_support([str(x) for x in truth], [str(x) for x in predictions], average="macro")
        aggregated_precision.append(precision)
        aggregated_recall.append(recall)
        aggregated_f1.append(f1)

        value_counts = defaultdict(int, {v: 0 for v in POSSIBLE_VALUES})
        for p in predictions:
            value_counts[p] += 1
            aggregated_value_counts[p] += 1
        distribution = {rating: round((count * 100 / len(predictions)), 2) for rating, count in sorted(value_counts.items())}
        logger.info(f"Distribution: {distribution}")

    logger.info("Aggregated stats.")
    rmse_s = pd.Series(aggregated_rmse)
    logger.info(f"Aggregated RMSE. Median: {rmse_s.median()}. STD: {rmse_s.std(ddof=1)}")

    precision_s = pd.Series(aggregated_precision)
    logger.info(f"Aggregated Precision-macro. Median: {precision_s.median()}. STD: {precision_s.std(ddof=1)}")

    recall_s = pd.Series(aggregated_recall)
    logger.info(f"Aggregated Recall-macro. Median: {recall_s.median()}. STD: {recall_s.std(ddof=1)}")

    f1_s = pd.Series(aggregated_f1)
    logger.info(f"Aggregated F1-macro. Median: {f1_s.median()}. STD: {f1_s.std(ddof=1)}")

    total = sum(aggregated_value_counts.values())
    aggregated_distribution = {rating: round((count * 100 / total), 2) for rating, count in sorted(aggregated_value_counts.items())}
    logger.info(f"Aggregated Distribution: {aggregated_distribution}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT)
    main()
