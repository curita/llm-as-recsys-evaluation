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
from sklearn.metrics import mean_squared_error, classification_report


logger = logging.getLogger(__name__)


def round_to_nearest_half(number):
    return round(number * 2) / 2

class MovieLensDataSet:
    def __init__(self, training_ratio: float) -> None:
        self.ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
        self.movies_df = pd.read_csv("ml-latest-small/movies.csv")
        self.normalize_movie_titles()

        self.training_df = self.ratings_df.sample(frac=training_ratio, replace=False)
        self.testing_df = self.ratings_df.loc[self.ratings_df.index.difference(self.training_df.index)]

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

    def get_movie_name(self, movie_id: int) -> str:
        return self.movies_df[self.movies_df["movieId"] == movie_id][
            "normalize_title"
        ].iloc[0]

    def get_movie_genres(self, movie_id: int) -> list[str]:
        return self.movies_df[self.movies_df["movieId"] == movie_id]["genres"].iloc[0].split("|")

    def get_movie_global_rating(self, movie_id: int) -> float:
        return round_to_nearest_half(self.ratings_df[self.ratings_df["movieId"] == movie_id]["rating"].median())
    

class PromptGenerator:

    def __init__(self, dataset: MovieLensDataSet, with_genre: bool, with_global_rating: bool, likes_first: bool, likes_count: int, dislikes_count: int, task_desc_version: int, with_context: bool, shot: int, **kwargs) -> None:
        self.dataset = dataset
        self.with_genre = with_genre
        self.with_global_rating = with_global_rating
        self.likes_first = likes_first
        self.likes_count = likes_count
        self.dislikes_count = dislikes_count
        self.task_desc_version = task_desc_version
        self.with_context = with_context
        self.shot = shot

    def get_rated_movies(
        self,
        user_id: int,
        n: int,
        rating: float,
    ) -> list[int]:
        # XXX: Fallback to close ratings if there are not enough movies to fill that sample
        rated_movies = self.dataset.training_df[
            (self.dataset.training_df["userId"] == user_id) & (self.dataset.training_df["rating"] == rating)
        ]
        if len(rated_movies) == 0:
            return []

        n = min(len(rated_movies), n)
        return (
            rated_movies["movieId"].sample(n=n, replace=False).to_list()
        )

    def get_movie_info(self, movie_id: int, with_genre: bool, with_global_rating: bool) -> str:
        info = f'"{self.dataset.get_movie_name(movie_id)}"'
        if with_genre:
            info += f' ({"|".join(self.dataset.get_movie_genres(movie_id))})'
        if with_global_rating:
            info += f' (Average rating: {self.dataset.get_movie_global_rating(movie_id)} stars out of 5)'
        return info

    def get_rated_movies_context(self, rating: float, sample: list[int], initial_prefix: str = "A") -> str:
        context = ""
        prefix = initial_prefix
        for x in sample:
            movie_info = self.get_movie_info(movie_id=x, with_genre=self.with_genre, with_global_rating=False)
            context += f'{prefix} user rated with {rating} stars the movie {movie_info}.'
            prefix = " The"

        return context

    def get_context(self, user_id: int) -> str:
        user_max_rating = self.dataset.training_df[self.dataset.training_df["userId"] == user_id]["rating"].max()
        user_min_rating = self.dataset.training_df[self.dataset.training_df["userId"] == user_id]["rating"].min()

        if user_max_rating == user_min_rating:
            # There are no 0.0 ratings, so this essentially turns off the dislikes
            user_min_rating = 0.0

        likes_sample = self.get_rated_movies(
            user_id=user_id, n=self.likes_count, rating=user_max_rating
        )
        dislikes_sample = self.get_rated_movies(
            user_id=user_id, n=self.dislikes_count, rating=user_min_rating
        )

        assert likes_sample or dislikes_sample

        rated_context_data = [(user_max_rating, likes_sample), (user_min_rating, dislikes_sample)]

        if not self.likes_first:
            rated_context_data = reversed(rated_context_data)

        context = ""
        for rating, sample in rated_context_data:
            if not sample:
                continue

            if not context:
                prefix = "A"
            else:
                prefix = "\n\nThe"
            context += self.get_rated_movies_context(rating=rating, sample=sample, initial_prefix=prefix)

        return context

    def get_task_description(self, movie_id: int) -> str:
        versioned_descriptions = {
            1: 'On a scale of 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, how would the user rate the movie {}?',
            2: 'How would the user rate the movie {} on a scale of 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.0?',
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
        example_ratings = self.dataset.training_df.sample(n=self.shot, replace=False)
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

@click.command()
@click.option("--dataset-seed", default=0, type=int)
@click.option("--training-ratio", default=0.8, type=float)
@click.option("--batch-size", default=8, type=int)
@click.option("--prompt-seed", default=0, type=int)
@click.option("--model", default="google/flan-t5-base", type=str)
@click.option("--likes-count", default=10, type=int)
@click.option("--dislikes-count", default=10, type=int)
@click.option("--with-context/--without-context", default=True)
@click.option("--likes-first/--dislikes-first", default=True)
@click.option("--task-desc-version", default=1, type=int)
@click.option("--shot", default=0, type=int)
@click.option("--with-genre/--without-genre", default=False)
@click.option("--with-global-rating/--without-global-rating", default=False)
@click.option("--temperature", default=0, type=float)
@click.pass_context
def main(ctx, dataset_seed, training_ratio, batch_size, prompt_seed, model, likes_count, dislikes_count, with_context, likes_first, task_desc_version, shot, with_genre, with_global_rating, temperature):

    logger.info(f"Run {' '.join(str(k) + '=' + str(v) for k, v in ctx.params.items())}.")
    logger.info("Creating dataset...")
    np.random.seed(dataset_seed)
    dataset = MovieLensDataSet(training_ratio=training_ratio)
    logger.info("Generating prompts...")
    np.random.seed(prompt_seed)
    torch.manual_seed(prompt_seed)
    prompt_generator = PromptGenerator(dataset=dataset, **ctx.params)
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

    folder_name = f"experiment_{'_'.join(str(k) + '=' + str(v) for k, v in ctx.params.items())}".replace("/", ":")
    output_folder = Path(f"results") / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Path: {output_folder / 'results.csv'}")

    with open(output_folder / "results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Prompt", "Movie", "MovieID", "UserID", "Output", "Prediction", "Truth"])

        writer.writeheader()
        for prmpt, out, pred, row in zip(prompts, outputs, predictions, dataset.testing_df.itertuples()):
            writer.writerow({'Prompt': prmpt, "Movie": dataset.get_movie_name(row.movieId), "MovieID": row.movieId, "UserID": row.userId, "Output": out, "Prediction": str(pred), "Truth": str(row.rating)})

    logger.info("Reporting metrics...")
    logger.info(f"RMSE: {mean_squared_error(truth, predictions, squared=False)}")
    logger.info(f"Classification report:\n{classification_report([str(x) for x in truth], [str(x) for x in predictions])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT)
    main()
