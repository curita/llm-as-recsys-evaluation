from pathlib import Path
import re
import csv
import logging

import click
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, classification_report


logger = logging.getLogger(__name__)


class MovieLensDataSet:
    def __init__(self, training_ratio: float, seed: int) -> None:
        self.ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
        self.movies_df = pd.read_csv("ml-latest-small/movies.csv")
        self.normalize_movie_titles()

        training_size = int(len(self.ratings_df) * training_ratio)
        self.training_df = self.ratings_df.sample(n=training_size, replace=False, random_state=seed)
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


def get_rated_movies(
    dataset: MovieLensDataSet,
    user_id: int,
    n: int,
    rating: float,
    seed: int,
) -> list[int]:
    # XXX: Fallback to close ratings if there are not enough movies to fill that sample
    rated_movies = dataset.ratings_df[
        (dataset.ratings_df["userId"] == user_id) & (dataset.ratings_df["rating"] == rating)
    ]
    if len(rated_movies) == 0:
        return []

    n = min(len(rated_movies), n)
    return (
        rated_movies["movieId"].sample(n=n, random_state=seed, replace=False).to_list()
    )


def get_context(dataset: MovieLensDataSet, user_id: int, likes_count: int, dislikes_count: int, seed: int):
    user_max_rating = dataset.ratings_df[dataset.ratings_df["userId"] == user_id]["rating"].max()
    user_min_rating = dataset.ratings_df[dataset.ratings_df["userId"] == user_id]["rating"].min()

    if user_max_rating == user_min_rating:
        # There are no 0.0 ratings, so this essentially turns off the dislikes
        user_min_rating = 0.0

    likes_sample = get_rated_movies(
        dataset=dataset, user_id=user_id, n=likes_count, rating=user_max_rating, seed=seed
    )
    dislikes_sample = get_rated_movies(
        dataset=dataset, user_id=user_id, n=dislikes_count, rating=user_min_rating, seed=seed
    )

    assert likes_sample or dislikes_sample

    # XXX: Abstract these into a single helper for both cases
    context = ""
    if likes_sample:
        context = f'A user rated with {user_max_rating} stars the movie "{dataset.get_movie_name(likes_sample[0])}".'
        for x in likes_sample[1:]:
            context += f' The user rated with {user_max_rating} stars the movie "{dataset.get_movie_name(x)}".'

    if dislikes_sample:
        if not context:
            prefix = "A"
        else:
            prefix = "\n\nThe"

        context += f'{prefix} user rated with {user_min_rating} stars the movie "{dataset.get_movie_name(dislikes_sample[0])}".'
        for x in dislikes_sample[1:]:
            context += f' The user rated with {user_min_rating} stars the movie "{dataset.get_movie_name(x)}".'

    return context


def get_task_description(dataset: MovieLensDataSet, movie_id: int):
    return f'On the scale of 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5 (0.5 being lowest and 5 being highest), how would the user rate the movie "{dataset.get_movie_name(movie_id)}"?'


def generate_prompt(
    dataset: MovieLensDataSet, user_id: int, movie_id: int, with_context: bool, likes_count: int, dislikes_count: int, seed: int
) -> str:
    task_description = get_task_description(dataset=dataset, movie_id=movie_id)

    if with_context:
        context = get_context(dataset=dataset, user_id=user_id, likes_count=likes_count, dislikes_count=dislikes_count, seed=seed)
        return f"{context}.\n\n{task_description}"

    return task_description


class MockListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def parse_model_output(output: str) -> bool:
    return float(re.findall(r"(\d(?:.\d)?)(?: stars)?", output)[0])

@click.command()
@click.option("--dataset-seed", default=0, type=int)
@click.option("--training-ratio", default=0.8, type=float)
@click.option("--batch-size", default=8, type=int)
@click.option("--prompt-seed", default=0, type=int)
@click.option("--model", default="google/flan-t5-base", type=str)
@click.option("--likes-count", default=10, type=int)
@click.option("--dislikes-count", default=10, type=int)
@click.option("--with-context/--without-context", default=True)
def main(dataset_seed, training_ratio, batch_size, prompt_seed, model, likes_count, dislikes_count, with_context):
    logger.info(f"Run {dataset_seed=} {training_ratio=} {batch_size=} {prompt_seed=} {model=} {likes_count=} {dislikes_count=} {with_context=}.")
    logger.info("Creating dataset...")
    dataset = MovieLensDataSet(seed=dataset_seed, training_ratio=training_ratio)
    logger.info("Generating prompts...")
    prompts = [
      generate_prompt(dataset=dataset, user_id=row.userId, movie_id=row.movieId, with_context=with_context, likes_count=likes_count, dislikes_count=dislikes_count, seed=prompt_seed)
      for row in dataset.testing_df.itertuples()
    ]
    logger.info("Initializing text-generation pipeline...")
    text2textgenerator = pipeline("text2text-generation", model=model, device_map="auto")
    logger.info("Running model...")
    outputs = [p[0] for p in tqdm(text2textgenerator(MockListDataset(prompts), batch_size=batch_size, do_sample=False), total=len(prompts))]
    logger.info("Parsing outputs...")
    predictions = [parse_model_output(o["generated_text"]) for o in outputs]
    truth = [row.rating for row in dataset.testing_df.itertuples()]

    logger.info("Dumping results...")

    folder_name = f"experiment_{training_ratio=}_{prompt_seed=}_{model=}_{with_context=}_{likes_count=}_{dislikes_count=}".replace("/", ":")
    output_folder = Path(f"results") / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(output_folder / "results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Prompt", "Movie", "Output", "Prediction", "Truth"])

        writer.writeheader()
        for prmpt, out, pred, row in zip(prompts, outputs, predictions, dataset.testing_df.itertuples()):
            writer.writerow({'Prompt': prmpt, "Movie": dataset.get_movie_name(row.movieId), "Output": out, "Prediction": str(pred), "Truth": str(row.rating)})

    logger.info("Reporting metrics...")
    logger.info(f"RMSE: {mean_squared_error(truth, predictions, squared=False)}")
    logger.info(f"Classification report:\n{classification_report([str(x) for x in truth], [str(x) for x in predictions])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT)
    main()
