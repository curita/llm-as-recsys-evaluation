import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from numpy.random import RandomState

from llm_rec_eval.constants import FREQUENCY_CATEGORIES


def round_to_nearest_half(number):
    return round(number * 2) / 2


class PyTorchListDataset(Dataset):
    """Class that wraps a list to be used as a PyTorch dataset.

    When a list of prompts wrapped in this class is passed to HuggingFace's
    `pipeline()`, the latter will return a generator with the model outputs
    instead of a list. This way batched responses will be yielded as they are
    ready instead of waiting for the whole prompt list to be processed.
    """

    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


class MovieLensDataSet:
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        testing_ratio: float = 0.2,
        training_popularity: tuple[str] = None,
        popularity: tuple[str] = None,
        random_state: int | RandomState = None,
    ) -> None:
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.normalize_movie_titles()
        self.categorize_movie_popularity()

        if popularity:
            self.ratings_df = self.ratings_df[
                self.ratings_df.popularity.isin(popularity)
            ]

        self.training_df, self.testing_df = train_test_split(
            self.ratings_df, test_size=testing_ratio, random_state=random_state
        )

        if training_popularity:
            self.training_df = self.training_df[
                self.training_df.popularity.isin(training_popularity)
            ]

    @classmethod
    def from_csv(
        cls,
        ratings_path: str = "./data/ml-latest-small/ratings.csv",
        movies_path: str = "./data/ml-latest-small/movies.csv",
        **kwargs,
    ):
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
        return cls(ratings_df, movies_df, **kwargs)

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
        rating_counts = (
            self.ratings_df.groupby("movieId")
            .count()
            .userId.to_frame()
            .rename(columns={"userId": "ratingCount"})
        )
        rating_counts["popularity"] = pd.cut(
            rating_counts.ratingCount,
            [0, 2, 10, 50, 300],
            labels=FREQUENCY_CATEGORIES,
        )
        self.ratings_df = self.ratings_df.merge(rating_counts, on="movieId", how="left")

    def get_movie_name(self, movie_id: int) -> str:
        return self.movies_df[self.movies_df["movieId"] == movie_id][
            "normalize_title"
        ].iloc[0]

    def get_movie_genres(self, movie_id: int) -> list[str]:
        return (
            self.movies_df[self.movies_df["movieId"] == movie_id]["genres"]
            .iloc[0]
            .split("|")
        )

    def get_movie_global_rating(self, movie_id: int) -> float | None:
        movie_ratings = self.training_df[self.training_df["movieId"] == movie_id][
            "rating"
        ]
        if not len(movie_ratings):
            return
        return round_to_nearest_half(movie_ratings.mean())
