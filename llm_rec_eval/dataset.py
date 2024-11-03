import pandas as pd
from sklearn.model_selection import train_test_split

from llm_rec_eval.constants import FREQUENCY_CATEGORIES


def round_to_nearest_half(number):
    return round(number * 2) / 2


class MovieLensDataSet:
    def __init__(
        self,
        testing_ratio: float,
        training_popularity: tuple[str],
        popularity: tuple[str],
    ) -> None:
        self.ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
        self.movies_df = pd.read_csv("ml-latest-small/movies.csv")
        self.normalize_movie_titles()
        self.categorize_movie_popularity()

        if popularity:
            self.ratings_df = self.ratings_df[
                self.ratings_df.popularity.isin(popularity)
            ]

        self.training_df, self.testing_df = train_test_split(
            self.ratings_df, test_size=testing_ratio
        )

        if training_popularity:
            self.training_df = self.training_df[
                self.training_df.popularity.isin(training_popularity)
            ]

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
