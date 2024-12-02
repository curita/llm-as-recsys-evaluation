from enum import Enum

import numpy as np
from numpy.random import RandomState
import pandas as pd

from llm_rec_eval.config import Config
from llm_rec_eval.constants import POSSIBLE_VALUES
from llm_rec_eval.dataset import MovieLensDataSet


class SampleKind(Enum):
    lowest = "lowest"
    highest = "highest"


class PromptGenerator:
    CONTEXT_HEADER_FORMATS = {
        1: "",
        2: "Here are some movie ratings from {user}.\n\n",
        3: "{user} has provided ratings for various movies.\n\n",
        4: "This is a selection of {user}'s history of movie ratings.\n\n",
        5: "Here are some of the highest and lowest ratings that {user} has given to movies.\n\n",
    }

    SAMPLE_HEADER_FORMATS = {
        1: "",
        2: "Some of {user}'s {kind}-rated movies:\n",
        3: "Some {kind}-rated movies by {user} include:\n",
    }

    RATING_LISTING_FORMATS = {
        1: "{user} rated with {rating} stars the movie {movie}.\n",
        2: "- {movie}: {rating} stars.\n",
        3: "* {movie} - {rating} stars.\n",
        4: "* {movie} ({rating} stars).\n",
    }

    TASK_DESCRIPTION_FORMATS = {
        1: "On a scale of {values}, how would {user} rate the movie {movie}?",
        2: "How would {user} rate the movie {movie} on a scale of {values}?",
        3: "On a scale of {min_value} to {max_value}, how would {user} rate the movie {movie}?",
        4: "How would {user} rate the movie {movie}?\nOPTIONS:{bulleted_values}",
        5: "How would {user} rate the movie {movie}?",
        6: "Predict {user}'s likely rating for the movie {movie} on a scale from {min_value} to {max_value}.",
        7: "{user} hasn't seen the movie {movie} yet. Predict how {user} will likely rate the movie on a scale from {min_value} to {max_value}.",
        8: "How would {user} rate the movie {movie} on a scale of {min_value} to {max_value}?",
    }

    ANSWER_MARK_FORMATS = {
        1: "\n\n",
        2: "\n\nRating: ",
        3: "\n\nEstimated rating: ",
        4: "\n\nPredicted rating: ",
    }

    def __init__(
        self,
        dataset: MovieLensDataSet,
        config: Config,
        random_state: int | RandomState | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.rng = (
            np.random.default_rng(random_state)
            if isinstance(random_state, int)
            else random_state
        )

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
        if self.config.double_range:
            rating *= 2

        if self.config.keep_trailing_zeroes:
            return str(rating)
        else:
            return f"{rating:g}"

    def get_user_identifier(self, shot: int) -> str:
        _id = shot + 1 if self.config.numeric_user_identifier else chr(65 + shot)

        return f'User "{_id}"'

    def get_user_rating_display(self, shot: int, rating: float, movie_id: int) -> str:
        movie_info = self.get_movie_info(
            movie_id=movie_id,
            with_genre=self.config.with_genre,
            with_global_rating=self.config.with_global_rating_in_context,
        )

        return self.RATING_LISTING_FORMATS[self.config.rating_listing_version].format(
            user=self.get_user_identifier(shot=shot),
            rating=self.convert_rating_to_str(rating),
            movie=movie_info,
        )

    def get_rated_movies_context(self, ratings_sample: pd.DataFrame, shot: int) -> str:
        context = ""
        for rating in ratings_sample.itertuples():
            context += self.get_user_rating_display(
                shot=shot, rating=rating.rating, movie_id=rating.movieId
            )

        return context.strip()

    def get_sample_header(self, kind: SampleKind, shot: int) -> str:
        return self.SAMPLE_HEADER_FORMATS[self.config.sample_header_version].format(
            user=self.get_user_identifier(shot=shot), kind=kind.value
        )

    def get_context(self, user_id: int, shot: int) -> str:
        # Shuffled user ratings
        user_ratings = (
            self.dataset.training_df[self.dataset.training_df["userId"] == user_id]
            .sample(frac=1, random_state=self.rng)
            .sort_values("rating", ascending=False)
        )

        likes_sample = user_ratings[user_ratings.rating >= 4][: self.config.likes_count]
        dislikes_sample = user_ratings[user_ratings.rating <= 2][::-1][
            : self.config.dislikes_count
        ]

        assert len(likes_sample) or len(dislikes_sample)

        rated_context_data = [
            (likes_sample, SampleKind.highest),
            (dislikes_sample, SampleKind.lowest),
        ]

        if not self.config.likes_first:
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
        return self.CONTEXT_HEADER_FORMATS[self.config.context_header_version].format(
            user=self.get_user_identifier(shot=shot)
        )

    def get_task_description(self, movie_id: int, shot: int) -> str:
        movie_info = self.get_movie_info(
            movie_id=movie_id,
            with_genre=self.config.with_genre,
            with_global_rating=self.config.with_global_rating_in_task,
        )
        values = ", ".join(self.convert_rating_to_str(x) for x in POSSIBLE_VALUES)
        min_value = self.convert_rating_to_str(min(POSSIBLE_VALUES))
        max_value = self.convert_rating_to_str(max(POSSIBLE_VALUES))
        bulleted_values = "\n- " + "\n- ".join(
            self.convert_rating_to_str(x) for x in POSSIBLE_VALUES
        )
        return self.TASK_DESCRIPTION_FORMATS[self.config.task_desc_version].format(
            user=self.get_user_identifier(shot=shot),
            movie=movie_info,
            values=values,
            min_value=min_value,
            max_value=max_value,
            bulleted_values=bulleted_values,
        )

    def generate_zeroshot_prompt(self, user_id: int, movie_id: int, shot: int) -> str:
        task_description = self.get_task_description(movie_id=movie_id, shot=shot)
        task_description += self.get_answer_mark()

        if self.config.with_context:
            context = self.get_context(user_id=user_id, shot=shot)
            return f"{context}\n\n{task_description}"

        return task_description

    def get_answer_mark(self) -> str:
        return self.ANSWER_MARK_FORMATS[self.config.answer_mark_version]

    def __call__(self, user_id: int, movie_id: int) -> str:
        prompt = ""
        movie_ratings = self.dataset.training_df[
            self.dataset.training_df["movieId"] == movie_id
        ]
        example_ratings = movie_ratings.sample(
            n=min(self.config.shots, len(movie_ratings)),
            replace=False,
            random_state=self.rng,
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
