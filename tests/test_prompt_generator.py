import pytest
import pandas as pd
from llm_rec_eval.config import Config
from llm_rec_eval.dataset import MovieLensDataSet
from llm_rec_eval.prompts import PromptGenerator


@pytest.fixture
def sample_ratings_df():
    """Create a sample ratings DataFrame for testing."""
    # fmt: off
    ratings_df =  pd.DataFrame({
        'userId': [
            1, 1, 1, 1, 1,  # User 1's ratings
            2, 2, 2, 2, 2,  # User 2's ratings
        ],
        'movieId': [
            101, 102, 103, 104, 105,  # User 1's movies
            201, 202, 203, 204, 105,  # User 2's movies
        ],
        'rating': [
            5.0, 4.5, 1.0, 2.0, 4.5,  # User 1's ratings
            5.0, 4.5, 1.0, 2.0, 4.0,  # User 2's ratings
        ]
    })
    # fmt: on
    return ratings_df


@pytest.fixture
def sample_movies_df():
    """Create a sample movies DataFrame for testing."""
    # fmt: off
    movies_df = pd.DataFrame({
        'movieId': [
            101, 102, 103, 104, 105,
            201, 202, 203, 204,
        ],
        'title': [
            'The Dark Knight', 'The Matrix', 'Epic Movie', 'Catwoman', 'Interstellar',
            'The Lord of the Rings: The Return of the King', 'Inception', 'Disaster Movie', 'Gigli',
        ],
        'genres': [
            'Action|Crime', 'Action|Sci-Fi', 'Comedy', 'Action', 'Sci-Fi',
            'Adventure|Fantasy', 'Action|Sci-Fi', 'Comedy', 'Comedy|Romance',
        ]
    })
    # fmt: on
    return movies_df


@pytest.fixture
def dataset(sample_ratings_df, sample_movies_df):
    """Create a MovieLensDataSet instance for testing."""
    return MovieLensDataSet(
        ratings_df=sample_ratings_df,
        movies_df=sample_movies_df,
        testing_ratio=0.1,
        random_state=2,
    )


@pytest.fixture
def default_format_params():
    return {
        "context_header_version": 3,
        "sample_header_version": 2,
        "rating_listing_version": 2,
        "task_desc_version": 3,
        "answer_mark_version": 2,
    }


def test_prompt_gen_call_zero_shots(dataset, default_format_params):
    config = Config(**default_format_params, shots=0, with_context=True)
    prompt_generator = PromptGenerator(dataset, config, random_state=24)

    prompt = prompt_generator(user_id=1, movie_id=105)

    expected_prompt = (
        'User "A" has provided ratings for various movies.\n\n'
        'Some of User "A"\'s highest-rated movies:\n'
        '- "The Dark Knight": 5.0 stars.\n'
        '- "The Matrix": 4.5 stars.\n\n'
        'Some of User "A"\'s lowest-rated movies:\n'
        '- "Epic Movie": 1.0 stars.\n'
        '- "Catwoman": 2.0 stars.\n\n'
        'On a scale of 0.5 to 5.0, how would User "A" rate the movie "Interstellar"?\n\n'
        "Rating: "
    )

    assert prompt == expected_prompt


def test_prompt_gen_call_one_shot(dataset, default_format_params):
    config = Config(**default_format_params, shots=1, with_context=True)
    prompt_generator = PromptGenerator(dataset, config, random_state=24)

    prompt = prompt_generator(user_id=1, movie_id=105)

    expected_prompt = (
        'User "A" has provided ratings for various movies.\n\n'
        'Some of User "A"\'s highest-rated movies:\n'
        '- "The Lord of the Rings: The Return of the King": 5.0 stars.\n'
        '- "Inception": 4.5 stars.\n'
        '- "Interstellar": 4.0 stars.\n\n'
        'Some of User "A"\'s lowest-rated movies:\n'
        '- "Disaster Movie": 1.0 stars.\n'
        '- "Gigli": 2.0 stars.\n\n'
        'On a scale of 0.5 to 5.0, how would User "A" rate the movie "Interstellar"?\n\n'
        "Rating: 4.0\n\n\n"
        'User "B" has provided ratings for various movies.\n\n'
        'Some of User "B"\'s highest-rated movies:\n'
        '- "The Dark Knight": 5.0 stars.\n'
        '- "The Matrix": 4.5 stars.\n\n'
        'Some of User "B"\'s lowest-rated movies:\n'
        '- "Epic Movie": 1.0 stars.\n'
        '- "Catwoman": 2.0 stars.\n\n'
        'On a scale of 0.5 to 5.0, how would User "B" rate the movie "Interstellar"?\n\n'
        "Rating: "
    )

    assert prompt == expected_prompt


def test_prompt_gen_call_without_context(dataset, default_format_params):
    config = Config(**default_format_params, shots=0, with_context=False)
    prompt_generator = PromptGenerator(dataset, config, random_state=24)

    prompt = prompt_generator(user_id=1, movie_id=105)

    expected_prompt = (
        'On a scale of 0.5 to 5.0, how would User "A" rate the movie "Interstellar"?\n\n'
        "Rating: "
    )

    assert prompt == expected_prompt


def test_prompt_gen_call_with_double_range(dataset, default_format_params):
    config = Config(
        **default_format_params, shots=0, with_context=True, double_range=True
    )
    prompt_generator = PromptGenerator(dataset, config, random_state=24)

    prompt = prompt_generator(user_id=1, movie_id=105)

    expected_prompt = (
        'User "A" has provided ratings for various movies.\n\n'
        'Some of User "A"\'s highest-rated movies:\n'
        '- "The Dark Knight": 10.0 stars.\n'
        '- "The Matrix": 9.0 stars.\n\n'
        'Some of User "A"\'s lowest-rated movies:\n'
        '- "Epic Movie": 2.0 stars.\n'
        '- "Catwoman": 4.0 stars.\n\n'
        'On a scale of 1.0 to 10.0, how would User "A" rate the movie "Interstellar"?\n\n'
        "Rating: "
    )

    assert prompt == expected_prompt
