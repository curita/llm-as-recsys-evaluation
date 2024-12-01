from dataclasses import dataclass


@dataclass
class Config:
    # Experiment
    runs: int = 1
    initial_run_seed: int = 0
    # Dataset
    testing_ratio: float = 0.2
    popularity: list = None
    training_popularity: list = None
    # Pipeline
    model: str = "google/flan-t5-base"
    precision: str = "default"
    use_flash_attention_2: bool = False
    batch_size: int = 8
    temperature: float = 0.0
    # In-Context Learning
    shots: int = 0
    likes_count: int = 10
    dislikes_count: int = 10
    double_range: bool = False
    with_context: bool = True
    with_genre: bool = False
    with_global_rating_in_context: bool = False
    with_global_rating_in_task: bool = False
    # Prompt Formatting
    rating_listing_version: int = 1
    context_header_version: int = 1
    sample_header_version: int = 1
    task_desc_version: int = 1
    answer_mark_version: int = 1
    numeric_user_identifier: bool = False
    likes_first: bool = True
    keep_trailing_zeroes: bool = True
