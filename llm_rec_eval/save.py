import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
    "B": "batch_size",
    "PR": "precision",
    "FL": "use_flash_attention_2",
}

CSV_FIELDNAMES = [
    "Prompt",
    "Movie",
    "MovieID",
    "UserID",
    "Output",
    "Prediction",
    "Truth",
    "Parameters",
]


def save_results(prompts, outputs, predictions, dataset, run_params):
    logger.info("Dumping results...")
    output_folder = create_output_folder(run_params)
    output_file = output_folder / "results.csv"

    logger.info(f"Path: {output_file}")

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
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


def create_output_folder(run_params: dict[str, Any]) -> Path:
    folder_name = f"experiment_{'_'.join(k + '=' + str(run_params[v]) for k, v in FILENAME_PARAMETERS.items())}".replace(
        "/", ":"
    )
    output_folder = Path("results") / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder
