from dataclasses import dataclass
import logging
from collections import defaultdict
from typing import Any

import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    root_mean_squared_error,
)

from llm_rec_eval.constants import POSSIBLE_VALUES

logger = logging.getLogger(__name__)


@dataclass
class Stats:
    rmse: float
    precision: float
    recall: float
    f1: float
    value_counts: dict[float, int]


def get_distribution(value_counts: dict[Any, int]) -> dict[Any, float]:
    total = sum(value_counts.values())
    return {
        rating: round((count * 100 / total), 2)
        for rating, count in sorted(value_counts.items())
    }


def report_metrics(
    truth: list[float], predictions: list[float], threshold: float = 4.0
) -> Stats | None:
    if not predictions:
        return

    logger.info("Reporting metrics...")
    rmse = root_mean_squared_error(truth, predictions)
    logger.info(f"RMSE: {rmse}")

    predictions_df = pd.DataFrame(
        {
            "Truth": truth,
            "Prediction": predictions,
        }
    )

    logger.info(
        f"Classification report:\n{classification_report(predictions_df['Truth'] >= threshold, predictions_df['Prediction'] >= threshold)}"
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        predictions_df["Truth"] >= threshold,
        predictions_df["Prediction"] >= threshold,
        average="macro",
        zero_division=0.0,
    )

    value_counts = defaultdict(int, {v: 0 for v in POSSIBLE_VALUES})
    for p in predictions:
        value_counts[p] += 1
    distribution = get_distribution(value_counts)
    logger.info(f"Distribution: {distribution}")

    return Stats(
        rmse=rmse,
        precision=precision,
        recall=recall,
        f1=f1,
        value_counts=value_counts,
    )


class AggregatedStats:
    def __init__(self):
        self.rmse = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.value_counts = dict()
        self.prompts_count = 0
        self.retried_prompts = 0
        self.retries = 0
        self.unpredicted = 0
        self.over_token_limit = 0

    def add_rmse(self, value):
        self.rmse.append(value)

    def add_precision(self, value):
        self.precision.append(value)

    def add_recall(self, value):
        self.recall.append(value)

    def add_f1(self, value):
        self.f1.append(value)

    def update_value_counts(self, value_counts):
        for value, count in value_counts.items():
            self.value_counts[value] = self.value_counts.get(value, 0) + count

    def increment_prompts_count(self, count):
        self.prompts_count += count

    def increment_retried_prompts(self, count):
        self.retried_prompts += count

    def increment_retries(self, count):
        self.retries += count

    def increment_unpredicted(self, count):
        self.unpredicted += count

    def increment_over_token_limit(self, count):
        self.over_token_limit += count

    def report(self):
        logger.info("Aggregated stats.")
        self._report_stat("RMSE", self.rmse)
        self._report_stat("Precision", self.precision)
        self._report_stat("Recall", self.recall)
        self._report_stat("F1", self.f1)
        self._report_distribution()
        self._report_prompt_counts("Retried Prompts", self.retried_prompts)
        logger.info(f"Aggregated Retries: {self.retries}")
        self._report_prompt_counts("Unknown Prompts", self.unpredicted)
        self._report_prompt_counts("Over Limit Prompts", self.over_token_limit)

    def _report_stat(self, name: str, values: list[float]):
        series = pd.Series(values)
        logger.info(
            f"Aggregated {name}. Median: {series.median()}. STD: {series.std(ddof=1)}"
        )

    def _report_prompt_counts(self, name, count):
        logger.info(
            f"Aggregated {name}: {count} ({round(count * 100 / self.prompts_count, 2)}%)"
        )

    def _report_distribution(self):
        distribution = get_distribution(self.value_counts)
        logger.info(f"Aggregated Distribution: {distribution}")
