import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)


class AggregatedStats:
    def __init__(self, possible_values):
        self.rmse = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.value_counts = defaultdict(int, {v: 0 for v in possible_values})
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

    def add_value_count(self, value):
        self.value_counts[value] += 1

    def increment_prompts_count(self, count):
        self.prompts_count += count

    def increment_retried_prompts(self, count):
        self.retried_prompts += count

    def increment_retries(self, count):
        self.retries += count

    def increment_unpredicted(self, count):
        self.unpredicted += count

    def increment_over_token_limit(self):
        self.over_token_limit += 1

    def report(self):
        logger.info("Aggregated stats.")
        rmse_s = pd.Series(self.rmse)
        logger.info(
            f"Aggregated RMSE. Median: {rmse_s.median()}. STD: {rmse_s.std(ddof=1)}"
        )

        precision_s = pd.Series(self.precision)
        logger.info(
            f"Aggregated Precision. Median: {precision_s.median()}. STD: {precision_s.std(ddof=1)}"
        )

        recall_s = pd.Series(self.recall)
        logger.info(
            f"Aggregated Recall. Median: {recall_s.median()}. STD: {recall_s.std(ddof=1)}"
        )

        f1_s = pd.Series(self.f1)
        logger.info(f"Aggregated F1. Median: {f1_s.median()}. STD: {f1_s.std(ddof=1)}")

        total = sum(self.value_counts.values())
        aggregated_distribution = {
            rating: round((count * 100 / total), 2)
            for rating, count in sorted(self.value_counts.items())
        }
        logger.info(f"Aggregated Distribution: {aggregated_distribution}")

        logger.info(
            f"Aggregated Retried Prompts: {self.retried_prompts} ({round(self.retried_prompts * 100 / self.prompts_count, 2)}%)"
        )
        logger.info(f"Aggregated Retries: {self.retries}")
        logger.info(
            f"Aggregated Unknown Predictions: {self.unpredicted} ({round(self.unpredicted * 100 / self.prompts_count, 2)}%)"
        )
        logger.info(
            f"Aggregated Over Limit Prompts: {self.over_token_limit} ({round(self.over_token_limit * 100 / self.prompts_count, 2)}%)"
        )
