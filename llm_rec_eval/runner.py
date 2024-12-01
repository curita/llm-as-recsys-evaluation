from dataclasses import asdict
import logging

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers.pipelines.base import Pipeline

from llm_rec_eval.config import Config
from llm_rec_eval.dataset import MovieLensDataSet, PyTorchListDataset
from llm_rec_eval.metrics import AggregatedStats, report_metrics
from llm_rec_eval.pipeline import get_inference_kwargs
from llm_rec_eval.prompts import PromptGenerator
from llm_rec_eval.save import save_results
from llm_rec_eval.parse import Parser

logger = logging.getLogger(__name__)


class StopExperiment(Exception):
    pass


class ExperimentRunner:
    def __init__(self, predictor: Pipeline, config: Config):
        self.predictor = predictor
        self.stats = AggregatedStats()
        self.parser = Parser(double_range=config.double_range)
        self.config = config

    def run(self):
        for x in range(self.config.runs):
            self.run_single_experiment(x)
        self.stats.report()

    def run_single_experiment(self, run_index):
        run_seed = self.set_run_seed(run_index)
        dataset = self.create_dataset()
        prompts = self.generate_prompts(dataset)
        outputs = self.run_model(prompts)
        predictions, unpredicted_indexes = self.parse_outputs(outputs, prompts)
        truth = [row.rating for row in dataset.testing_df.itertuples()]

        save_results(prompts, outputs, predictions, dataset, run_seed, self.config)
        self.remove_unpredicted_items(truth, predictions, unpredicted_indexes)
        self.report_metrics(truth, predictions)

    def set_run_seed(self, run_index: int) -> int:
        run_seed = self.config.initial_run_seed + run_index

        logger.info(f"Run {run_seed=}.")
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        return run_seed

    def create_dataset(self):
        logger.info("Creating dataset...")
        return MovieLensDataSet(
            testing_ratio=self.config.testing_ratio,
            training_popularity=self.config.training_popularity,
            popularity=self.config.popularity,
        )

    def generate_prompts(self, dataset):
        logger.info("Generating prompts...")
        prompt_generator = PromptGenerator(dataset=dataset, **asdict(self.config))
        prompts = [
            prompt_generator(user_id=row.userId, movie_id=row.movieId)
            for row in dataset.testing_df.itertuples()
        ]
        logger.info(f"Prompt Example:\n{prompts[0]}")
        self.stats.increment_prompts_count(len(prompts))
        return prompts

    def run_model(self, prompts):
        logger.info("Running model...")
        inference_kwargs = get_inference_kwargs(
            model=self.config.model, temperature=self.config.temperature
        )
        self.predictor.over_token_limit = 0
        outputs = [
            p[0]["generated_text"]
            for p in tqdm(
                self.predictor(
                    PyTorchListDataset(prompts),
                    batch_size=self.config.batch_size,
                    **inference_kwargs,
                ),
                total=len(prompts),
            )
        ]
        self.stats.increment_over_token_limit(self.predictor.over_token_limit)
        return outputs

    def parse_outputs(self, outputs, prompts):
        logger.info("Parsing outputs...")
        retried_prompts = 0
        retries = 0
        max_retries = 3
        predictions = []
        unpredicted_indexes = set()
        for index, out in enumerate(outputs):
            try:
                pred = self.parser.parse(out)
            except ValueError:
                try:
                    retried_prompts += 1
                    (
                        retried_output,
                        retried_prediction,
                        retried_attempts,
                    ) = self.retry_inference(
                        prompt=prompts[index],
                        max_retries=max_retries,
                    )
                    retries += retried_attempts
                    outputs[index] = retried_output
                    pred = retried_prediction
                except ValueError:
                    retries += max_retries
                    unpredicted_indexes.add(index)
                    pred = "N/A"
            predictions.append(pred)

        logger.info(f"Retried prompts: {retried_prompts}")
        logger.info(f"Retries: {retries}")
        self.stats.increment_retried_prompts(retried_prompts)
        self.stats.increment_retries(retries)
        return predictions, unpredicted_indexes

    def retry_inference(self, prompt, max_retries=3):
        inference_kwargs = get_inference_kwargs(
            model=self.config.model, temperature=self.config.temperature
        )
        inference_kwargs["do_sample"] = True

        for attempt in range(1, max_retries + 1):
            logger.info(f"Retrying, {attempt=}")
            output = self.predictor(prompt, **inference_kwargs)[0]["generated_text"]
            try:
                pred = self.parser.parse(output)
            except ValueError:
                continue
            else:
                return output, pred, attempt

        raise ValueError("Couldn't get prediction")

    def remove_unpredicted_items(self, truth, predictions, unpredicted_indexes):
        logger.info("Removing unpredicted items...")
        truth[:] = [
            value for i, value in enumerate(truth) if i not in unpredicted_indexes
        ]
        predictions[:] = [
            value for i, value in enumerate(predictions) if i not in unpredicted_indexes
        ]
        logger.info(f"Unknown predictions: {len(unpredicted_indexes)}")
        self.stats.increment_unpredicted(len(unpredicted_indexes))

        if not predictions:
            logger.info("All predictions are unknown, stopping experiment...")
            raise StopExperiment("All predictions are unknown")

    def report_metrics(self, truth, predictions):
        stats = report_metrics(truth, predictions)
        self.stats.add_rmse(stats.rmse)
        self.stats.add_precision(stats.precision)
        self.stats.add_recall(stats.recall)
        self.stats.add_f1(stats.f1)
        self.stats.update_value_counts(stats.value_counts)
