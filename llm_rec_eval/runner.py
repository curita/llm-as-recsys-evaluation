import logging

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers.pipelines.base import Pipeline

from llm_rec_eval.dataset import MovieLensDataSet, PyTorchListDataset
from llm_rec_eval.metrics import AggregatedStats, report_metrics
from llm_rec_eval.pipeline import get_inference_kwargs
from llm_rec_eval.prompts import PromptGenerator
from llm_rec_eval.save import save_results
from llm_rec_eval.parse import Parser

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, predictor: Pipeline, stats: AggregatedStats, **params):
        self.predictor = predictor
        self.stats = stats
        self.parser = Parser(double_range=params["double_range"])
        self.params = params

    def run(self):
        for x in range(self.params["runs"]):
            self.run_single_experiment(x)
        self.stats.report()

    def run_single_experiment(self, run_index):
        run_params = self.prepare_run_params(run_index)
        dataset = self.create_dataset(run_params)
        prompts = self.generate_prompts(dataset, run_params)
        outputs = self.run_model(prompts)
        predictions, unpredicted_indexes = self.parse_outputs(outputs, prompts)
        truth = [row.rating for row in dataset.testing_df.itertuples()]

        save_results(prompts, outputs, predictions, dataset, run_params)
        self.remove_unpredicted_items(truth, predictions, unpredicted_indexes)
        self.report_metrics(truth, predictions)

    def prepare_run_params(self, run_index: int) -> dict:
        run_params = self.params.copy()
        run_seed = self.params["initial_run_seed"] + run_index
        run_params["run_seed"] = run_seed

        logger.info(f"Run {run_seed=}.")
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        return run_params

    def create_dataset(self, run_params):
        logger.info("Creating dataset...")
        return MovieLensDataSet(
            testing_ratio=self.params["testing_ratio"],
            training_popularity=self.params["training_popularity"],
            popularity=self.params["popularity"],
        )

    def generate_prompts(self, dataset, run_params):
        logger.info("Generating prompts...")
        prompt_generator = PromptGenerator(dataset=dataset, **run_params)
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
            model=self.params["model"], temperature=self.params["temperature"]
        )
        return [
            p[0]["generated_text"]
            for p in tqdm(
                self.predictor(
                    PyTorchListDataset(prompts),
                    batch_size=self.params["batch_size"],
                    **inference_kwargs,
                ),
                total=len(prompts),
            )
        ]

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
            model=self.params["model"], temperature=self.params["temperature"]
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

    def report_metrics(self, truth, predictions):
        if not predictions:
            logger.info("All predictions are unknown, stopping experiment...")
            return

        stats = report_metrics(truth, predictions)
        self.stats.add_rmse(stats.rmse)
        self.stats.add_precision(stats.precision)
        self.stats.add_recall(stats.recall)
        self.stats.add_f1(stats.f1)
        self.stats.update_value_counts(stats.value_counts)
