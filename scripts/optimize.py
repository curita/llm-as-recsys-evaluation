import logging
from functools import partial

import optuna
from optuna.trial import TrialState
import click
from transformers.pipelines.base import Pipeline

from trabajo import load_pipeline, run_experiment, get_default_task


def objective(trial: optuna.Trial, include_empty_answer_mark: bool, metric: str, predictors: list[Pipeline], **params):
    task_desc_version = trial.suggest_categorical(
        "task_desc_version", list(range(1, 8 + 1))
    )
    likes_first = trial.suggest_categorical("likes_first", [True, False])
    keep_trailing_zeroes = trial.suggest_categorical(
        "keep_trailing_zeroes", [True, False]
    )
    sample_header_version = trial.suggest_categorical(
        "sample_header_version", list(range(1, 3 + 1))
    )
    rating_listing_version = trial.suggest_categorical(
        "rating_listing_version", list(range(1, 4 + 1))
    )
    context_header_version = trial.suggest_categorical(
        "context_header_version", list(range(1, 5 + 1))
    )

    answer_mark_version_range = range(1, 4 + 1) if include_empty_answer_mark else range(2, 4 + 1)
    answer_mark_version = trial.suggest_categorical(
        "answer_mark_version", list(answer_mark_version_range)
    )
    numeric_user_identifier = trial.suggest_categorical(
        "numeric_user_identifier", [True, False]
    )

    for t in trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if t.params == trial.params:
            # return t.value  # Return the previous value without re-evaluating it.
            raise optuna.TrialPruned("Duplicate parameter set")

    rmses = []
    f1s = []
    for predictor in predictors:
        rmse, f1 = run_experiment(
            task_desc_version=task_desc_version,
            likes_first=likes_first,
            keep_trailing_zeroes=keep_trailing_zeroes,
            sample_header_version=sample_header_version,
            rating_listing_version=rating_listing_version,
            context_header_version=context_header_version,
            answer_mark_version=answer_mark_version,
            numeric_user_identifier=numeric_user_identifier,
            predictor=predictor,
            model=predictor.model.name_or_path,
            task=predictor.task,
            **params,
        )
        rmses.append(rmse)
        f1s.append(f1)

    if metric == "f1":
        return f1s
    elif metric == "rmse":
        return rmses


def print_best_callback(study, trial):

    best_value = None
    best_trial = None
    for trial in study.best_trials:
        value = sum(trial.values) / len(trial.values)
        if not best_value or best_value < value:
            best_value = value
            best_trial = trial

    print(f"Best Trial: {best_trial.number}. Best values: {best_trial.values}. Best params: {best_trial.params}")


@click.command
@click.option("--testing-ratio", default=0.02, type=float)
@click.option("--runs", default=5, type=int)
@click.option("--model", "models", type=str, multiple=True)
@click.option(
    "--precision", default="default", type=click.Choice(["default", "16", "8", "4"])
)
@click.option("--shots", default=0, type=int)
@click.option("--study-name", default=None, type=str)
@click.option("--trials", default=None, type=int)
@click.option("--timeout", default=None, type=int)
@click.option("--include-empty-answer-mark/--exclude-empty-answer-mark", default=True)
@click.option("--metric", default="f1", type=click.Choice(["f1", "rmse"]))
def main(
    testing_ratio, runs, models, precision, shots, study_name, trials, timeout, include_empty_answer_mark, metric
):
    if metric == "f1":
        direction = "maximize"
    elif metric == "rmse":
        direction = "minimize"

    study = optuna.create_study(
        directions=[direction] * len(models),
        study_name=study_name,
        storage="sqlite:///optimize.db",
        load_if_exists=True,
    )
    predictors = []
    for model in models:
        predictor = load_pipeline(task=get_default_task(model=model), precision=precision, model=model)
        predictors.append(predictor)

    partial_objective = partial(
        objective,
        include_empty_answer_mark=include_empty_answer_mark,
        metric=metric,
        predictors=predictors,
        testing_ratio=testing_ratio,
        runs=runs,
        precision=precision,
        shots=shots,
    )
    study.optimize(
        partial_objective,
        n_trials=trials,
        timeout=timeout,
        callbacks=[print_best_callback],
        show_progress_bar=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="[%(asctime)s] " + logging.BASIC_FORMAT
    )
    main()
