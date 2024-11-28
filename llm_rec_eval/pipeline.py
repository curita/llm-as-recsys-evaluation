import logging

import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import pipeline, BitsAndBytesConfig
from transformers.pipelines.base import Pipeline

from llm_rec_eval.metrics import AggregatedStats


logger = logging.getLogger(__name__)


def get_default_task(model: str) -> str:
    if "t5" in model.lower():
        return "text2text-generation"
    return "text-generation"


@retry(stop=stop_after_attempt(5), wait=wait_exponential())
def get_pipeline(task: str, model: str, pipeline_kwargs: dict) -> Pipeline:
    return pipeline(task, model=model, token=True, **pipeline_kwargs)


def load_pipeline(
    stats: AggregatedStats,
    precision: str = "default",
    use_flash_attention_2: bool = False,
    model: str = "google/flan-t5-base",
) -> Pipeline:
    task = get_default_task(model)
    logger.info(f"Initializing {task} pipeline...")

    pipeline_kwargs = build_pipeline_kwargs(precision, use_flash_attention_2)
    predictor = get_pipeline(task=task, model=model, pipeline_kwargs=pipeline_kwargs)
    patch_preprocess(predictor, stats)
    configure_padding(predictor)
    return predictor


def build_pipeline_kwargs(precision: str, use_flash_attention_2: bool) -> dict:
    pipeline_kwargs = {"model_kwargs": {"device_map": "auto"}}
    if precision == "16":
        pipeline_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    elif precision in ["8", "4"]:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=(precision == "8"),
            load_in_4bit=(precision == "4"),
        )
        pipeline_kwargs["model_kwargs"] = {"quantization_config": quantization_config}

    if use_flash_attention_2:
        pipeline_kwargs["model_kwargs"] = {
            "torch_dtype": torch.float16,
            "attn_implementation": "flash_attention_2",
        }

    return pipeline_kwargs


def patch_preprocess(predictor: Pipeline, stats: AggregatedStats) -> None:
    preprocess_method_name = (
        "_parse_and_tokenize"
        if hasattr(predictor, "_parse_and_tokenize")
        else "preprocess"
    )
    original_preprocess = getattr(predictor, preprocess_method_name)
    max_token_length = get_max_token_length(predictor)
    logger.info(f"Model context limit: {max_token_length}")

    def _patched_preprocess(*args, **kwargs):
        inputs = original_preprocess(*args, **kwargs)
        # NOTE: Only valid for PyTorch tensors
        input_length = inputs["input_ids"].shape[-1]

        if input_length > max_token_length:
            stats.increment_over_token_limit()

        return inputs

    setattr(predictor, preprocess_method_name, _patched_preprocess)


def get_max_token_length(predictor: Pipeline) -> int:
    max_token_length = getattr(predictor.model.config, "max_position_embeddings", None)
    if not max_token_length and "t5" in predictor.model.name_or_path.lower():
        # https://huggingface.co/google/flan-t5-xxl/discussions/41#65c3c3706b793334ef78dffc
        max_token_length = 1024
    return max_token_length


def configure_padding(predictor: Pipeline) -> None:
    model_config = predictor.model.config
    tokenizer = predictor.tokenizer

    if tokenizer.pad_token_id is None:
        if model_config.pad_token_id is not None:
            tokenizer.pad_token_id = model_config.pad_token_id
        if "llama-2" in predictor.model.name_or_path.lower():
            tokenizer.pad_token = "[PAD]"
            tokenizer.padding_side = "left"
        else:
            tokenizer.pad_token_id = model_config.eos_token_id

    model_config.pad_token_id = tokenizer.pad_token_id


def get_inference_kwargs(model: str, temperature: float) -> dict:
    inference_kwargs = {}
    if temperature == 0.0:
        inference_kwargs["do_sample"] = False
    else:
        inference_kwargs["do_sample"] = True
        inference_kwargs["temperature"] = temperature

    inference_kwargs["max_new_tokens"] = 20

    if get_default_task(model=model) == "text-generation":
        inference_kwargs["return_full_text"] = False

    return inference_kwargs
