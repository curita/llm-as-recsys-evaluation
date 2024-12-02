# Evaluation of LLMs as Recommender Systems

## Installation

```sh
pip install -e .
```

## (Optional) Login to HuggingFace

Only needed if gated models will be employed.

```sh
huggingface-cli login
```

## Download Dataset

```sh
python scripts/download_dataset.py
```

## (Optional) Download Model

Models will be downloaded automatically when used by other scripts otherwise.

```sh
python scripts/download_model.py --model google/flan-t5-small
```

## Evaluate

```sh
python scripts/run.py --testing-ratio 0.001 --model google/flan-t5-small --shots 1
```

## Optimize Format

```sh
python scripts/optimize_format.py --timeout 60 --runs 1 --testing-ratio 0.001 --exclude-empty-answer-mark --model google/flan-t5-small
```
