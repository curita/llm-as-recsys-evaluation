# Evaluation of LLMs as Recommender Systems

## Installation

```sh
pip install -e .
```

## Evaluate

```sh
python scripts/run.py --testing-ratio 0.001 --model google/flan-t5-small --shots 1
```

## Optimize Format

```sh
python scripts/optimize_format.py --timeout 60 --runs 1 --testing-ratio 0.001 --exclude-empty-answer-mark --model google/flan-t5-small
```
