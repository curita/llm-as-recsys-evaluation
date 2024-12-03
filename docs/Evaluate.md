# run.py: LLMs Evaluation

## Overview

This script provides a framework for evaluating Large Language Models (LLMs) in the context of movie rating prediction. It allows researchers and developers to systematically assess how well LLMs can predict user movie ratings based on user history using the MovieLens dataset.

## Key Features

- Cross-validation across multiple experimental runs
- Configurable dataset filtering by movie popularity
- Flexible prompt generation with multiple in-context learning and formatting options
- Detailed metrics reporting and result tracking
- Support for various model configurations and inference settings

## Prerequisites

- Python 3.10
- Dependencies:
  - pandas
  - numpy
  - tqdm
  - huggingface_hub
  - transformers
  - torch
  - scikit-learn
  - click
  - tenacity
  - accelerate
  - bitsandbytes

## Installation

```bash
pip install -e .
```

## Usage

```bash
python scripts/run.py [OPTIONS]
```

## Command Line Parameters

### Experiment Run

- `--runs INTEGER`: Number of experiment iterations for cross-validation. *Default: `1`*
- `--initial-run-seed INTEGER`: Initial random seed for reproducibility. *Default: `0`*

### Dataset Filtering

- `--testing-ratio FLOAT`: Proportion of dataset reserved for testing (between 0 and 1). *Default: `0.2`*
- `--popularity [rare|unfrequent|normal|very_frequent]`: Filter dataset by selected movie popularity categories. *Default: `None`*
- `--training-popularity [rare|unfrequent|normal|very_frequent]`: Filter training dataset by selected movie popularity categories. *Default: `None`*

### Model and Inference

- `--model TEXT`: HuggingFace model identifier or local model path to be evaluated. *Default: `google/flan-t5-base`*
- `--precision [default|16|8|4]`: Numerical precision to load the model weights. *Default: `default`*
- `--use-flash-attention-2`: Enable Flash Attention 2. *Default: `False`*
- `--batch-size INTEGER`: Inference batch size determining how many prompts are processed at the same time. *Default: `8`*
- `--temperature FLOAT`: Sampling temperature controlling randomness of model outputs. *Default: `0.0`*

### Prompt Generation

#### In-Context Learning

- `--shots INTEGER`: Number of in-context learning examples. *Default: `0`*
- `--likes-count INTEGER`: Number of top-rated movies to use in user history. *Default: `10`*
- `--dislikes-count INTEGER`: Number of bottom-rated movies to use in user history. *Default: `10`*
- `--double-range/--single-range`: Rating scale (1-10 or 0.5-5). *Default: `--single-range`*
- `--with-context/--without-context`: Include/exclude user history in prompts. *Default: `--with-context`*
- `--with-genre/--without-genre`: Include/exclude movie genres in prompts. *Default: `--without-genre`*
- `--with-global-rating-in-context/--without-global-rating-in-context`: Include/exclude global ratings in user history movies. *Default: `--without-global-rating-in-context`*
- `--with-global-rating-in-task/--without-global-rating-in-task`: Include/exclude global rating in target movie. *Default: `--without-global-rating-in-task`*

#### Prompt Formatting

- `--context-header-version INTEGER`: Formatting version of user history header. *Default: `1`*
- `--sample-header-version INTEGER`: Formatting version for sample header. *Default: `1`*
- `--rating-listing-version INTEGER`: Formatting version for rating listing. *Default: `1`*
- `--task-desc-version INTEGER`: Formatting version for task description. *Default: `1`*
- `--answer-mark-version INTEGER`: Formatting version for answer mark. *Default: `1`*
- `--numeric-user-identifier/--alphabetic-user-identifier`: Use numeric or alphabetic user identifiers. *Default: `--alphabetic-user-identifier`*
- `--likes-first/--dislikes-first`: List top-rated or bottom-rated movies in user histories. *Default: `--likes-first`*
- `--keep-trailing-zeroes/--strip-trailing-zeroes`: Preserve/remove trailing zeroes in ratings. *Default: `--keep-trailing-zeroes`*

### Misc

- `--help`: Show available options

## Reported Metrics

The script logs performance metrics for individual and aggregated runs:

1. Root Mean Square Error (RMSE)
2. Precision
3. Recall
4. F1 Score
5. Rating Distribution
6. Experiment Statistics
   - Retried Prompts and Total Retries
   - Unknown Predictions
   - Over Token Limit Prompts

## Example Commands

1. Basic Evaluation:

```bash
python scripts/run.py --model google/flan-t5-small
```

2. Advanced Evaluation:

```bash
python scripts/run.py \
    --model google/flan-t5-small \
    --runs 5 \
    --shots 3 \
    --popularity common --popularity frequent \
    --with-genre
```

3. Low Precision Inference:

```bash
python scripts/run.py \
    --model google/flan-t5-small \
    --precision 8 \
    --use-flash-attention-2
```

## Results

Experiment results are saved in CSV files at the end of the run, storing the following data:

- Generated Prompts
- Model Outputs
- Parsed Predictions
- Ground-Truth Values
- Experiment Configuration
