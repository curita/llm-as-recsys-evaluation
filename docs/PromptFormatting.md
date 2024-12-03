## Formatting Template Options

### Context Header Versions (`--context-header-version`)

| Version | Template |
|---------|----------|
| 1 | `""` (Empty string) |
| 2 | `"Here are some movie ratings from {user}."` |
| 3 | `"{user} has provided ratings for various movies."` |
| 4 | `"This is a selection of {user}'s history of movie ratings."` |
| 5 | `"Here are some of the highest and lowest ratings that {user} has given to movies."` |

### Sample Header Versions (`--sample-header-version`)

| Version | Template |
|---------|----------|
| 1 | `""` (Empty string) |
| 2 | `"Some of {user}'s {kind}-rated movies:"` |
| 3 | `"Some {kind}-rated movies by {user} include:"` |

### Rating Listing Versions (`--rating-listing-version`)

| Version | Template |
|---------|----------|
| 1 | `"{user} rated with {rating} stars the movie {movie}."` |
| 2 | `"- {movie}: {rating} stars."` |
| 3 | `"* {movie} - {rating} stars."` |
| 4 | `"* {movie} ({rating} stars)."` |

### Task Description Versions (`--task-desc-version`)

| Version | Template |
|---------|----------|
| 1 | `"On a scale of {values}, how would {user} rate the movie {movie}?"` |
| 2 | `"How would {user} rate the movie {movie} on a scale of {values}?"` |
| 3 | `"On a scale of {min_value} to {max_value}, how would {user} rate the movie {movie}?"` |
| 4 | `"How would {user} rate the movie {movie}?\nOPTIONS:{bulleted_values}"` |
| 5 | `"How would {user} rate the movie {movie}?"` |
| 6 | `"Predict {user}'s likely rating for the movie {movie} on a scale from {min_value} to {max_value}."` |
| 7 | `"{user} hasn't seen the movie {movie} yet. Predict how {user} will likely rate the movie on a scale from {min_value} to {max_value}."` |
| 8 | `"How would {user} rate the movie {movie} on a scale of {min_value} to {max_value}?"` |

### Answer Mark Versions (`--answer-mark-version`)

| Version | Template |
|---------|----------|
| 1 | `""` (Empty string) |
| 2 | `"Rating: "` |
| 3 | `"Estimated rating: "` |
| 4 | `"Predicted rating: "` |

### Example of Using Different Formatting Versions

```bash
python scripts/run.py \
    --model google/flan-t5-small \
    --context-header-version 3 \
    --sample-header-version 2 \
    --rating-listing-version 3 \
    --task-desc-version 8 \
    --answer-mark-version 3
```

This command would generate prompts with:

- A context header: `"{user} has provided ratings for various movies."`
- A sample header: `"Some of {user}'s {kind}-rated movies:"`
- Rating listings: `"* {movie} - {rating} stars."`
- Task description: `"How would {user} rate the movie {movie} on a scale of {min_value} to {max_value}?"`
- Answer mark: `"Estimated rating: "`

Which should generate a prompt like this one:

```text
User "A" has provided ratings for various movies.

Some of User "A"'s highest-rated movies:
* "The Dark Knight" - 5.0 stars.
* "The Matrix" - 4.5 stars.

Some of User "A"'s lowest-rated movies:
* "Epic Movie" - 1.0 stars.
* "Catwoman" - 2.0 stars.

How would User "A" rate the movie "Interstellar" on a scale of 0.5 to 5.0?

Estimated rating:
```
