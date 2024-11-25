import re
import logging

from llm_rec_eval.constants import POSSIBLE_VALUES

logger = logging.getLogger(__name__)


def parse_model_output(output: str, double_range: bool) -> float:
    original_output = output

    try:
        output = re.sub(
            r"^[^\d\w]+", "", output
        )  # Strip leading puntuation, spaces or emojis
        value = float(re.findall(r"^(\d+(?:\.\d+)?)", output)[0])

        min_value, max_value = POSSIBLE_VALUES[0], POSSIBLE_VALUES[-1]
        if double_range:
            min_value *= 2
            max_value *= 2

        assert value >= min_value
        assert value <= max_value

        if double_range:
            value /= 2

        return value

    except Exception as err:
        msg = f"Can't parse: {original_output!r}"
        logger.exception(msg)
        raise ValueError(msg) from err
