import re
import logging

from llm_rec_eval.constants import POSSIBLE_VALUES

logger = logging.getLogger(__name__)


class Parser:
    def __init__(self, double_range: bool) -> None:
        self.double_range = double_range

    def parse(self, output: str) -> float:
        original_output = output

        try:
            output = self.clean_output(output)
            value = self.extract_value(output)
            self.validate_value(value)
            value = self.adjust_value(value)
            return value

        except (ValueError, AssertionError) as err:
            msg = f"Can't parse: {original_output!r}"
            logger.exception(msg)
            raise ValueError(msg) from err

    def clean_output(self, output: str) -> str:
        # Strip leading puntuation, spaces or emojis
        return re.sub(r"^[^\d\w]+", "", output)

    def extract_value(self, output: str) -> float:
        match = re.findall(r"^(\d+(?:\.\d+)?)", output)
        if not match:
            raise ValueError("No numerical value found in the output.")
        return float(match[0])

    def validate_value(self, value: float) -> None:
        min_value, max_value = POSSIBLE_VALUES[0], POSSIBLE_VALUES[-1]
        if self.double_range:
            min_value *= 2
            max_value *= 2

        assert min_value <= value <= max_value, "Value out of range."

    def adjust_value(self, value: float) -> float:
        if self.double_range:
            value /= 2
        return value
