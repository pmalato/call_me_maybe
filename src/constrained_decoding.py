from .models import FunctionValidator, PromptValidator
from .parser import Parser


class ConstrainedDecoder():
    def __init__(self, parser: Parser) -> None:
        self._parser: Parser = parser
        self._functions: list[FunctionValidator] = parser.get_tuple()[0]
        self._prompts: list[PromptValidator] = parser.get_tuple()[1]
        self._possible_tokens: list[int]

    def set_possible_tokens(self, tokens: list[int]) -> None:
        self._possible_tokens = []

    def find_valid_tokens(self) -> None:
        ...

    def make_invalid_logits(self) -> None:
        ...

    def sample_valid_tokens(self) -> None:
        ...
