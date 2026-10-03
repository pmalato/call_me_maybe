from llm_sdk.llm_sdk import Small_LLM_Model
from argparse import ArgumentParser, Namespace
from pydantic import TypeAdapter
from .models import FunctionValidator, PromptValidator
from json import load


class Parser():
    def __init__(self, llm: Small_LLM_Model):
        self._llm: Small_LLM_Model = llm
        self._tuple: tuple[list[FunctionValidator], list[PromptValidator]]
        self._tuple = self.read_files()
        self._vocabs: list[str] = []
        self.read_vocabs()

    def parse_args(self) -> Namespace:
        parser: ArgumentParser = ArgumentParser()
        parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json")
        parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json")
        parser.add_argument(
            "--output",
            default="data/output/function_calling_results.json")
        return parser.parse_args()

    def read_files(self) -> tuple[
            list[FunctionValidator],
            list[PromptValidator]]:
        parser = self.parse_args()
        with open(parser.functions_definition, "r") as func_file:
            adapter1 = TypeAdapter(list[FunctionValidator])
            functions: list[FunctionValidator] = adapter1.validate_json(
                func_file.read())
        with open(parser.input, "r") as prompt_file:
            adapter2: TypeAdapter = TypeAdapter(list[PromptValidator])
            prompts: list[PromptValidator] = adapter2.validate_json(
                prompt_file.read())
        return functions, prompts

    def read_vocabs(self) -> None:
        path: str = self._llm.get_path_to_vocab_file()
        with open(path, "r") as file:
            vocabs: dict[str, int] = load(file)
        for x in vocabs:
            self._vocabs += [x]

    def get_tuple(self) -> tuple[
            list[FunctionValidator],
            list[PromptValidator]]:
        return self._tuple

    def get_vocabs(self) -> list[str]:
        return self._vocabs
