from llm_sdk.llm_sdk import Small_LLM_Model
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pydantic import TypeAdapter
from typing import Any
from .models import FunctionValidator, PromptValidator, OutputValidator
from json import load


class Parser():
    """Handles program arguments, input files, vocabulary and output file.

    Attributes:
        _llm: The LLM wrapper, used to locate the vocabulary file.
        _args: Parsed command line arguments.
        _tuple: The validated (functions, prompts) pair read from disk.
        _vocabs: Tokens of the vocabulary, in file order.
    """
    def __init__(self, llm: Small_LLM_Model):
        """Parse arguments, then load input files and the vocabulary.

        Args:
            llm: The LLM wrapper providing the path to the vocab file.
        """
        self._llm: Small_LLM_Model = llm
        self._args: Namespace = self.parse_args()
        self._tuple: tuple[list[FunctionValidator], list[PromptValidator]]
        self._tuple = self.read_files()
        self._vocabs: list[str] = []
        self.read_vocabs()

    def parse_args(self) -> Namespace:
        """Parse the command line arguments.

        Returns:
            A namespace with `functions_definition`, `input` and `output`
            paths (each with a default under data/).
        """
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
        """Read and validate the function definitions and the prompts.

        Returns:
            A tuple (functions, prompts) of validated models.

        Raises:
            ValidationError: If a file has invalid JSON or wrong schema.
            OSError: If a file is missing or cannot be read.
        """
        with open(self._args.functions_definition, "r") as func_file:
            adapter1 = TypeAdapter(list[FunctionValidator])
            functions: list[FunctionValidator] = adapter1.validate_json(
                func_file.read())
        with open(self._args.input, "r") as prompt_file:
            adapter2: TypeAdapter = TypeAdapter(list[PromptValidator])
            prompts: list[PromptValidator] = adapter2.validate_json(
                prompt_file.read())
        return functions, prompts

    def read_vocabs(self) -> None:
        """Load the tokens of the vocabulary file into `self._vocabs`."""
        path: str = self._llm.get_path_to_vocab_file()
        with open(path, "r") as file:
            vocabs: dict[str, int] = load(file)
        for x in vocabs:
            self._vocabs += [x]

    @staticmethod
    def check_output(
            prompt: str, function: str,
            args: dict[str, Any]) -> OutputValidator:
        """Build and validate one output entry.

        Args:
            prompt: The original natural-language request.
            function: The chosen function name.
            args: The extracted arguments.

        Returns:
            The validated output model.

        Raises:
            ValidationError: If the entry does not match the schema.
        """
        return OutputValidator(
            prompt=prompt, name=function, parameters=args)

    def write_output(self, output: list[OutputValidator]) -> None:
        """Write all results to the output JSON file.

        The parent directory is created if it does not exist.

        Args:
            output: The validated results, one per prompt.
        """
        path: Path = Path(self._args.output)
        adapter: TypeAdapter = TypeAdapter(list[OutputValidator])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(adapter.dump_json(output, indent=2).decode())

    def get_tuple(self) -> tuple[
            list[FunctionValidator],
            list[PromptValidator]]:
        """Return the validated (functions, prompts) pair."""
        return self._tuple

    def get_vocabs(self) -> list[str]:
        """Return the list of vocabulary tokens read from the vocab file."""
        return self._vocabs
