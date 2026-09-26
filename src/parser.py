from json import load
from typing import Any
from src.errors import (
    MissingSeperatorError,
    PromptFileError,
    DuplicatePromptError,
    FunctionsFileError
    )


class Parser():
    def __init__(self) -> None:
        self._file_path1: str = "data/input/function_calling_tests.json"
        self._file_path2: str = "data/input/functions_definition.json"
        self._prompt: list[str] = []
        self._functions: list[dict[str, Any]] = []

    def read_prompts(self) -> None:
        line_count: int = 0
        text: list[str]
        with open(self._file_path1, "r") as file:
            text = file.readlines()
            for line in text:
                line_count += 1
                line = line.strip()
                if line == '[' or \
                        line == ']' or \
                        line == '{' or \
                        line == '}' or \
                        line == '},':
                    continue
                if not line.startswith('"prompt"'):
                    raise PromptFileError(line_count)
                if ": " not in line:
                    raise MissingSeperatorError(line_count)
                key, value = line.split(": ")
                if key != '"prompt"':
                    raise PromptFileError(line_count)
                if not value.startswith('"') or not value.endswith('"'):
                    raise PromptFileError(line_count)
                if value in self._prompt:
                    raise DuplicatePromptError(line_count)
                self._prompt += [value]

    def read_functions(self) -> None:
        with open(self._file_path2, "r") as file:
            self._functions = load(file)
            for x in self._functions:
                if not x["name"] and \
                        not x["description"] and \
                        not x["parameters"] and \
                        not x["returns"]:
                    raise FunctionsFileError()
                for y in x:
                    if y != "name" and \
                            y != "description" and \
                            y != "parameters" and \
                            y != "returns":
                        raise FunctionsFileError()
                continue

    def get_prompts(self) -> list[str]:
        return self._prompt

    def get_functions(self) -> list[dict[str, Any]]:
        return self._functions
