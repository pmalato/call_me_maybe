from json import load
from typing import Any
from src.errors import (
    MissingSeperatorError,
    PromptFileError,
    DuplicatePromptError,
    FunctionsFileError
    )
from llm_sdk.llm_sdk import Small_LLM_Model


class Parser():
    def __init__(self) -> None:
        self._file_path1: str = "data/input/function_calling_tests.json"
        self._file_path2: str = "data/input/functions_definition.json"
        self._tokenizer: dict[str, Any]
        self._vocabs: dict[int, str] = {}
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

    def read_tokenizer(self, llm: Small_LLM_Model) -> None:
        path: str = llm.get_path_to_tokenizer_file()
        with open(path, "r") as file:
            self._tokenizer = load(file)

    def read_vocabs(self, llm: Small_LLM_Model) -> None:
        path: str = llm.get_path_to_vocab_file()
        temp: dict[str, int]
        with open(path, "r") as file:
            temp = load(file)
            for x in temp:
                self._vocabs |= {temp[x]: x}

    def get_prompts(self) -> list[str]:
        return self._prompt

    def get_functions(self) -> list[dict[str, Any]]:
        return self._functions

    def get_tokenizer(self) -> dict[str, Any]:
        return self._tokenizer

    def get_vocabs(self) -> dict[int, str]:
        return self._vocabs
