from json import loads, dumps


class Parser():
    def __init__(self) -> None:
        self._file_path1: str = "data/input/function_calling_tests.json"
        self._file_path2: str = "data/input/functions_definition.json"
        self._prompt: list[str]
        self._functions: list[str]

    def functions(self) -> None:
        line_count: int = 0
