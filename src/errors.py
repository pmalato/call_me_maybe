class CallMeMaybeErrors(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MissingSeperatorError(CallMeMaybeErrors):
    def __init__(
            self, lineno: int,
            message: str = "Missing seperator ': '") -> None:
        super().__init__(f"{message}, line {lineno}")


class PromptFileError(CallMeMaybeErrors):
    def __init__(
            self, lineno: int,
            message: str = 'Invalid Prompt. Expected: '
            '"prompt": "prompt_as_str"') -> None:
        super().__init__(f"{message}, line {lineno}")


class DuplicatePromptError(CallMeMaybeErrors):
    def __init__(
            self, lineno: int,
            message: str = "Duplicate Prompt") -> None:
        super().__init__(f"{message}, line {lineno}")


class FunctionsFileError(CallMeMaybeErrors):
    def __init__(self, message: str = "Invalid File Format") -> None:
        super().__init__(message)
