from llm_sdk.llm_sdk import Small_LLM_Model
from .models import FunctionValidator, PromptValidator
from .parser import Parser


MAX_NAME_TOKENS: int = 32


class ConstrainedDecoder():
    def __init__(self, llm: Small_LLM_Model, parser: Parser) -> None:
        self._llm: Small_LLM_Model = llm
        self._parser: Parser = parser
        self._functions: list[FunctionValidator] = self._parser.get_tuple()[0]
        self._prompts: list[PromptValidator] = self._parser.get_tuple()[1]
        self._vocabs: list[str] = self._parser.get_vocabs()
        self._func_ids: list[list[int]]
        self.set_valid_func_tokens()

    def set_valid_func_tokens(self) -> None:
        self._func_ids = [[
            int(x) for x in self._llm.encode(
                self._functions[i].name)[0]] for i in range(
                    len(self._functions))]

    def set_invalid_func_logits(
            self, logits: list[float], generated: list[int]) -> list[float]:
        n: int = len(generated)
        allowed: set = set()
        for x in self._func_ids:
            if x[:n] == generated and len(x) > n:
                allowed.add(x[n])
        mask: list[float] = [float('-inf')] * len(logits)
        for y in allowed:
            mask[y] = logits[y]
        return mask

    def func_tokens_helper(self, function: FunctionValidator) -> str:
        params: str = ", ".join(
            f"{x}: {y.type}" for x, y in function.parameters.items())
        return f"- {function.name}({params}): {function.description}\n"

    def create_func_tokens(self) -> str:
        temp: str = ""
        for x in self._functions:
            temp += self.func_tokens_helper(x)
        return temp

    def is_func_name_over(self, generated: list[int]) -> bool:
        return generated in self._func_ids

    def append_prompt_func(self, prompt: str) -> list[int]:
        text: str = (
            "<|im_start|>system\n"
            "You are a function-calling assistant. Choose the best function "
            "for the user's request and answer only with a JSON function "
            "call. \n\nAvaliable functions:\n"
            + self.create_func_tokens()
            + "<|im_end|>\n<|im_start|>user\n" + prompt
            + "<|im_end|>\n<|im_start|>assistante\n<think>\n\n</think>\n\n"
            + '{"name": "'
        )
        return [int(x) for x in self._llm.encode(text)[0]]

    def solve_function(self, prompt: str) -> str:
        ids: list[int] = self.append_prompt_func(prompt)
        generated: list[int] = []
        for _ in range(MAX_NAME_TOKENS):
            if self.is_func_name_over(generated):
                break
            logits: list[float] = self._llm.get_logits_from_input_ids(
                ids + generated)
            valid: list[float] = self.set_invalid_func_logits(
                logits, generated)
            generated.append(valid.index(max(valid)))
        if not self.is_func_name_over(generated):
            raise ValueError("Unable to generate function name.")
        return self._llm.decode(generated)

    def find_valid_tokens(self) -> None:
        ...

    def sample_valid_tokens(self) -> None:
        ...
