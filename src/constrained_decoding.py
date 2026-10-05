from llm_sdk.llm_sdk import Small_LLM_Model
from .models import FunctionValidator, PromptValidator
from .parser import Parser
from typing import Any


MAX_NAME_TOKENS: int = 32
MAX_VALUE_TOKENS: int = 64


class ConstrainedDecoder():
    def __init__(self, llm: Small_LLM_Model, parser: Parser) -> None:
        self._llm: Small_LLM_Model = llm
        self._parser: Parser = parser
        self._functions: list[FunctionValidator] = self._parser.get_tuple()[0]
        self._prompts: list[PromptValidator] = self._parser.get_tuple()[1]
        self._vocabs: list[str] = self._parser.get_vocabs()
        self.adjust_vocabs()
        self._func_ids: list[list[int]] = [[
            int(x) for x in self._llm.encode(
                self._functions[i].name)[0]] for i in range(
                    len(self._functions))]
        self._args_ids: dict[str, set[int]] = {}
        self.define_args_ids()

    def adjust_vocabs(self) -> None:
        """Turns vocabs into valid characters (replacing Ġ and Ċ)"""
        for i in range(len(self._vocabs)):
            self._vocabs[i] = self._llm.decode([i])

    def define_args_ids(self) -> None:
        """Builds sets of tokens which will filter the functions' values"""
        arg_sets: dict[str, set[int]] = {
            "bool": set(), "digits": set(), "str": set(),
            "minus": set(), "dot": set(), "num_end": set(),
            "str_end": set()}
        for i, t in enumerate(self._vocabs):
            if not t or t.startswith("<|"):
                continue
            if all(c in "0123456789" for c in t):
                arg_sets["digits"].add(i)
            elif t == '-':
                arg_sets["minus"].add(i)
            elif t == '.':
                arg_sets["dot"].add(i)
            if t[0] in ",}":
                arg_sets["num_end"].add(i)
            if t[0] == '"':
                arg_sets["str_end"].add(i)
            elif not any(c in '"\\' or ord(c) < 32 for c in t):
                arg_sets["str"].add(i)
            if t in ("true", "false"):
                arg_sets["bool"].add(i)
        self._args_ids = arg_sets

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

    def set_valid_args_logits(self, kind: str, text: str) -> set[int]:
        """Return the token ids allowed next, given the value so far."""
        temp: dict[str, set[int]] = self._args_ids
        if kind == "boolean":
            return temp["bool"]
        if kind == "string":
            return temp["str"] | temp["str_end"]
        allowed: set[int] = set(temp["digits"])
        if text == "":
            allowed |= temp["minus"]
        has_digit: bool = any([c.isdigit() for c in text])
        if kind == "number" and has_digit and "." not in text:
            allowed |= temp["dot"]
        if has_digit and not text.endswith("."):
            allowed |= temp["num_end"]
        return allowed

    def func_tokens_helper(self, function: FunctionValidator) -> str:
        params: str = ", ".join(
            f"{x}: {y.type}" for x, y in function.parameters.items())
        return f"- {function.name}({params}): {function.description}\n"

    def create_func_tokens(self) -> str:
        temp: str = ""
        for x in self._functions:
            temp += self.func_tokens_helper(x)
        return temp

    def append_context(self, prompt: str) -> list[int]:
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

    def is_func_name_over(self, generated: list[int]) -> bool:
        return generated in self._func_ids

    def solve_function(self, prompt: str) -> FunctionValidator:
        ids: list[int] = self.append_context(prompt)
        generated: list[int] = []
        for _ in range(MAX_NAME_TOKENS):
            if self.is_func_name_over(generated):
                break
            logits: list[float] = self._llm.get_logits_from_input_ids(
                ids + generated)
            valid: list[float] = self.set_invalid_func_logits(
                logits, generated)
            generated.append(valid.index(max(valid)))
        for f in self._functions:
            if self._llm.decode(generated) == f.name:
                return f
        raise ValueError("Unable to generate function name.")

    def solve_value(self, ids: list[int], kind: str) -> tuple[str, list[int]]:
        """Generates one value, token by token. Returns its text and ids"""
        text: str = ""
        generated: list[int] = []
        for _ in range(MAX_VALUE_TOKENS):
            allowed: set[int] = self.set_valid_args_logits(kind, text)
            logits: list[float] = self._llm.get_logits_from_input_ids(
                ids + generated)
            best: int = max(allowed, key=lambda t: logits[t])
            if kind == "boolean":
                return self._vocabs[best], [best]
            if best in self._args_ids["num_end"] and kind != "string":
                break
            if best in self._args_ids["str_end"] and kind == "string":
                break
            generated.append(best)
            text += self._vocabs[best]
        return text, generated

    def solve_args(
            self, prompt: str, function: FunctionValidator) -> dict[str, Any]:
        """Fill every parameter of the chosen function, in order."""
        ids: list[int] = self.append_context(prompt)
        ids += [int(x) for x in self._llm.encode(function.name)[0]]
        ids += [int(x) for x in self._llm.encode('", "parameters": {')[0]]
        res: dict[str, Any] = {}
        res_text: str
        gen: list[int]
        for i, (type, arg) in enumerate(function.parameters.items()):
            label: str = ", " if i > 0 else ""
            label += f'"{type}": '
            if arg.type == "string":
                label += '"'
            ids += [int(x) for x in self._llm.encode(label)[0]]
            res_text, gen = self.solve_value(ids, arg.type)
            ids += gen
            if arg.type == "string":
                ids += [int(x) for x in self._llm.encode('"')[0]]
                res[type] = res_text
            elif arg.type == "boolean":
                res[type] = res_text == "true"
            elif arg.type == "integer":
                res[type] = int(res_text)
            else:
                res[type] = float(res_text)
        return res
