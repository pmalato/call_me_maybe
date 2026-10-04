from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from .parser import Parser
from .models import PromptValidator
from .constrained_decoding import ConstrainedDecoder
from time import perf_counter


def main() -> None:
    try:
        start: float = perf_counter()
        llm: Small_LLM_Model = Small_LLM_Model()
        prs: Parser = Parser(llm)
        decoder: ConstrainedDecoder = ConstrainedDecoder(llm, prs)
        prompts: list[PromptValidator] = prs.get_tuple()[1]
        for p in prompts:
            res = decoder.solve_args(
                p.prompt, decoder.solve_function(p.prompt))
            [print(f"{x}: {res[x]}") for x in res]
        end: float = perf_counter()
        print(end - start)
    except (
            KeyboardInterrupt, ValidationError,
            ValueError, ImportError) as error:
        print(error)


if __name__ == "__main__":
    main()
