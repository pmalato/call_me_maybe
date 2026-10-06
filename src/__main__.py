from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from typing import Any
from .parser import Parser
from .models import FunctionValidator, PromptValidator, OutputValidator
from .constrained_decoding import ConstrainedDecoder
from time import perf_counter


def main() -> None:
    """Run the function calling pipeline over every input prompt.

    For each prompt, the function is chosen with constrained decoding,
    then its arguments are generated, validated, printed, and finally
    all results are written to the output file. Expected errors are
    reported with a message instead of a traceback.
    """
    try:
        start: float = perf_counter()
        llm: Small_LLM_Model = Small_LLM_Model()
        parser: Parser = Parser(llm)
        decoder: ConstrainedDecoder = ConstrainedDecoder(llm, parser)
        prompts: list[PromptValidator] = parser.get_tuple()[1]
        res: list[OutputValidator] = []
        visual: list[dict[str, Any]] = []
        for i, p in enumerate(prompts):
            function: FunctionValidator = decoder.solve_function(p.prompt)
            args: dict[str, Any] = decoder.solve_args(p.prompt, function)
            output: OutputValidator = parser.check_output(
                p.prompt, function.name, args)
            res += [output]
            visual += [
                {"prompt": output.prompt,
                 "name": output.name,
                 "parameters": output.parameters}
            ]
            [print(f"{x}: {visual[i][x]}") for x in visual[i]]
            print()
        parser.write_output(res)
        end: float = perf_counter()
        print(end - start)
    except (
            KeyboardInterrupt, ValidationError,
            ValueError, ImportError) as error:
        print(error)


if __name__ == "__main__":
    main()
