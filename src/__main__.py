from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from .parser import Parser
from .constrained_decoding import ConstrainedDecoder


def main() -> None:
    try:
        llm: Small_LLM_Model = Small_LLM_Model()
        prs: Parser = Parser(llm)
        decoder: ConstrainedDecoder = ConstrainedDecoder(llm, prs)
        [print(i) for i in decoder.func_prompt_solver()]
    except (ValidationError, ValueError, ImportError) as error:
        print(error)


if __name__ == "__main__":
    main()
