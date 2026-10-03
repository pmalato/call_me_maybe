from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from .parser import Parser
from .constrained_decoding import ConstrainedDecoder


def main() -> None:
    try:
        llm: Small_LLM_Model = Small_LLM_Model()
        prs: Parser = Parser(llm)
        decoder: ConstrainedDecoder = ConstrainedDecoder(llm, prs)
        prompts = prs.get_tuple()[1]
        print(decoder.solve_function(prompts[0].prompt))
    except (ValidationError, ValueError, ImportError) as error:
        print(error)


if __name__ == "__main__":
    main()
