from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from .parser import Parser


def main() -> None:
    try:
        translated: dict[str, float] = {}
        llm: Small_LLM_Model = Small_LLM_Model()
        prs: Parser = Parser()
        file_tuple = prs.get_tuple()
        ids: list[int] = [int(x) for x in llm.encode(file_tuple[1][0].prompt)[0]]
        logits: list[float] = llm.get_logits_from_input_ids(ids)
    except (ValidationError, ValueError, ImportError) as error:
        print(error)


if __name__ == "__main__":
    main()
