# from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
from src.errors import CallMeMaybeErrors
from src.parser import Parser

try:
    parse: Parser = Parser()
    parse.read_functions()
    test: list[dict[str, Any]] = parse.get_functions()
    for x in test:
        print(x)
    # llm: Small_LLM_Model = Small_LLM_Model()
    # with open(llm.get_path_to_vocab_file()) as file:
    #     x = load(file)
    # for i in x:
    #     print(i)
except (CallMeMaybeErrors, ImportError) as e:
    print("Error: ", e)
