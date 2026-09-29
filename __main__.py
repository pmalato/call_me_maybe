from llm_sdk.llm_sdk import Small_LLM_Model
from src.errors import CallMeMaybeErrors
from src.parser import Parser

try:
    id_list: list[int] = []
    llm: Small_LLM_Model = Small_LLM_Model()
    parse: Parser = Parser()
    to_decode: list[int] = []
    parse.read_prompts()
    prompt1: str = parse.get_prompts()[0]
    for i in llm.encode(prompt1)[0]:
        id_list += [int(i)]
    logits: list[float] = llm.get_logits_from_input_ids(id_list)
    for j in range(0, 50):
        value: float = max(logits)
        to_decode += [logits.index(value)]
        logits.remove(value)
    print(llm.decode(to_decode))
except (CallMeMaybeErrors, ImportError) as e:
    print("Error: ", e)
