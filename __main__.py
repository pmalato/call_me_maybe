from llm_sdk.llm_sdk import Small_LLM_Model
from src.errors import CallMeMaybeErrors
from src.parser import Parser

try:
    llm: Small_LLM_Model = Small_LLM_Model()
    parse: Parser = Parser(llm)
    parse.read_vocabs()
    parse.read_tokenizer()
    print(len(parse.get_vocabs()))
    print(len(parse.get_tokenizer()["model"]["merges"]))
except (CallMeMaybeErrors, ImportError) as e:
    print("Error: ", e)
