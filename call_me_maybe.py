from src.errors import CallMeMaybeErrors
from src.parser import Parser

try:
    parse: Parser = Parser()
    parse.read_prompts()
    test = parse.get_prompts()
    for x in test:
        print(x)
except (CallMeMaybeErrors, ImportError) as e:
    print("Error: ", e)
