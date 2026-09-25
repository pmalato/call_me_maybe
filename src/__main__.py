from src.parser import Parser
parse: Parser = Parser()
parse.read_prompts()
test = parse.get_prompts()
for x in test:
    print(x)
