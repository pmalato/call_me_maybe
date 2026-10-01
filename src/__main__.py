# from llm_sdk.llm_sdk import Small_LLM_Model
# from pydantic import ValidationError
# from .parser import Parser


# def main() -> None:
#     try:
#         llm: Small_LLM_Model = Small_LLM_Model()
#         prs: Parser = Parser()
#         file_tuple = prs.get_tuple()
#     except (ValidationError, ValueError, ImportError) as error:
#         print(error)


# if __name__ == "__main__":
#     main()
