from llm_sdk.llm_sdk import Small_LLM_Model

def main() -> None:
    try:
        llm: Small_LLM_Model = Small_LLM_Model()
    except (ValueError, ImportError) as error:
        print(error)
