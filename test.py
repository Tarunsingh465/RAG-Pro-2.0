from app.llm.client import LLMClient

llm = LLMClient()

response = llm.invoke("Say Hello")

print(response)