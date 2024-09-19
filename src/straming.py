from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyBZNQhwzyjIFMjyvc6T600e5cUiZAU_6zs"

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

prompt = "Escreva um conto científico com base no conto de fadas de A Bela e a Fera."

response = llm.invoke(prompt)

print(response.content)

for block in llm.stream(response.content):
    print(block.content, end = '\n')
    print('-' * 100)