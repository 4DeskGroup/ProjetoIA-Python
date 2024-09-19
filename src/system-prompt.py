from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyBZNQhwzyjIFMjyvc6T600e5cUiZAU_6zs"

my_llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

output = my_llm.invoke(
    [
        SystemMessage(content="Responda somente SIM ou NÃO."),
        HumanMessage(content="Formiga tem proteína?")
    ]
)

print(output.content)