from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    (
        "system",
        "Você é um assistente que traduz do inglês para o português. Traduza a sentença do usuário.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)

print(ai_msg.content)