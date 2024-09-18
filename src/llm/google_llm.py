from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=os.getenv("GEMINI_API_KEY"),  
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
