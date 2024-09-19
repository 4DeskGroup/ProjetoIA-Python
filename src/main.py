from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyBZNQhwzyjIFMjyvc6T600e5cUiZAU_6zs"

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)

def generate_cat_name():
    names = llm.invoke("Você tem um gatinho fêmea novo e gostaria de dar um nome. Me dê uma lista de 5 nomes.")

    return names.content

def states_of_brazil():
    states = llm.invoke("Liste todos os estados do Brasil e diga quanto tem ao final da sentença.")

    return states.content

print(states_of_brazil())