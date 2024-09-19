from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyBZNQhwzyjIFMjyvc6T600e5cUiZAU_6zs"

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)

# Preparando o modelo
my_llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Preparando o Prompt Template
my_prompt= PromptTemplate.from_template("Disserte sobre a seguinte sentença: '{topic}' em uma frase.")

# Criando a Chain
chain = LLMChain(
    llm=my_llm,
    prompt=my_prompt,
    verbose=True
)

# Topico
topic = "A IA pode mudar o mundo?"

# Invoking e Retornando

response = chain.invoke(input=topic)
print(response)
