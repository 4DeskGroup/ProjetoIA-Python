from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

def oscar(filme, ano, llm):
    # Use Google Generative AI model
    #llm = GoogleGenerativeAI(model="gemini-pro")
    
    prompt_oscar= PromptTemplate(input_variables=['filme', 'ano'],
                                        template="Quantos oscars o filme {filme} ganhou em {ano}")    
    oscar_chain = LLMChain(llm=llm, prompt=prompt_oscar)
    
    response = oscar_chain({'filme':filme, 'ano':ano})

    return response

llm = GoogleGenerativeAI(model="gemini-pro")

if __name__ == "__main__":
    response =oscar('Minions', '2024', llm=llm) 
    print(response['text'])
