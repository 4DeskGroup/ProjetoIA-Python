import os
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Carregar variáveis de ambiente
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Baixar e salvar o arquivo CSV
url = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/4639429ec698d7821fc99a0bc665fa213d9fcd5a/B2W-Reviews01.csv"
response = requests.get(url)
rawdata = response.text

with open("B2W-Reviews01.csv", "w", encoding='utf-8') as f:
    f.write(rawdata)

# Ler o arquivo CSV
data = pd.read_csv("B2W-Reviews01.csv", low_memory=False)

# Remover duplicatas
data = data.drop_duplicates(subset=['product_id', 'product_name'])

documents = []

# Processar cada linha do DataFrame
for _, row in data.head(9).iterrows():
    text = (
        f"Data de Submissão: {row.get('submission_date', 'Não disponível')}\n"
        f"ID do Revisor: {row.get('reviewer_id', 'Não disponível')}\n"
        f"ID do Produto: {row.get('product_id', 'Não disponível')}\n"
        f"Nome do Produto: {row.get('product_name', 'Não disponível')}\n"
        f"Marca do Produto: {row.get('product_brand', 'Não disponível')}\n"
        f"Categoria do Site LV1: {row.get('site_category_lv1', 'Não disponível')}\n"
        f"Categoria do Site LV2: {row.get('site_category_lv2', 'Não disponível')}\n"
        f"Título da Revisão: {row.get('review_title', 'Não disponível')}\n"
        f"Avaliação Geral: {row.get('overall_rating', 'Não disponível')}\n"
        f"Recomendaria a um Amigo: {row.get('recommend_to_a_friend', 'Não disponível')}\n"
        f"Texto da Revisão: {row.get('review_text', 'Não disponível')}\n"
        f"Ano de Nascimento do Revisor: {row.get('reviewer_birth_year', 'Não disponível')}\n"
        f"Gênero do Revisor: {row.get('reviewer_gender', 'Não disponível')}\n"
        f"Estado do Revisor: {row.get('reviewer_state', 'Não disponível')}\n"
    )
    documents.append(Document(page_content=text))

# Dividir os documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=150)  # Aumentar tamanho do chunk
text_chunks = text_splitter.split_documents(documents)

# Remover duplicatas dos chunks
unique_chunks = []
seen_texts = set()
for chunk in text_chunks:
    if chunk.page_content not in seen_texts:
        seen_texts.add(chunk.page_content)
        unique_chunks.append(chunk)

# Criar embeddings e vetorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(unique_chunks, embeddings)
retriever = vectorstore.as_retriever()

# Imprimir chunks após remover duplicatas
print("Chunks após remover duplicatas:")
for chunk in unique_chunks:
    print(chunk.page_content)
    print('-' * 80)

print(f"Total de chunks criados: {len(text_chunks)}")
print(f"Total de chunks únicos: {len(unique_chunks)}")

# Definir template e prompt
template = """Você é um assistente que precisa responder perguntas com base no conteúdo de {{retriever}}.
Use os pedaços de texto retornados como base para suas respostas. Se não souber, diga "Não Posso te Responder Isso :)". 
Se houver mais de uma resposta possível, liste todas.
Pergunta: {question}
Contexto: {context}
Resposta:
"""

prompt = PromptTemplate.from_template(template)
output_parser = StrOutputParser()

# Inicializar modelo e cadeia de processamento
llm_model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.5)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

# Receber e processar a pergunta do usuário
user_question = input("Digite sua Pergunta: ")
print("Pergunta recebida:", user_question)
responseQuestion = rag_chain.invoke(user_question)
print("Resposta gerada:", responseQuestion)
