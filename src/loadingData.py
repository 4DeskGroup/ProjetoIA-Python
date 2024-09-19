import os
import requests
import pandas as pd

from dotenv import load_dotenv

# from datasets import load_dataset

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ds = load_dataset("ruanchaves/b2w-reviews01")

url = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/4639429ec698d7821fc99a0bc665fa213d9fcd5a/B2W-Reviews01.csv"
response = requests.get(url)
rawdata = response.text

with open("B2W-Reviews01.csv", "w", encoding='utf-8') as f:
    f.write(rawdata)

data = pd.read_csv("B2W-Reviews01.csv", low_memory = False)

documents = []

for _, row in data.head(20).iterrows():  # Use apenas as primeiras 10 linhas
    text = (
        f"Submission Date: {row['submission_date']}\n"
        f"Reviewer ID: {row['reviewer_id']}\n"
        f"Product ID: {row['product_id']}\n"
        f"Product Name: {row['product_name']}\n"
        f"Product Brand: {row['product_brand']}\n"
        f"Site Category LV1: {row['site_category_lv1']}\n"
        f"Site Category LV2: {row['site_category_lv2']}\n"
        f"Review Title: {row['review_title']}\n"
        f"Overall Rating: {row['overall_rating']}\n"
        f"Recommend to a Friend: {row['recommend_to_a_friend']}\n"
        f"Review Text: {row['review_text']}\n"
        f"Reviewer Birth Year: {row['reviewer_birth_year']}\n"
        f"Reviewer Gender: {row['reviewer_gender']}\n"
        f"Reviewer State: {row['reviewer_state']}\n"
    )
    documents.append(Document(page_content=text))


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
text_chunks = text_splitter.split_documents(documents)

# for i, chunk in enumerate(text_chunks):
#     print(f"Chunk {i+1}:\n{chunk.page_content}\n")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore=FAISS.from_documents(text_chunks, embeddings)
retriever = vectorstore.as_retriever()

template="""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

llm_model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.5)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

print(rag_chain.invoke("Quantos produtos overall_rating igual a 4? Destaque os nomes"))

