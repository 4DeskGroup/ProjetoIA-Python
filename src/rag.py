from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import dataset_to_vector
from main import llm
from dotenv import load_dotenv

print("Iniciando rag.py")

try:
    print("Carregando variáveis de ambiente...")
    load_dotenv()
    
    print("Configurando o prompt...")
    prompt = ChatPromptTemplate.from_template(""" Responda a pergunta com base apenas no contexto
    {context}
    Pergunta: {input}                                       
    """)
    
    print("Criando a document chain...")
    document_chain = create_stuff_documents_chain(llm, prompt=prompt)
    
    print("Convertendo dataset para vetores...")
    retriever = dataset_to_vector('ruanchaves/b2w-reviews01')
    
    print("Criando a retrieval chain...")
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    
    print("Invocando a chain com a pergunta...")
    response = retriever_chain.invoke({"input": "Quais perguntas você consegue responder dado o contexto?"})
    
    print("Resposta recebida:")
    print(response['answer'])  # Imprime toda a resposta para verificar a estrutura
    print(response.get('answer', 'Nenhuma resposta encontrada'))
    
except Exception as e:
    print(f"Ocorreu um erro: {e}")