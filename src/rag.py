from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import dataset_to_vector
from main import llm
from dotenv import load_dotenv
import logging

# Configurando logging para melhor rastreamento
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_retrieval_chain():
    logger.info("Carregando variáveis de ambiente...")
    load_dotenv()

    logger.info("Configurando o prompt...")
    prompt = ChatPromptTemplate.from_template("""
    Use as seguintes informações para responder à pergunta de maneira concisa e precisa:
    {context}
    
    Pergunta: {input}
    """)

    logger.info("Criando a document chain...")
    document_chain = create_stuff_documents_chain(llm, prompt=prompt)

    logger.info("Convertendo dataset para vetores...")
    retriever = dataset_to_vector('ruanchaves/b2w-reviews01', use_saved_embeddings=False)

    if retriever is None:
        logger.error("Retriever não foi criado corretamente. Verifique o processo de vetor.")
        return None

    logger.info("Criando a retrieval chain...")
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    return retriever_chain

def ask_question(retriever_chain, question):
    logger.info(f"Invocando a chain com a pergunta: {question}")
    try:
        response = retriever_chain.invoke({"input": question})

        # Verifica se a resposta contém o campo 'answer'
        if 'answer' in response:
            # logger.info("Resposta recebida com sucesso.")
            return response['answer']
        else:
            logger.warning("Nenhuma resposta encontrada ou 'answer' não presente na resposta.")
            return "Nenhuma resposta encontrada."
    except Exception as e:
        logger.error(f"Erro ao tentar responder à pergunta: {e}", exc_info=True)
        return "Erro ao processar a pergunta."

def main():
    # logger.info("Iniciando o processo de inicialização única.")
    retriever_chain = initialize_retrieval_chain()

    if retriever_chain is None:
        logger.error("Falha ao inicializar a chain. Encerrando.")
        return

    logger.info("Sistema pronto para receber perguntas.")
    while True:
        question = input("Faça sua pergunta (ou digite 'sair' para encerrar): ")
        if question.lower() == 'sair':
            logger.info("Encerrando o sistema.")
            break

        answer = ask_question(retriever_chain, question)
        print(f"Resposta: {answer}")

if __name__ == "__main__":
    main()
