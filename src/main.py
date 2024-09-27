from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import dataset_to_vector
from langchain_google_genai import GoogleGenerativeAI
import logging
from dotenv import load_dotenv

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dynamic_prompt(context_type, question_type):
    template = f"""
    Contexto: {{context}}

    Pergunta ({question_type}): {{input}}
    
    Baseado no contexto fornecido e no tipo de pergunta, forneça uma resposta clara e concisa.
    """
    return ChatPromptTemplate.from_template(template)

def create_specific_prompt(context_type, question_type):
    template = f"""
    Você está recebendo informações sobre ({context_type}). Utilize essas informações para responder à pergunta a seguir:

    Contexto:
    {{context}}
    
    Pergunta ({question_type}):
    {{input}}
    
    Para responder de forma precisa, considere as reviews e detalhes fornecidos. Inclua recomendações baseadas nas características do produto e nas preferências do usuário.Para responder de forma precisa, considere as reviews e detalhes fornecidos.
    """
    return ChatPromptTemplate.from_template(template)

def initialize_retrieval_chain():
    logger.info("Carregando variáveis de ambiente...")
    load_dotenv()

    logger.info("Configurando o prompt...")
    prompt = create_specific_prompt("Geral", "Pergunta sobre os dados do contexto")
    
    llm = GoogleGenerativeAI(model="gemini-pro")
    document_chain = create_stuff_documents_chain(llm, prompt=prompt)

    logger.info("Convertendo dataset para vetores...")
    retriever = dataset_to_vector('ruanchaves/b2w-reviews01', use_saved_embeddings=False)

    if retriever is None:
        logger.error("Retriever não foi criado corretamente.")
        return None

    logger.info("Criando a retrieval chain...")
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    return retriever_chain

def ask_question(retriever_chain, question):
    # logger.info(f"Invocando a chain com a pergunta: {question}")
    try:
        response = retriever_chain.invoke({"input": question})

        if 'answer' in response:
            return response['answer']
        else:
            logger.warning("Nenhuma resposta encontrada ou 'answer' não presente na resposta.")
            return "Nenhuma resposta encontrada."
    except Exception as e:
        logger.error(f"Erro ao tentar responder à pergunta: {e}", exc_info=True)
        return "Erro ao processar a pergunta."

def main():
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