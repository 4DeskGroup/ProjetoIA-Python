from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import dataset_to_vector
from langchain_google_genai import GoogleGenerativeAI
import logging
from dotenv import load_dotenv
from vector_memory import *
import re

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
    
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.2)
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
    logger.info(f"Invocando a chain com a pergunta: {question}")

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

    index = initialize_faiss_index()
    embedder = get_embedder_model()

    logger.info("Sistema pronto para receber perguntas.")

    while True:
        question = input("Faça sua pergunta (ou digite 'sair' para encerrar): ")

        # Detectar se o usuário pediu pela n-ésima pergunta ou resposta
        nth_question_match = re.search(r"(\d+)[ªº] pergunta", question.lower())
        nth_answer_match = re.search(r"(\d+)[ªº] resposta", question.lower())

        # Verificar n-ésima pergunta
        if nth_question_match:
            n = int(nth_question_match.group(1))
            nth_question = get_nth_human_message(n)
            if nth_question:
                print(f"Sua {n}ª pergunta foi: {nth_question}")
            else:
                print(f"Você ainda não fez a {n}ª pergunta.")
            continue

        # Verificar n-ésima resposta
        if nth_answer_match:
            n = int(nth_answer_match.group(1))
            nth_answer = get_nth_ai_message(n)
            if nth_answer:
                print(f"A minha {n}ª resposta foi: {nth_answer}")
            else:
                print(f"Eu ainda não forneci a {n}ª resposta.")
            continue

        if question.lower() == 'sair':
            logger.info("Encerrando o sistema.")
            break
        
        # Vetoriza a entrada do usuário
        user_input_vector = embedder.encode([question])

        # Busca no FAISS por perguntas anteriores semelhantes
        search_similar_question(index, user_input_vector)

        # Adiciona a nova pergunta ao histórico de mensagens
        add_human_message(question)

        # Constrói o histórico para ser enviado ao LLM
        formatted_history = build_prompt_from_history() #+ f"Usuário: {question}"

        answer = ask_question(retriever_chain, formatted_history)

        print(f"Resposta: {answer}")

        # Armazena a resposta no histórico de mensagens
        add_ai_message(answer)

        # Vetoriza e armazena a entrada do usuário no FAISS
        add_vector_to_index(index, user_input_vector)

if __name__ == "__main__":
    main()
