from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nltk
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import dataset_to_vector
from langchain_google_genai import GoogleGenerativeAI
import logging
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializando o FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Altere para os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

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
    
    Para responder de forma precisa, considere as reviews e detalhes fornecidos. Inclua recomendações baseadas nas características do produto e nas 
    preferências do usuário, e também baseie sua resposta no histórico gerado das perguntas anteriores, considerando padrões de comportamento e preferências ao longo do tempo.
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

retriever_chain = initialize_retrieval_chain()

# Lista para armazenar o histórico de perguntas e respostas
conversation_history = []

# Função para construir o contexto a partir do histórico de conversas
def build_context_from_history():
    context = ""
    for entry in conversation_history:
        context += f"Pergunta: {entry['question']}\n"
        context += f"Resposta: {entry['answer']}\n"
    return context

# Função que ajusta o retriever chain e inclui o contexto das respostas anteriores
def ask_question(retriever_chain, question):
    try:
        # Construa o contexto das conversas anteriores
        context = build_context_from_history()
        prompt_with_context = f"Contexto:\n{context}\nPergunta atual: {question}"

        response = retriever_chain.invoke({"input": prompt_with_context})

        if 'answer' in response:
            answer = response['answer']
            # Armazena a pergunta e a resposta no histórico
            conversation_history.append({"question": question, "answer": answer})

            sia = SentimentIntensityAnalyzer()
            sentiment_score = sia.polarity_scores(answer)
        
            logger.info(f"Sentiment Score: {sentiment_score}")

            return answer
        else:
            logger.warning("Nenhuma resposta encontrada ou 'answer' não presente na resposta.")
            return "Nenhuma resposta encontrada."
    except Exception as e:
        logger.error(f"Erro ao tentar responder à pergunta: {e}", exc_info=True)
        return "Erro ao processar a pergunta."

# Rota principal da API que recebe a pergunta e retorna a resposta
@app.post("/ask/")
def ask(request: QuestionRequest):
    if not retriever_chain:
        return {"error": "O sistema não foi inicializado corretamente."}

    question = request.question
    answer = ask_question(retriever_chain, question)
    
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)