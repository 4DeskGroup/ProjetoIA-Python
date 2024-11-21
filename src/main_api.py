from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import dataset_to_vector
from langchain_google_genai import GoogleGenerativeAI
import logging
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

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

# Baixar recursos do NLTK
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

# Inicializando o SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def load_sentiment_dictionary(file_path):
    sentiment_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word, score = line.strip().split('\t')
            sentiment_dict[word] = int(score)
    return sentiment_dict

sentiment_dict = load_sentiment_dictionary('dados_sentimentos.txt')

def analyze_review_with_custom_dict(review, sentiment_dict):
    words = review.split()
    sentiment_score = 0
    for word in words:
        sentiment_score += sentiment_dict.get(word.lower(), 0)  # Palavra convertida para lowercase
    
    if sentiment_score > 5:
        return "Positivo", sentiment_score
    elif sentiment_score < -5:
        return "Negativo", sentiment_score
    else:
        return "Neutro", sentiment_score


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

    Quando o usuário informar a área de trabalho, hobby, segmento de vida, local de trabalho ou profissão, recomende produtos que estejam alinhados com as necessidades dessa área de trabalho, hobby, segmento de vida, local de trabalho ou profissão

    Se a pergunta tiver relação com a área de trabalho, hobby, segmento de vida, local de trabalho ou profissão, ao formular sua resposta, reforce recomendações que estejam alinhadas com as necessidades dessa área de trabalho, hobby, segmento de vida, local de trabalho ou profissão

    Para responder de forma precisa, considere as reviews e detalhes fornecidos. Inclua recomendações baseadas nas características do produto e nas 
    preferências do usuário. Caso exista um histórico de interações, utilize-o somente se a pergunta atual tiver relação com respostas anteriores, considerando padrões de comportamento e preferências ao longo do tempo.
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

# Função para análise de sentimento em textos longos
def analyze_long_text(text):
    sentences = sent_tokenize(text)
    total_score = 0

    for sentence in sentences:
        _, score = analyze_review_with_custom_dict(sentence, sentiment_dict)
        total_score += score

    if total_score > 5:
        return {"Positiva": total_score}
    elif total_score < -5:
        return {"Negativa": total_score}
    else:
        return {"Neutra": total_score}

# Função que ajusta o retriever chain e inclui o contexto das respostas anteriores
def ask_question(retriever_chain, question):
    try:
        # Construa o contexto das conversas anteriores
        context = build_context_from_history()
        prompt_with_context = f"Contexto:\n{context}\nPergunta atual: {question}"

        # Recebe a resposta do retriever_chain
        response = retriever_chain.invoke({"input": prompt_with_context})

        if 'answer' in response:
            answer = response['answer']

            # Armazena a pergunta e a resposta no histórico
            conversation_history.append({"question": question, "answer": answer})

            # Realiza a análise de sentimento da resposta usando a função para textos longos
            sentiment_analysis = analyze_long_text(answer)
            logger.info(f"Análise de sentimento: {sentiment_analysis}")

            return {
                "answer": answer,
                "sentiment_analysis": sentiment_analysis
            }
        else:
            logger.warning("Nenhuma resposta encontrada ou 'answer' não presente na resposta.")
            return {"answer": "Nenhuma resposta encontrada."}
    except Exception as e:
        logger.error(f"Erro ao tentar responder à pergunta: {e}", exc_info=True)
        return {"answer": "Erro ao processar a pergunta."}

# Rota principal da API que recebe a pergunta e retorna a resposta com a análise de sentimento
@app.post("/ask/")
def ask(request: QuestionRequest):
    if not retriever_chain:
        return {"error": "O sistema não foi inicializado corretamente."}

    question = request.question
    result = ask_question(retriever_chain, question)
    
    return {
        "question": question,
        "answer": result["answer"],
        "sentiment_analysis": result["sentiment_analysis"]
    }

# Rota para limpar o histórico (sem modificar o histórico anterior)
@app.put("/clear")
def clear_history():
    conversation_history.clear()
    return {"Success": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
